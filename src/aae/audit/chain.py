"""Hash chaining for the audit log.

Every audit record carries the hash of its predecessor, so the log is
tamper-evident: altering any historical record invalidates every hash that
follows it. This is deliberately pure — no database, no I/O — so the integrity
logic can be property-tested exhaustively and reused by any storage backend.

Immutability is enforced twice, independently:

* here, cryptographically, so tampering is *detectable*; and
* in the database, by granting the application role ``INSERT`` and ``SELECT``
  but not ``UPDATE`` or ``DELETE``, so tampering is *prevented*.

Neither mechanism relies on application code behaving well.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type JsonValue = str | bool | int | float | list[JsonValue] | dict[str, JsonValue] | None
"""A value that can be canonically serialised.

Uses PEP 695 syntax because the alias is recursive and Pydantic cannot resolve
an implicit recursive alias. ``bool`` precedes ``int`` so that booleans are not
coerced to integers during validation. Callers convert datetimes to ISO strings
before hashing.
"""

GENESIS_HASH: Final[str] = "0" * 64
"""The ``prev_hash`` of the first record in a chain."""

_HASH_PATTERN: Final[str] = r"^[0-9a-f]{64}$"


def canonical_payload(payload: Mapping[str, JsonValue]) -> str:
    """Serialise a payload deterministically.

    Keys are sorted and separators are tight, so the same logical payload always
    produces the same bytes regardless of dictionary insertion order or the
    Python version.

    Args:
        payload: JSON-compatible mapping. Datetimes must already be ISO strings.

    Returns:
        The canonical JSON representation.

    Raises:
        TypeError: If the payload contains a non-JSON-serialisable value.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def compute_record_hash(payload: Mapping[str, JsonValue], prev_hash: str) -> str:
    """Compute the hash binding a record to its predecessor.

    Args:
        payload: The record content.
        prev_hash: The ``record_hash`` of the preceding record, or
            :data:`GENESIS_HASH` for the first record in the chain.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    material = f"{prev_hash}\n{canonical_payload(payload)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ChainedRecord(BaseModel):
    """A record as stored, carrying its chain links."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=0, description="Position in the chain, starting at 0.")
    payload: dict[str, JsonValue]
    prev_hash: str = Field(pattern=_HASH_PATTERN)
    record_hash: str = Field(pattern=_HASH_PATTERN)

    def recomputed_hash(self) -> str:
        """Recompute this record hash from its own payload and link.

        Returns:
            What :attr:`record_hash` should be if the record is untampered.
        """
        return compute_record_hash(self.payload, self.prev_hash)


def link(payload: Mapping[str, JsonValue], previous: ChainedRecord | None) -> ChainedRecord:
    """Build the next record in a chain.

    Args:
        payload: Content of the new record.
        previous: The current tail of the chain, or ``None`` to start one.

    Returns:
        The new record, hashed and linked.
    """
    prev_hash = GENESIS_HASH if previous is None else previous.record_hash
    sequence = 0 if previous is None else previous.sequence + 1
    return ChainedRecord(
        sequence=sequence,
        payload=dict(payload),
        prev_hash=prev_hash,
        record_hash=compute_record_hash(payload, prev_hash),
    )


class ChainVerification(BaseModel):
    """The outcome of verifying a chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intact: bool
    checked: int = Field(ge=0)
    broken_at: int | None = Field(
        default=None, description="Sequence number of the first bad record, if any."
    )
    reason: str | None = None


def verify_chain(records: Sequence[ChainedRecord]) -> ChainVerification:
    """Verify that a chain is internally consistent and unbroken.

    Checks, in order, that sequence numbers are contiguous from zero, that each
    record links to its predecessor, and that every stored hash matches a fresh
    recomputation of its payload.

    Args:
        records: The chain, in ascending sequence order.

    Returns:
        The verification outcome. An empty chain is intact.
    """
    expected_prev = GENESIS_HASH
    for index, record in enumerate(records):
        if record.sequence != index:
            return ChainVerification(
                intact=False,
                checked=index,
                broken_at=record.sequence,
                reason=f"expected sequence {index}, found {record.sequence}",
            )
        if record.prev_hash != expected_prev:
            return ChainVerification(
                intact=False,
                checked=index,
                broken_at=record.sequence,
                reason="prev_hash does not match the preceding record_hash",
            )
        if record.recomputed_hash() != record.record_hash:
            return ChainVerification(
                intact=False,
                checked=index,
                broken_at=record.sequence,
                reason="record_hash does not match the payload; record was altered",
            )
        expected_prev = record.record_hash

    return ChainVerification(intact=True, checked=len(records))
