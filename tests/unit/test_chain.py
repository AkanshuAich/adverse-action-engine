"""Tests for the audit hash chain.

The property tests are the point. Example-based tests show the chain works on
the cases we thought of; the property tests assert that *no* alteration to
*any* record can go undetected, which is the claim the audit log actually makes.
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from aae.audit.chain import (
    GENESIS_HASH,
    ChainedRecord,
    canonical_payload,
    compute_record_hash,
    link,
    verify_chain,
)

# JSON-compatible scalars. NaN and infinity are excluded because canonical
# serialisation rejects them.
_scalars = st.one_of(
    st.text(max_size=40),
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.none(),
)
_payloads = st.dictionaries(st.text(min_size=1, max_size=20), _scalars, max_size=8)


def _chain(payloads: list[dict[str, object]]) -> list[ChainedRecord]:
    records: list[ChainedRecord] = []
    previous: ChainedRecord | None = None
    for payload in payloads:
        previous = link(payload, previous)  # type: ignore[arg-type]
        records.append(previous)
    return records


class TestCanonicalPayload:
    def test_key_order_does_not_matter(self):
        assert canonical_payload({"a": 1, "b": 2}) == canonical_payload({"b": 2, "a": 1})

    def test_output_is_valid_json(self):
        payload = {"z": 1, "a": [1, 2], "n": None}
        assert json.loads(canonical_payload(payload)) == payload

    def test_no_incidental_whitespace(self):
        assert canonical_payload({"a": 1, "b": 2}) == '{"a":1,"b":2}'


class TestLinking:
    def test_first_record_uses_genesis_hash(self):
        record = link({"event": "created"}, None)
        assert record.prev_hash == GENESIS_HASH
        assert record.sequence == 0

    def test_each_record_links_to_its_predecessor(self):
        first = link({"n": 1}, None)
        second = link({"n": 2}, first)
        assert second.prev_hash == first.record_hash
        assert second.sequence == 1

    def test_identical_payloads_get_different_hashes_at_different_positions(self):
        first = link({"n": 1}, None)
        second = link({"n": 1}, first)
        assert first.record_hash != second.record_hash


class TestVerifyChain:
    def test_empty_chain_is_intact(self):
        assert verify_chain([]).intact

    def test_well_formed_chain_is_intact(self):
        result = verify_chain(_chain([{"n": i} for i in range(10)]))
        assert result.intact
        assert result.checked == 10
        assert result.broken_at is None

    def test_altered_payload_is_detected(self):
        records = _chain([{"n": i} for i in range(5)])
        tampered = [
            *records[:2],
            ChainedRecord(
                sequence=records[2].sequence,
                payload={"n": 999},
                prev_hash=records[2].prev_hash,
                record_hash=records[2].record_hash,
            ),
            *records[3:],
        ]
        result = verify_chain(tampered)
        assert not result.intact
        assert result.broken_at == 2
        assert "altered" in (result.reason or "")

    def test_removed_record_is_detected(self):
        records = _chain([{"n": i} for i in range(5)])
        result = verify_chain([*records[:2], *records[3:]])
        assert not result.intact
        assert result.broken_at == 3

    def test_reordered_records_are_detected(self):
        records = _chain([{"n": i} for i in range(4)])
        reordered = [records[0], records[2], records[1], records[3]]
        assert not verify_chain(reordered).intact


class TestChainProperties:
    @settings(max_examples=200)
    @given(st.lists(_payloads, max_size=12))
    def test_any_chain_built_by_link_verifies(self, payloads: list[dict[str, object]]):
        assert verify_chain(_chain(payloads)).intact

    @settings(max_examples=200)
    @given(_payloads, _payloads)
    def test_hash_is_a_function_of_the_canonical_form(
        self, left: dict[str, object], right: dict[str, object]
    ):
        """Two payloads hash alike exactly when they serialise alike.

        Note this is deliberately stated over the canonical form rather than
        Python equality: ``{"x": False} == {"x": 0}`` is true in Python but the
        two serialise differently, and the audit log must keep them distinct.
        See ``test_boolean_is_not_conflated_with_integer_zero``.
        """
        left_hash = compute_record_hash(left, GENESIS_HASH)  # type: ignore[arg-type]
        right_hash = compute_record_hash(right, GENESIS_HASH)  # type: ignore[arg-type]
        same_serialisation = canonical_payload(left) == canonical_payload(right)  # type: ignore[arg-type]
        assert (left_hash == right_hash) == same_serialisation

    def test_boolean_is_not_conflated_with_integer_zero(self):
        """``False`` and ``0`` must produce different hashes.

        Python treats them as equal, so a naive implementation would let
        "verification passed: false" be swapped for "passed: 0" without
        breaking the chain. An auditor reading the record needs those to be
        distinguishable, so canonical JSON keeps them apart.
        """
        as_bool = compute_record_hash({"passed": False}, GENESIS_HASH)
        as_int = compute_record_hash({"passed": 0}, GENESIS_HASH)
        assert as_bool != as_int
        assert canonical_payload({"passed": False}) == '{"passed":false}'
        assert canonical_payload({"passed": 0}) == '{"passed":0}'

    @settings(max_examples=200)
    @given(st.lists(_payloads, min_size=1, max_size=8), st.data())
    def test_no_payload_alteration_survives_verification(
        self, payloads: list[dict[str, object]], data: st.DataObject
    ):
        """Replacing any record payload with a different one must be detected."""
        records = _chain(payloads)
        index = data.draw(st.integers(min_value=0, max_value=len(records) - 1))
        target = records[index]
        replacement = data.draw(_payloads.filter(lambda p: p != target.payload))

        tampered = [
            *records[:index],
            ChainedRecord(
                sequence=target.sequence,
                payload=replacement,
                prev_hash=target.prev_hash,
                record_hash=target.record_hash,
            ),
            *records[index + 1 :],
        ]
        result = verify_chain(tampered)
        assert not result.intact
        assert result.broken_at == index
