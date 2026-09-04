"""SQLAlchemy models for the audit log.

One table, append-only. Every stage of a decision writes a record here, and each
record is cryptographically linked to its predecessor by
:mod:`aae.audit.chain`.

The append-only guarantee has two independent enforcers. This module and the
chain module make tampering *detectable*; the migration that creates this table
grants the application role ``INSERT`` and ``SELECT`` but not ``UPDATE`` or
``DELETE``, which makes it *impossible* through the application connection.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import BigInteger, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for every ORM model in this package."""


class AuditEventType(StrEnum):
    """The kinds of event recorded in the audit log.

    Values are stored as text so that adding an event type is a code change
    rather than a database migration.
    """

    APPLICATION_RECEIVED = "application_received"
    DECISION_SCORED = "decision_scored"
    FACTORS_EXPLAINED = "factors_explained"
    NOTICE_GENERATED = "notice_generated"
    NOTICE_VERIFIED = "notice_verified"
    NOTICE_REJECTED = "notice_rejected"
    ESCALATED_TO_HUMAN = "escalated_to_human"
    HUMAN_REVIEWED = "human_reviewed"


class AuditRecord(Base):
    """One immutable entry in the hash chain.

    Attributes are written once and never updated. ``sequence`` is contiguous
    from zero and defines chain order; ``record_hash`` covers both the payload
    and ``prev_hash``, so altering any historical row invalidates every hash
    after it.
    """

    __tablename__ = "audit_record"
    __table_args__ = (
        # Reconstructing a single decision is the dominant read pattern:
        # "show me every step that led to this outcome, in order".
        Index("ix_audit_record_decision_sequence", "decision_id", "sequence"),
        Index("ix_audit_record_application_id", "application_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    sequence: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        unique=True,
        doc="Position in the chain, contiguous from zero.",
    )
    event_type: Mapped[str] = mapped_column(
        String(64), nullable=False, doc="An AuditEventType value."
    )
    application_id: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_id: Mapped[str] = mapped_column(
        String(64), nullable=False, doc="Correlation id shared by every record for one decision."
    )

    # Typed as ``Any`` rather than ``JsonValue`` deliberately: SQLAlchemy resolves
    # mapped annotations at runtime and cannot evaluate the recursive PEP 695
    # alias. The storage contract here is simply JSONB; the precise value type is
    # enforced where it matters, in :mod:`aae.audit.chain`, before hashing.
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        doc="Canonical record content: model version, feature values, SHAP output, "
        "prompt and response hashes, verifier result, human sign-off.",
    )

    prev_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, doc="record_hash of the preceding record, or 64 zeroes."
    )
    record_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, doc="SHA-256 over prev_hash and the payload."
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="Server clock, not the application clock.",
    )

    note: Mapped[str | None] = mapped_column(
        Text, nullable=True, doc="Optional free text, e.g. an underwriter comment."
    )

    def __repr__(self) -> str:
        """Return a concise developer representation.

        Returns:
            A string naming the sequence, event type, and decision.
        """
        return (
            f"AuditRecord(sequence={self.sequence}, "
            f"event_type={self.event_type!r}, decision_id={self.decision_id!r})"
        )
