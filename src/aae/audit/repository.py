"""Reading and writing the audit chain.

The hard part is not writing a row, it is writing the *next* row. Every record
carries the hash of its predecessor, so appending means reading the current
tail and extending it. Two decisions being scored at the same moment would both
read tail ``N`` and both try to write ``N+1``: one succeeds, and the other
either collides on the unique constraint or, worse, silently forks the chain.

Appends are therefore serialised with a Postgres transaction-level advisory
lock. It is held for the duration of the transaction and released on commit or
rollback by the database itself, so a crashed process cannot wedge the chain.
Only appends contend; reads never take the lock.

This is a deliberate trade. Serialising appends caps audit throughput at one
writer at a time, which is the price of a single totally-ordered chain. For a
credit decisioning system that is the right trade: decisions arrive at human
speed, and a chain that can fork is not evidence of anything.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from sqlalchemy import func, select, text

from aae.audit.chain import ChainedRecord, ChainVerification, JsonValue, link, verify_chain
from aae.audit.models import AuditEventType, AuditRecord
from aae.domain.errors import AuditIntegrityError
from aae.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session, sessionmaker

    from aae.domain.models import CreditDecision, VerificationResult

logger = get_logger(__name__)

AUDIT_CHAIN_LOCK_KEY: Final[int] = 0x4141_4541
"""Advisory lock key serialising appends. Arbitrary but fixed; "AAEA" in hex."""


def decision_payload(decision: CreditDecision) -> dict[str, JsonValue]:
    """Render a credit decision as an audit payload.

    Everything needed to reconstruct the outcome is captured: the exact inputs,
    the calibrated probability, the threshold and model version in force, and
    the ranked factors with their contributions. Datetimes become ISO strings
    because the payload is hashed, and hashing requires a canonical form.

    Args:
        decision: The decision to record.

    Returns:
        A JSON-compatible payload.
    """
    return {
        "application_id": decision.application_id,
        "decision": decision.decision.value,
        "probability_default": decision.probability_default,
        "threshold": decision.threshold,
        "model_version": decision.model_version,
        "scored_at": decision.scored_at.astimezone(UTC).isoformat(),
        "feature_values": dict(decision.feature_values),
        "factors": [
            {
                "factor_id": factor.factor_id,
                "display_name": factor.display_name,
                "value": factor.value,
                "shap_value": factor.shap_value,
                "direction": factor.direction.value,
                "rank": factor.rank,
            }
            for factor in decision.factors
        ],
    }


def verification_payload(result: VerificationResult) -> dict[str, JsonValue]:
    """Render a verification outcome as an audit payload.

    Args:
        result: The verifier's decision on a generated notice.

    Returns:
        A JSON-compatible payload.
    """
    return {
        "passed": result.passed,
        "attempt": result.attempt,
        "violations": list(result.rendered_violations()),
    }


class AuditRepository:
    """Append-only access to the audit chain."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Build the repository.

        Args:
            session_factory: Produces sessions bound to the application role.
        """
        self._session_factory = session_factory

    @staticmethod
    def _to_record(row: AuditRecord) -> ChainedRecord:
        return ChainedRecord(
            sequence=row.sequence,
            payload=row.payload,
            prev_hash=row.prev_hash,
            record_hash=row.record_hash,
        )

    def append(
        self,
        *,
        event_type: AuditEventType,
        application_id: str,
        decision_id: str,
        payload: dict[str, JsonValue],
        note: str | None = None,
    ) -> ChainedRecord:
        """Append one record to the chain.

        Args:
            event_type: Which pipeline stage this records.
            application_id: The application concerned.
            decision_id: Correlation id shared by every record for one decision.
            payload: Record content. Must be JSON-compatible.
            note: Optional free text, such as an underwriter comment.

        Returns:
            The record as written, including its chain links.

        Raises:
            AuditIntegrityError: If the write fails.
        """
        return self.append_many(
            [
                (
                    event_type,
                    application_id,
                    decision_id,
                    payload,
                    note,
                )
            ]
        )[0]

    def append_many(
        self,
        entries: Sequence[tuple[AuditEventType, str, str, dict[str, JsonValue], str | None]],
    ) -> tuple[ChainedRecord, ...]:
        """Append several records in one transaction.

        The whole batch shares a single advisory lock and a single commit, so
        a multi-stage decision either lands complete or not at all. A partial
        chain is worse than no chain: it looks like evidence and is not.

        Args:
            entries: Tuples of event type, application id, decision id,
                payload, and optional note, in the order they should be
                chained.

        Returns:
            The written records, in order.

        Raises:
            AuditIntegrityError: If the write fails.
        """
        if not entries:
            return ()

        try:
            with self._session_factory() as session, session.begin():
                # Serialise appends. Held until this transaction ends; released
                # by Postgres on commit or rollback, including on crash.
                session.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": AUDIT_CHAIN_LOCK_KEY},
                )

                tail_row = session.execute(
                    select(AuditRecord).order_by(AuditRecord.sequence.desc()).limit(1)
                ).scalar_one_or_none()
                previous = self._to_record(tail_row) if tail_row is not None else None

                written: list[ChainedRecord] = []
                for event_type, application_id, decision_id, payload, note in entries:
                    record = link(payload, previous)
                    session.add(
                        AuditRecord(
                            sequence=record.sequence,
                            event_type=event_type.value,
                            application_id=application_id,
                            decision_id=decision_id,
                            payload=record.payload,
                            prev_hash=record.prev_hash,
                            record_hash=record.record_hash,
                            note=note,
                        )
                    )
                    written.append(record)
                    previous = record
        # Broad on purpose: every failure mode here - constraint violation,
        # lock timeout, dropped connection - means the same thing to the
        # caller, which is that the decision was not recorded.
        except Exception as exc:
            msg = f"Failed to append {len(entries)} audit record(s): {exc}"
            raise AuditIntegrityError(msg) from exc

        logger.info(
            "audit_records_appended",
            count=len(written),
            first_sequence=written[0].sequence,
            last_sequence=written[-1].sequence,
        )
        return tuple(written)

    def record_decision(
        self,
        decision: CreditDecision,
        decision_id: str,
        *,
        event_type: AuditEventType = AuditEventType.DECISION_SCORED,
    ) -> ChainedRecord:
        """Record a scored decision.

        Args:
            decision: The decision to record.
            decision_id: Correlation id for this decision.
            event_type: Which stage this represents.

        Returns:
            The written record.
        """
        return self.append(
            event_type=event_type,
            application_id=decision.application_id,
            decision_id=decision_id,
            payload=decision_payload(decision),
        )

    def records_for_decision(self, decision_id: str) -> tuple[ChainedRecord, ...]:
        """Return every record for one decision, in chain order.

        Args:
            decision_id: The correlation id to reconstruct.

        Returns:
            The records, ascending by sequence.
        """
        with self._session_factory() as session:
            rows = session.execute(
                select(AuditRecord)
                .where(AuditRecord.decision_id == decision_id)
                .order_by(AuditRecord.sequence)
            ).scalars()
            return tuple(self._to_record(row) for row in rows)

    def all_records(self) -> tuple[ChainedRecord, ...]:
        """Return the entire chain, ascending by sequence.

        Returns:
            Every record ever written.
        """
        with self._session_factory() as session:
            rows = session.execute(select(AuditRecord).order_by(AuditRecord.sequence)).scalars()
            return tuple(self._to_record(row) for row in rows)

    def verify(self) -> ChainVerification:
        """Verify the integrity of the whole chain.

        Returns:
            Whether the chain is intact, and where it first breaks if not.
        """
        result = verify_chain(self.all_records())
        if not result.intact:
            logger.error(
                "audit_chain_broken",
                broken_at=result.broken_at,
                reason=result.reason,
            )
        return result

    def count(self) -> int:
        """Return how many records the chain holds.

        Returns:
            The record count.
        """
        with self._session_factory() as session:
            return int(session.execute(select(func.count()).select_from(AuditRecord)).scalar_one())

    def tail(self) -> ChainedRecord | None:
        """Return the most recent record.

        Returns:
            The chain tail, or ``None`` if the chain is empty.
        """
        with self._session_factory() as session:
            row = session.execute(
                select(AuditRecord).order_by(AuditRecord.sequence.desc()).limit(1)
            ).scalar_one_or_none()
            return self._to_record(row) if row is not None else None

    def pending_review(self, limit: int = 50) -> tuple[str, ...]:
        """Return decisions that have a notice but no human sign-off yet.

        Derived from the chain rather than from a separate queue table. A
        queue that can disagree with the audit log is a second source of truth
        about what happened, and the whole point of the log is that there is
        only one.

        Args:
            limit: Maximum decisions to return, oldest first.

        Returns:
            Correlation ids awaiting review.
        """
        awaiting = {
            AuditEventType.NOTICE_VERIFIED.value,
            AuditEventType.ESCALATED_TO_HUMAN.value,
        }

        with self._session_factory() as session:
            rows = session.execute(
                select(
                    AuditRecord.decision_id, AuditRecord.event_type, AuditRecord.sequence
                ).order_by(AuditRecord.sequence)
            ).all()

        needs_review: dict[str, int] = {}
        reviewed: set[str] = set()
        for decision_id, event_type, sequence in rows:
            if event_type in awaiting:
                needs_review.setdefault(decision_id, sequence)
            elif event_type == AuditEventType.HUMAN_REVIEWED.value:
                reviewed.add(decision_id)

        ordered = sorted(
            (decision_id for decision_id in needs_review if decision_id not in reviewed),
            key=lambda decision_id: needs_review[decision_id],
        )
        return tuple(ordered[:limit])

    def record_human_review(
        self,
        decision_id: str,
        application_id: str,
        *,
        reviewer: str,
        action: str,
        note: str | None = None,
        edited_body: str | None = None,
    ) -> ChainedRecord:
        """Record an underwriter's decision on a generated notice.

        The reviewer, the action, and any edit are appended to the same chain
        as the machine steps. A human overriding the system is part of the
        decision's history, not an annotation beside it.

        Args:
            decision_id: Which decision was reviewed.
            application_id: The application concerned.
            reviewer: Who reviewed it.
            action: What they did - approved, rejected, or edited.
            note: Their comment, if any.
            edited_body: The revised letter, when they changed it.

        Returns:
            The written record.
        """
        payload: dict[str, JsonValue] = {
            "reviewer": reviewer,
            "action": action,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "edited": edited_body is not None,
        }
        if edited_body is not None:
            payload["edited_body"] = edited_body

        return self.append(
            event_type=AuditEventType.HUMAN_REVIEWED,
            application_id=application_id,
            decision_id=decision_id,
            payload=payload,
            note=note,
        )

    def new_decision_id(self) -> str:
        """Generate a correlation id for a new decision.

        Returns:
            A time-ordered identifier, readable in logs and sortable.
        """
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        return f"DEC-{stamp}"
