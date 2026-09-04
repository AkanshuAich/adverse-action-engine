"""Assembling a case for human review.

Everything an underwriter needs is reconstructed from the audit chain: the
decision, the factors behind it, the notice that was generated, and why it was
escalated if it was. Nothing is read from a side table.

That is the same claim the project makes to a regulator - that a decision can
be reconstructed years later from the chain alone - so the console exercises
it daily rather than leaving it to be discovered untrue during an audit.

The logic lives here rather than in the Streamlit module so that it can be
tested. A review screen whose correctness depends on clicking through it is a
review screen nobody checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aae.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aae.audit.chain import ChainedRecord
    from aae.audit.repository import AuditRepository

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReviewFactor:
    """One factor behind the decision, as shown to a reviewer."""

    rank: int
    display_name: str
    factor_id: str
    value: float | str | None
    direction: str


@dataclass(frozen=True)
class ReviewReason:
    """One principal reason as recorded in the chain."""

    factor_id: str
    text: str


@dataclass(frozen=True)
class ReviewCase:
    """One decision awaiting sign-off, reconstructed from the chain."""

    decision_id: str
    application_id: str
    decision: str
    probability_default: float
    threshold: float
    model_version: str
    factors: tuple[ReviewFactor, ...]
    reasons: tuple[ReviewReason, ...]
    citations: tuple[tuple[str, str], ...]
    body: str | None
    escalated: bool
    escalation_reason: str | None
    attempts: int
    violations: tuple[str, ...]
    provider: str
    record_count: int

    @property
    def needs_attention(self) -> bool:
        """Whether this case could not be issued without a human."""
        return self.escalated or self.body is None

    @property
    def headline(self) -> str:
        """A one-line summary for the queue."""
        state = "ESCALATED" if self.escalated else "awaiting sign-off"
        return (
            f"{self.application_id} - {state} - "
            f"p(default) {self.probability_default:.1%} - {self.attempts} attempt(s)"
        )


def build_case(records: Sequence[ChainedRecord], decision_id: str) -> ReviewCase | None:
    """Reconstruct a reviewable case from its audit records.

    Args:
        records: Every record for one decision, in chain order.
        decision_id: The correlation id.

    Returns:
        The case, or ``None`` if the records do not contain a scored decision.
    """
    decision_payload: dict[str, Any] | None = None
    generation_payload: dict[str, Any] | None = None

    # Records are identified by the shape of their payload rather than by the
    # event type column. The payload is what a reader of the chain actually
    # has - an export, a regulator's copy - and it should be self-describing
    # without a second table to join against.
    for record in records:
        payload = record.payload
        if "probability_default" in payload and decision_payload is None:
            decision_payload = payload
        if "attempts" in payload and "issued" in payload:
            generation_payload = payload

    if decision_payload is None:
        logger.warning("case_without_decision", decision_id=decision_id)
        return None

    factors = tuple(
        ReviewFactor(
            rank=int(factor["rank"]),
            display_name=str(factor["display_name"]),
            factor_id=str(factor["factor_id"]),
            value=factor.get("value"),
            direction=str(factor["direction"]),
        )
        for factor in decision_payload.get("factors", [])
    )

    generation = generation_payload or {}

    return ReviewCase(
        decision_id=decision_id,
        application_id=str(decision_payload["application_id"]),
        decision=str(decision_payload["decision"]),
        probability_default=float(decision_payload["probability_default"]),
        threshold=float(decision_payload["threshold"]),
        model_version=str(decision_payload["model_version"]),
        factors=factors,
        reasons=tuple(
            ReviewReason(factor_id=str(r.get("factor_id", "")), text=str(r.get("text", "")))
            for r in generation.get("reasons", ())
        ),
        citations=tuple(
            (str(c.get("section", "")), str(c.get("quoted_span", "")))
            for c in generation.get("citations", ())
        ),
        body=generation.get("body"),
        escalated=bool(generation.get("escalated", False)),
        escalation_reason=generation.get("escalation_reason"),
        attempts=int(generation.get("attempts", 0)),
        violations=tuple(generation.get("violations", ())),
        provider=str(generation.get("provider", "unknown")),
        record_count=len(records),
    )


@dataclass
class ReviewQueue:
    """The set of decisions awaiting human sign-off."""

    repository: AuditRepository

    def pending(self, limit: int = 50) -> tuple[ReviewCase, ...]:
        """Return cases awaiting review, oldest first.

        Args:
            limit: Maximum cases to return.

        Returns:
            The cases.
        """
        cases: list[ReviewCase] = []
        for decision_id in self.repository.pending_review(limit=limit):
            case = build_case(self.repository.records_for_decision(decision_id), decision_id)
            if case is not None:
                cases.append(case)
        return tuple(cases)

    def approve(self, case: ReviewCase, reviewer: str, note: str | None = None) -> None:
        """Record that a reviewer approved a notice as generated.

        Args:
            case: The case reviewed.
            reviewer: Who reviewed it.
            note: Their comment, if any.
        """
        self.repository.record_human_review(
            case.decision_id,
            case.application_id,
            reviewer=reviewer,
            action="approved",
            note=note,
        )

    def reject(self, case: ReviewCase, reviewer: str, note: str) -> None:
        """Record that a reviewer rejected a notice.

        Args:
            case: The case reviewed.
            reviewer: Who reviewed it.
            note: Why. Required: a rejection without a reason teaches nobody
                anything and cannot become evaluation data.
        """
        self.repository.record_human_review(
            case.decision_id,
            case.application_id,
            reviewer=reviewer,
            action="rejected",
            note=note,
        )

    def edit(self, case: ReviewCase, reviewer: str, body: str, note: str | None = None) -> None:
        """Record that a reviewer rewrote a notice before issuing it.

        The edit is appended, never applied over the original. Both versions
        stay in the chain, because what the system produced and what was sent
        are different facts and an auditor may want either.

        Args:
            case: The case reviewed.
            reviewer: Who reviewed it.
            body: The revised letter.
            note: Their comment, if any.
        """
        self.repository.record_human_review(
            case.decision_id,
            case.application_id,
            reviewer=reviewer,
            action="edited",
            note=note,
            edited_body=body,
        )
