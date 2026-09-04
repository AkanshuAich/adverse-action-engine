"""What the harness measures, and how each number is defined.

Definitions matter more than the arithmetic here, because a metric whose
meaning is vague is worse than no metric: it gets quoted.

``groundedness_rate`` is measured on the **first** attempt, before any repair.
That is the honest measure of how often the model gets it right unaided.
Reporting the post-repair figure as groundedness would credit the verifier's
work to the model.

``prohibited_content_rate`` is measured over **issued** notices only, and must
be zero. A model proposing a prohibited reason is not a failure of the system -
it is the case the check exists for, and it is reported separately as evidence
the check fires. What must never happen is one reaching an applicant.

``escalation_rate`` counts only cases the system could not make truthful.
Provider failures are excluded and counted apart: folding an outage into this
number would make a network problem look like the model degrading, and this
is the number people would watch to notice exactly that.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aae.generation.graph import GenerationOutcome

FACTOR_GROUNDING: Final[str] = "factor_grounding"
CITATION_VALIDITY: Final[str] = "citation_validity"
ELEMENT_COVERAGE: Final[str] = "element_coverage"
PROHIBITED_CONTENT: Final[str] = "prohibited_content"


def _rate(numerator: int, denominator: int) -> float:
    """Return a proportion, or zero when nothing was measured."""
    return round(numerator / denominator, 4) if denominator else 0.0


@dataclass(frozen=True)
class CaseOutcome:
    """One golden-set case, reduced to what the metrics need.

    Attributes:
        application_id: Which application this was.
        issued: Whether a notice could be sent without human intervention.
        escalated: Whether it went to a human instead.
        attempts: How many generation attempts were made.
        first_attempt_passed: Whether the model got it right unaided.
        first_attempt_reasons: Reasons proposed on the first attempt.
        first_attempt_citations: Citations proposed on the first attempt.
        first_attempt_codes: Violation codes on the first attempt.
        all_violation_codes: Codes across every attempt, for the caught counts.
        elements_required: How many elements the jurisdiction demands.
        prohibited_in_issued: Whether an issued notice contained a prohibited
            reference. Independently re-checked, not inferred from the pipeline
            having passed it.
        readability: Flesch reading ease of the issued letter.
        latency_ms: Wall-clock time for the whole case.
    """

    application_id: str
    issued: bool
    escalated: bool
    attempts: int
    first_attempt_passed: bool
    first_attempt_reasons: int
    first_attempt_citations: int
    first_attempt_codes: tuple[str, ...]
    all_violation_codes: tuple[str, ...]
    elements_required: int
    prohibited_in_issued: bool
    readability: float | None
    latency_ms: float

    @classmethod
    def from_outcome(
        cls,
        outcome: GenerationOutcome,
        *,
        application_id: str,
        elements_required: int,
        prohibited_in_issued: bool,
        readability: float | None,
        latency_ms: float,
    ) -> CaseOutcome:
        """Reduce a generation outcome to a measured case.

        Args:
            outcome: The result of running the workflow.
            application_id: Which application this was.
            elements_required: How many elements the jurisdiction demands.
            prohibited_in_issued: Result of an independent re-check.
            readability: Flesch score of the issued letter, if one exists.
            latency_ms: Wall-clock time for the case.

        Returns:
            The reduced case.
        """
        select_steps = [s for s in outcome.trace if s.node == "select"]
        verify_steps = [s for s in outcome.trace if s.node == "verify"]
        first_verify = verify_steps[0] if verify_steps else None

        return cls(
            application_id=application_id,
            issued=outcome.issued,
            escalated=outcome.escalated,
            attempts=outcome.attempts,
            first_attempt_passed=first_verify is not None and not first_verify.violation_codes,
            first_attempt_reasons=select_steps[0].reasons if select_steps else 0,
            first_attempt_citations=select_steps[0].citations if select_steps else 0,
            first_attempt_codes=first_verify.violation_codes if first_verify else (),
            all_violation_codes=tuple(
                code for step in outcome.trace for code in step.violation_codes
            ),
            elements_required=elements_required,
            prohibited_in_issued=prohibited_in_issued,
            readability=readability,
            latency_ms=latency_ms,
        )


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregate results over a golden set."""

    cases: int
    groundedness_rate: float
    post_repair_rate: float
    escalation_rate: float
    factor_fidelity: float
    citation_precision: float
    element_coverage: float
    prohibited_content_rate: float
    prohibited_attempts_caught: int
    mean_attempts: float
    readability_mean: float
    readability_below_floor: float
    latency_p50_ms: float
    latency_p95_ms: float
    provider_failures: int = 0
    violations_by_code: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_cases(
        cls, cases: Sequence[CaseOutcome], *, provider_failures: int = 0, readability_floor: float
    ) -> EvalMetrics:
        """Aggregate measured cases.

        Args:
            cases: The measured cases.
            provider_failures: Cases abandoned because the backend failed.
                Counted apart from escalations on purpose.
            readability_floor: Score below which a letter reads as dense.

        Returns:
            The aggregate metrics.
        """
        if not cases:
            return cls(
                cases=0,
                groundedness_rate=0.0,
                post_repair_rate=0.0,
                escalation_rate=0.0,
                factor_fidelity=0.0,
                citation_precision=0.0,
                element_coverage=0.0,
                prohibited_content_rate=0.0,
                prohibited_attempts_caught=0,
                mean_attempts=0.0,
                readability_mean=0.0,
                readability_below_floor=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                provider_failures=provider_failures,
            )

        issued = [case for case in cases if case.issued]
        latencies = sorted(case.latency_ms for case in cases)
        scores = [case.readability for case in issued if case.readability is not None]

        reasons_total = sum(case.first_attempt_reasons for case in cases)
        citations_total = sum(case.first_attempt_citations for case in cases)
        elements_total = sum(case.elements_required for case in cases)

        counts: dict[str, int] = {}
        for case in cases:
            for code in case.all_violation_codes:
                counts[code] = counts.get(code, 0) + 1

        def first_attempt(code: str) -> int:
            return sum(case.first_attempt_codes.count(code) for case in cases)

        return cls(
            cases=len(cases),
            groundedness_rate=_rate(
                sum(1 for case in cases if case.first_attempt_passed), len(cases)
            ),
            post_repair_rate=_rate(len(issued), len(cases)),
            escalation_rate=_rate(sum(1 for case in cases if case.escalated), len(cases)),
            factor_fidelity=(
                round(1.0 - first_attempt(FACTOR_GROUNDING) / reasons_total, 4)
                if reasons_total
                else 0.0
            ),
            citation_precision=(
                round(1.0 - first_attempt(CITATION_VALIDITY) / citations_total, 4)
                if citations_total
                else 0.0
            ),
            element_coverage=(
                round(1.0 - first_attempt(ELEMENT_COVERAGE) / elements_total, 4)
                if elements_total
                else 0.0
            ),
            prohibited_content_rate=_rate(
                sum(1 for case in issued if case.prohibited_in_issued), len(issued)
            ),
            prohibited_attempts_caught=counts.get(PROHIBITED_CONTENT, 0),
            mean_attempts=round(statistics.fmean(case.attempts for case in cases), 3),
            readability_mean=round(statistics.fmean(scores), 2) if scores else 0.0,
            readability_below_floor=_rate(
                sum(1 for score in scores if score < readability_floor), len(scores)
            ),
            latency_p50_ms=round(_percentile(latencies, 0.50), 2),
            latency_p95_ms=round(_percentile(latencies, 0.95), 2),
            provider_failures=provider_failures,
            violations_by_code=dict(sorted(counts.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-compatible data for the committed report.

        Returns:
            The metrics as a mapping.
        """
        return asdict(self)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Return a percentile from pre-sorted values, without interpolation.

    Nearest-rank rather than interpolated: with a golden set of a hundred
    cases, an interpolated p95 invents a value between two real measurements,
    and a latency figure should be one that actually occurred.

    Args:
        sorted_values: Values in ascending order.
        fraction: Percentile as a fraction, such as 0.95.

    Returns:
        The value at that rank, or zero if there are none.
    """
    if not sorted_values:
        return 0.0
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]
