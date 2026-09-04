"""Core domain types.

This module is deliberately pure: no I/O, no database, no HTTP, no framework
imports. Everything the verifier reasons about is defined here, which is what
makes the verifier trivially testable and property-testable.

The central design decision lives in this file. The language model does not
write prose that we then try to fact-check. It emits a typed
:class:`AdverseActionNotice`, every field of which is checkable against ground
truth the system already holds. Prose is rendered from the verified object.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    """The outcome of a credit application."""

    APPROVE = "approve"
    DECLINE = "decline"


class FactorDirection(StrEnum):
    """Whether a factor pushed the score toward decline or approval."""

    ADVERSE = "adverse"
    FAVOURABLE = "favourable"


class ViolationCode(StrEnum):
    """The six independent verifier checks.

    Each value names a way a generated notice can be wrong. A notice passes
    only when no check produces a violation.
    """

    FACTOR_GROUNDING = "factor_grounding"
    VALUE_ACCURACY = "value_accuracy"
    CITATION_VALIDITY = "citation_validity"
    ELEMENT_COVERAGE = "element_coverage"
    PROHIBITED_CONTENT = "prohibited_content"
    REASON_COUNT = "reason_count"


class _Frozen(BaseModel):
    """Base for immutable domain values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Factor(_Frozen):
    """One feature's contribution to a single decision, from SHAP.

    This is ground truth. Every reason a notice gives must correspond to one
    of these, with a matching direction.
    """

    factor_id: str = Field(description="Stable identifier, e.g. the feature name.")
    display_name: str = Field(description="Human-readable name for the notice.")
    value: float | str | None = Field(description="The applicant's actual value.")
    shap_value: float = Field(description="Signed contribution to the log-odds.")
    direction: FactorDirection
    rank: int = Field(ge=1, description="Rank 1 is the strongest contributor.")


class CreditDecision(_Frozen):
    """The scoring result for one application."""

    application_id: str
    probability_default: float = Field(ge=0.0, le=1.0)
    decision: Decision
    threshold: float = Field(ge=0.0, le=1.0)
    model_version: str
    feature_values: dict[str, float | str | None] = Field(
        description="The exact inputs scored, for audit reconstruction."
    )
    factors: tuple[Factor, ...] = Field(description="Top-K factors, rank ascending.")
    scored_at: datetime

    def adverse_factors(self) -> tuple[Factor, ...]:
        """Return only the factors that pushed the score toward decline.

        Returns:
            Adverse factors, in rank order.
        """
        return tuple(f for f in self.factors if f.direction is FactorDirection.ADVERSE)

    def factor_by_id(self, factor_id: str) -> Factor | None:
        """Look up a factor by its identifier.

        Args:
            factor_id: The identifier to find.

        Returns:
            The matching factor, or ``None`` if this decision has no such factor.
        """
        return next((f for f in self.factors if f.factor_id == factor_id), None)


class FactualClaim(_Frozen):
    """A checkable assertion the notice makes about the applicant.

    Verification compares ``stated_value`` against the real feature value in
    :attr:`CreditDecision.feature_values`.
    """

    field_name: str = Field(description="Must match a key in feature_values.")
    stated_value: float | str = Field(description="The value as asserted in the notice.")


class ReasonStatement(_Frozen):
    """One principal reason for the decline, tied to a real factor."""

    factor_id: str = Field(description="Must match a Factor on the decision.")
    text: str = Field(min_length=1, description="Plain-language statement of the reason.")


class Citation(_Frozen):
    """A reference to a provision of the governing regulation."""

    document_id: str = Field(description="Corpus document identifier.")
    section: str = Field(description="Section or clause reference.")
    quoted_span: str = Field(
        min_length=1,
        description="Text quoted verbatim from the corpus chunk; checked by substring match.",
    )


class AdverseActionNotice(_Frozen):
    """Stage-one output: the structured, verifiable notice.

    The language model fills this schema. It contains no free prose beyond the
    individual reason statements, every one of which is anchored to a factor.
    """

    application_id: str
    jurisdiction: str
    principal_reasons: tuple[ReasonStatement, ...] = Field(min_length=1)
    factual_claims: tuple[FactualClaim, ...] = ()
    citations: tuple[Citation, ...] = ()
    declared_elements: frozenset[str] = Field(
        default=frozenset(),
        description="Required legal elements the model asserts are present.",
    )


class RenderedNotice(_Frozen):
    """Stage-two output: customer-facing prose, constrained to a verified notice."""

    notice: AdverseActionNotice
    body: str = Field(min_length=1)


class Violation(_Frozen):
    """One specific way a notice failed verification."""

    code: ViolationCode
    detail: str = Field(min_length=1, description="What was wrong, specifically.")
    locator: str | None = Field(
        default=None, description="Which reason, claim, or citation was at fault."
    )

    def render(self) -> str:
        """Render the violation as a single line for prompts and audit records.

        Returns:
            A compact human- and model-readable description.
        """
        where = f" [{self.locator}]" if self.locator else ""
        return f"{self.code.value}{where}: {self.detail}"


class VerificationResult(_Frozen):
    """The outcome of running every check against a notice."""

    passed: bool
    violations: tuple[Violation, ...] = ()
    attempt: int = Field(default=1, ge=1)

    def rendered_violations(self) -> list[str]:
        """Render all violations for feedback into a repair prompt.

        Returns:
            One string per violation.
        """
        return [v.render() for v in self.violations]
