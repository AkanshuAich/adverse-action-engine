"""Request and response models for the decision API.

Field names match the Home Credit column names exactly. They are not idiomatic
JSON, and that is deliberate: the same identifier appears in the dataset, the
feature specification, the SHAP attribution, and the audit record, so keeping
one spelling end to end means a reviewer tracing a decision never has to
translate between naming conventions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aae.domain.models import CreditDecision, Decision, FactorDirection


class ApplicationRequest(BaseModel):
    """A credit application submitted for a decision.

    Protected attributes are deliberately absent. They may not lawfully
    influence the outcome, so requiring an applicant to supply them in order
    to be scored would be indefensible. Disparate impact is measured across
    populations, not collected per request.
    """

    model_config = ConfigDict(extra="forbid")

    SK_ID_CURR: int = Field(description="Application identifier.")

    AMT_INCOME_TOTAL: float = Field(gt=0, description="Total annual income.")
    AMT_CREDIT: float = Field(gt=0, description="Loan amount requested.")
    AMT_ANNUITY: float | None = Field(default=None, gt=0, description="Annual repayment.")
    AMT_GOODS_PRICE: float | None = Field(default=None, gt=0, description="Goods value.")

    DAYS_EMPLOYED: int = Field(
        description="Days employed, negative. 365243 is the not-employed sentinel."
    )
    DAYS_REGISTRATION: float = Field(le=0, description="Days since registration, negative.")
    DAYS_ID_PUBLISH: float = Field(le=0, description="Days since ID issued, negative.")

    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: float = Field(ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)

    CNT_CHILDREN: int = Field(ge=0)
    CNT_FAM_MEMBERS: float = Field(ge=1)

    REGION_POPULATION_RELATIVE: float = Field(gt=0)
    REGION_RATING_CLIENT: int = Field(ge=1, le=3)

    NAME_CONTRACT_TYPE: str
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_HOUSING_TYPE: str
    OCCUPATION_TYPE: str | None = None
    FLAG_OWN_CAR: str = Field(pattern="^[YN]$")
    FLAG_OWN_REALTY: str = Field(pattern="^[YN]$")


class FactorResponse(BaseModel):
    """One factor behind a decision."""

    factor_id: str
    display_name: str = Field(description="Plain language, suitable for a customer notice.")
    value: float | str | None = Field(description="The applicant's actual value.")
    contribution: float = Field(
        description=(
            "Signed contribution in log-odds, from SHAP. Positive counts against "
            "the applicant. Never quoted to an applicant as a probability."
        )
    )
    direction: FactorDirection
    rank: int = Field(ge=1, description="Rank 1 is the strongest contributor.")


class DecisionResponse(BaseModel):
    """The outcome of scoring an application."""

    decision_id: str = Field(description="Correlation id; use it to fetch the audit trail.")
    application_id: str
    decision: Decision
    probability_default: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    model_version: str
    scored_at: str
    factors: tuple[FactorResponse, ...]
    audit_sequence: int = Field(description="Position of this decision in the audit chain.")

    @classmethod
    def from_decision(
        cls, decision: CreditDecision, decision_id: str, audit_sequence: int
    ) -> DecisionResponse:
        """Build a response from a domain decision.

        Args:
            decision: The scored decision.
            decision_id: Correlation id assigned to it.
            audit_sequence: Chain position of the audit record just written.

        Returns:
            The API response.
        """
        return cls(
            decision_id=decision_id,
            application_id=decision.application_id,
            decision=decision.decision,
            probability_default=decision.probability_default,
            threshold=decision.threshold,
            model_version=decision.model_version,
            scored_at=decision.scored_at.isoformat(),
            factors=tuple(
                FactorResponse(
                    factor_id=factor.factor_id,
                    display_name=factor.display_name,
                    value=factor.value,
                    contribution=factor.shap_value,
                    direction=factor.direction,
                    rank=factor.rank,
                )
                for factor in decision.factors
            ),
            audit_sequence=audit_sequence,
        )


class AuditRecordResponse(BaseModel):
    """One entry from the audit chain."""

    sequence: int
    prev_hash: str
    record_hash: str
    payload: dict[str, object]


class AuditTrailResponse(BaseModel):
    """Every recorded step for one decision, with its integrity verdict."""

    decision_id: str
    records: tuple[AuditRecordResponse, ...]
    chain_intact: bool = Field(
        description="Whether the whole chain verifies, not merely these records."
    )


class ChainVerificationResponse(BaseModel):
    """The result of verifying the entire audit chain."""

    intact: bool
    records_checked: int
    broken_at: int | None = None
    reason: str | None = None


class HealthResponse(BaseModel):
    """Liveness and current model state."""

    status: str
    model_version: str
    threshold: float
    audit_records: int
    chain_intact: bool


class NoticeReasonResponse(BaseModel):
    """One principal reason as it appears in the notice."""

    factor_id: str
    text: str


class NoticeCitationResponse(BaseModel):
    """One regulatory citation, verified against the corpus."""

    document_id: str
    section: str
    quoted_span: str


class NoticeResponse(BaseModel):
    """The outcome of generating an adverse action notice."""

    decision_id: str
    application_id: str
    probability_default: float = Field(ge=0, le=1)
    issued: bool = Field(description="Whether the notice may be sent without human intervention.")
    escalated: bool
    escalation_reason: str | None = None
    attempts: int = Field(ge=1, description="Generation attempts made before this outcome.")
    provider: str
    model: str
    body: str | None = Field(
        default=None,
        description="The customer-facing letter. Absent when escalated: nothing "
        "may be sent whose content could not be verified.",
    )
    reasons: tuple[NoticeReasonResponse, ...] = ()
    citations: tuple[NoticeCitationResponse, ...] = ()
    violations: tuple[str, ...] = Field(
        default=(),
        description="Why verification failed, when it did.",
    )
    audit_sequences: tuple[int, ...] = Field(
        description="Positions in the audit chain of the records written."
    )
