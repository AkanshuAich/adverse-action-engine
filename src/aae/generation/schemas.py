"""What the model is asked to return.

Stage one returns a typed selection: which factors are the principal reasons,
what is claimed about the applicant, which provisions are cited. Every field is
checkable against evidence the system already holds, which is what makes
verification deterministic.

Note what is absent. The model does not supply the applicant identifier or the
jurisdiction; both are attached afterwards from the decision. It cannot
therefore issue a notice against the wrong person, and the verifier's
precondition check is left guarding against programming errors rather than
model errors.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aae.domain.models import (
    AdverseActionNotice,
    Citation,
    FactualClaim,
    ReasonStatement,
)


class SelectedReason(BaseModel):
    """One principal reason chosen by the model."""

    model_config = ConfigDict(extra="forbid")

    factor_id: str = Field(description="Must be one of the supplied factor identifiers.")
    text: str = Field(
        min_length=1,
        max_length=400,
        description="One sentence, addressed to the applicant, in plain language.",
    )


class SelectedClaim(BaseModel):
    """A statement of fact about the applicant."""

    model_config = ConfigDict(extra="forbid")

    field_name: str = Field(description="Must be one of the supplied factor identifiers.")
    stated_value: float | str = Field(description="The value as it will appear in the notice.")


class SelectedCitation(BaseModel):
    """A reference to a supplied provision."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    section: str
    quoted_span: str = Field(
        min_length=1,
        description="Copied word for word from the provision text supplied.",
    )


class SelectedNotice(BaseModel):
    """Stage one output: the structured, verifiable selection."""

    model_config = ConfigDict(extra="forbid")

    principal_reasons: list[SelectedReason] = Field(min_length=1)
    factual_claims: list[SelectedClaim] = Field(default_factory=list)
    citations: list[SelectedCitation] = Field(default_factory=list)
    included_elements: list[str] = Field(
        default_factory=list,
        description="Required elements the notice provides.",
    )

    def to_domain(self, application_id: str, jurisdiction: str) -> AdverseActionNotice:
        """Attach the identity the model was never given.

        Args:
            application_id: Taken from the decision, not from the model.
            jurisdiction: Taken from the active jurisdiction, not from the model.

        Returns:
            The domain notice, ready for verification.
        """
        return AdverseActionNotice(
            application_id=application_id,
            jurisdiction=jurisdiction,
            principal_reasons=tuple(
                ReasonStatement(factor_id=reason.factor_id, text=reason.text)
                for reason in self.principal_reasons
            ),
            factual_claims=tuple(
                FactualClaim(field_name=claim.field_name, stated_value=claim.stated_value)
                for claim in self.factual_claims
            ),
            citations=tuple(
                Citation(
                    document_id=citation.document_id,
                    section=citation.section,
                    quoted_span=citation.quoted_span,
                )
                for citation in self.citations
            ),
            declared_elements=frozenset(self.included_elements),
        )


class RenderedBody(BaseModel):
    """Stage two output: the customer-facing letter."""

    model_config = ConfigDict(extra="forbid")

    body: str = Field(
        min_length=1,
        max_length=4_000,
        description="The complete notice, addressed to the applicant.",
    )
