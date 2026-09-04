"""Constructing what a language model is allowed to see.

This is an allowlist, not a redactor. The distinction matters: a redactor
inspects a payload and removes what it recognises as sensitive, so anything it
fails to recognise is disclosed. Here the payload is *built* from a fixed set
of permitted fields, so a value can only reach a provider by being named. New
data added to the decision is excluded until someone deliberately lists it.

Three things are withheld even though the notice needs them.

The **applicant identifier** is never sent. The model returns reasons and
citations; the identifier is attached afterwards from the decision itself. A
model that never sees an id cannot attribute a notice to the wrong person, so
the failure the verifier's precondition check exists to catch becomes
structurally impossible to cause by generation.

**Protected attributes** are absent from the feature set already, and are
re-excluded here rather than assumed. Defence in depth is cheap; unlawful
discrimination is not.

**Raw SHAP magnitudes** are sent as direction and rank only. The log-odds
figure is meaningless to an applicant and inviting a model to paraphrase it
invites it into the letter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from aae.domain.errors import GenerationError
from aae.ml.features import DEFAULT_SPEC, PROTECTED_ATTRIBUTES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aae.domain.models import CreditDecision
    from aae.jurisdiction.base import Jurisdiction
    from aae.retrieval.corpus import Provision

ALLOWED_FEATURE_FIELDS: Final[frozenset[str]] = (
    frozenset(DEFAULT_SPEC.feature_names) - PROTECTED_ATTRIBUTES
)
"""The only applicant data that may be described to a model."""

MAX_TEXT_VALUE_LENGTH: Final[int] = 120

_CONTROL_CHARACTERS: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitise_text_value(value: str) -> str:
    """Make a categorical value safe to embed in a prompt.

    Strips control characters and caps length. Values here come from a fixed
    vocabulary in the dataset rather than from applicant free text, so this is
    a guard against a future field rather than a live threat - but the moment
    an applicant-supplied string enters the payload, it is the only thing
    standing between them and the instruction block.

    Args:
        value: The raw categorical value.

    Returns:
        The sanitised value.
    """
    cleaned = _CONTROL_CHARACTERS.sub("", value).strip()
    if len(cleaned) > MAX_TEXT_VALUE_LENGTH:
        cleaned = cleaned[:MAX_TEXT_VALUE_LENGTH] + "..."
    return cleaned


@dataclass(frozen=True)
class PayloadFactor:
    """One factor as described to the model.

    Attributes:
        factor_id: The identifier the model must cite in a reason.
        display_name: Plain-language name, suitable for a customer letter.
        value: The applicant's value for this factor.
        rank: Position by contribution, 1 being strongest.
    """

    factor_id: str
    display_name: str
    value: float | str | None
    rank: int


@dataclass(frozen=True)
class PayloadProvision:
    """One provision of regulation, quoted verbatim for citation."""

    document_id: str
    section: str
    title: str
    text: str


@dataclass(frozen=True)
class GenerationPayload:
    """Everything a model is permitted to see for one notice.

    Deliberately carries no applicant identifier.
    """

    factors: tuple[PayloadFactor, ...]
    provisions: tuple[PayloadProvision, ...]
    jurisdiction_name: str
    max_principal_reasons: int
    required_element_keys: tuple[str, ...]
    prohibited_description: str

    def factor_ids(self) -> frozenset[str]:
        """The identifiers a reason may legitimately cite."""
        return frozenset(factor.factor_id for factor in self.factors)


def build_payload(
    decision: CreditDecision,
    jurisdiction: Jurisdiction,
    provisions: Sequence[Provision],
) -> GenerationPayload:
    """Assemble the permitted view of a decision.

    Only adverse factors are included. A model given the favourable ones will
    sooner or later offer one as a reason for declining, and while the verifier
    catches that, not presenting the temptation is cheaper than repairing it.

    Args:
        decision: The declined decision to explain.
        jurisdiction: The governing rules.
        provisions: Regulation retrieved for this notice.

    Returns:
        The payload.

    Raises:
        GenerationError: If the decision has no adverse factors to explain, or
            if a protected attribute reached this point.
    """
    adverse = decision.adverse_factors()
    if not adverse:
        msg = (
            f"Decision for {decision.application_id} has no adverse factors, so there "
            "is nothing to give as a reason for declining it."
        )
        raise GenerationError(msg)

    leaked = {factor.factor_id for factor in adverse} & PROTECTED_ATTRIBUTES
    if leaked:  # pragma: no cover - the feature layer makes this unreachable
        msg = f"Protected attributes reached the generation payload: {sorted(leaked)}"
        raise GenerationError(msg)

    factors = tuple(
        PayloadFactor(
            factor_id=factor.factor_id,
            display_name=factor.display_name,
            value=(
                sanitise_text_value(factor.value) if isinstance(factor.value, str) else factor.value
            ),
            rank=factor.rank,
        )
        for factor in adverse
        if factor.factor_id in ALLOWED_FEATURE_FIELDS
    )

    if not factors:
        msg = "No adverse factor survived the allowlist; nothing can be explained."
        raise GenerationError(msg)

    return GenerationPayload(
        factors=factors,
        provisions=tuple(
            PayloadProvision(
                document_id=provision.document_id,
                section=provision.section,
                title=provision.title,
                text=provision.text,
            )
            for provision in provisions
        ),
        jurisdiction_name=jurisdiction.name,
        max_principal_reasons=jurisdiction.max_principal_reasons,
        required_element_keys=tuple(sorted(jurisdiction.required_keys)),
        prohibited_description=jurisdiction.prohibited_description,
    )
