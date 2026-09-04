"""A provider that fails in realistic ways, on purpose and reproducibly.

**What this measures, and what it does not.** Running the harness against this
provider measures how the *system* handles a fixed distribution of model
mistakes: whether the verifier catches them, whether the repair prompt is
specific enough to fix them, whether anything ungrounded escapes. It does not
measure how good any real model is. Those are different questions and
reporting one as the other would be exactly the dishonesty this project exists
to argue against.

It exists because the alternative is worse. A gate that calls a live model
cannot run in CI - no credential, rate limits, and a number that moves for
reasons unrelated to the change under review. A gate that calls a model
scripted to always succeed measures nothing. This produces a stable,
adversarial input distribution, so a change that weakens the verifier or makes
the repair prompt vaguer shows up as a regression in a number.

The headline figures for a real backend come from running the same harness
with `--provider cerebras`, which is a manual step and is reported separately.

Determinism comes from seeding on a hash of the prompt: the same prompt always
draws the same failures, on any machine, so a metric that moves did so because
the system changed.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel

from aae.domain.errors import ProviderError
from aae.generation.schemas import (
    RenderedBody,
    SelectedCitation,
    SelectedClaim,
    SelectedNotice,
    SelectedReason,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

SIMULATOR_NAME: Final[str] = "simulated"

_FACTOR_ID: Final[re.Pattern[str]] = re.compile(r'"factor_id":\s*"([^"]+)"')
_APPLICANT_VALUE: Final[re.Pattern[str]] = re.compile(
    r'"factor_id":\s*"([^"]+)",\s*"name":\s*"[^"]*",\s*"applicant_value":\s*([^,\n]+)'
)
_DOCUMENT_ID: Final[re.Pattern[str]] = re.compile(r'"document_id":\s*"([^"]+)"')
_SECTION: Final[re.Pattern[str]] = re.compile(r'"section":\s*"([^"]+)"')
_PROVISION_TEXT: Final[re.Pattern[str]] = re.compile(r'"text":\s*"([^"]+)"')
_MAX_REASONS: Final[re.Pattern[str]] = re.compile(r"Maximum principal reasons:\s*(\d+)")
_REQUIRED: Final[re.Pattern[str]] = re.compile(r"Required elements:\s*(.+)")

REPAIR_MARKER: Final[str] = "REJECTED"

INVENTED_FACTORS: Final[tuple[str, ...]] = (
    "SOCIAL_MEDIA_SENTIMENT",
    "DEVICE_RISK_SCORE",
    "APPLICATION_CHANNEL",
    "BUREAU_INQUIRY_COUNT",
)

PROHIBITED_SENTENCES: Final[tuple[str, ...]] = (
    "Applicants of your age default more frequently.",
    "Your marital status was taken into account.",
    "As a female applicant your profile carries more risk.",
)


@dataclass(frozen=True)
class FailureProfile:
    """How often the simulated model makes each kind of mistake.

    The defaults are deliberately pessimistic - considerably worse than a
    capable model on a well-constrained prompt. A gate calibrated on an
    optimistic distribution proves nothing when the real thing degrades.
    """

    invent_factor: float = 0.16
    favourable_factor: float = 0.06
    misquote_citation: float = 0.10
    fabricate_citation: float = 0.05
    misstate_value: float = 0.08
    omit_element: float = 0.07
    exceed_reason_cap: float = 0.04
    prohibited_content: float = 0.03
    prose_invents_figure: float = 0.05
    transport_failure: float = 0.0
    """Left at zero by default. Transport failures are an operational concern
    and are exercised by their own unit test; mixing them into a quality
    distribution would blur what the metrics mean."""

    repair_success: float = 0.82
    """How often a repair attempt actually fixes what it was told about.

    Not 1.0. A model that always fixes everything on request would make the
    repair loop look infallible and hide the case the escalation path exists
    for.
    """


PERFECT: Final[FailureProfile] = FailureProfile(
    invent_factor=0.0,
    favourable_factor=0.0,
    misquote_citation=0.0,
    fabricate_citation=0.0,
    misstate_value=0.0,
    omit_element=0.0,
    exceed_reason_cap=0.0,
    prohibited_content=0.0,
    prose_invents_figure=0.0,
)
"""A model that never errs, for asserting the harness reports a clean run."""


@dataclass(frozen=True)
class PromptFacts:
    """What the simulator reads out of the prompt.

    Parsing the prompt rather than being handed the payload is deliberate: it
    is the same information a real model has, so a prompt that omits something
    the model needs shows up here as the simulator being unable to comply.
    """

    factor_ids: tuple[str, ...]
    factor_values: dict[str, float | str | None]
    document_id: str
    sections: tuple[str, ...]
    provision_texts: tuple[str, ...]
    max_reasons: int
    required_elements: tuple[str, ...]


def parse_prompt(user_message: str) -> PromptFacts:
    """Extract the facts a model would work from.

    Args:
        user_message: The select or repair prompt.

    Returns:
        The parsed facts.

    Raises:
        ProviderError: If the prompt omits the factors or provisions a model
            would need. That is a defect in prompt construction, and it should
            surface loudly rather than as poor metrics.
    """
    factor_ids = tuple(dict.fromkeys(_FACTOR_ID.findall(user_message)))
    documents = _DOCUMENT_ID.findall(user_message)
    sections = tuple(dict.fromkeys(_SECTION.findall(user_message)))

    if not factor_ids:
        msg = "Prompt contained no factor identifiers; a model could not comply."
        raise ProviderError(msg)
    if not documents or not sections:
        msg = "Prompt contained no citable provisions; a model could not comply."
        raise ProviderError(msg)

    values: dict[str, float | str | None] = {}
    for factor_id, raw in _APPLICANT_VALUE.findall(user_message):
        stripped = raw.strip()
        if stripped in {"null", "None"}:
            values[factor_id] = None
        elif stripped.startswith('"'):
            values[factor_id] = stripped.strip('"')
        else:
            try:
                values[factor_id] = float(stripped)
            except ValueError:
                values[factor_id] = None

    cap = _MAX_REASONS.search(user_message)
    required = _REQUIRED.search(user_message)

    return PromptFacts(
        factor_ids=factor_ids,
        factor_values=values,
        document_id=documents[0],
        sections=sections,
        provision_texts=tuple(_PROVISION_TEXT.findall(user_message)),
        max_reasons=int(cap.group(1)) if cap else 4,
        required_elements=(
            tuple(part.strip() for part in required.group(1).split(",")) if required else ()
        ),
    )


def _quotable_span(text: str) -> str:
    """Take a verbatim phrase from a provision, as a compliant model would."""
    words = text.split()
    return " ".join(words[3:11]) if len(words) > 11 else text


class SimulatedProvider:
    """A model that errs at configured rates, reproducibly."""

    def __init__(self, profile: FailureProfile = FailureProfile()) -> None:  # noqa: B008
        """Build the simulator.

        Args:
            profile: How often each mistake occurs.
        """
        self._profile = profile
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        """Provider name recorded in the audit trail."""
        return SIMULATOR_NAME

    @property
    def model(self) -> str:
        """Model name recorded alongside it."""
        return f"failure-profile-{self._profile.invent_factor:.2f}"

    @staticmethod
    def _rng(user: str) -> random.Random:
        """Seed deterministically on the prompt, so runs are reproducible."""
        seed = int.from_bytes(hashlib.blake2b(user.encode(), digest_size=8).digest(), "big")
        return random.Random(seed)  # noqa: S311 - simulation, not cryptography

    def complete[ModelT: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float = 0.0,
    ) -> ModelT:
        """Produce a response of the requested shape.

        Args:
            system: Unused; recorded for completeness.
            user: The prompt, which is parsed for the facts a model would use.
            schema: Either the structured selection or the rendered body.
            temperature: Ignored; the simulator is deterministic by design.

        Returns:
            The generated object.

        Raises:
            ProviderError: If a transport failure is configured, or the prompt
                omits what a model would need.
        """
        self.calls.append(user)
        rng = self._rng(user)

        if rng.random() < self._profile.transport_failure:
            msg = f"{self.name}: simulated transport failure"
            raise ProviderError(msg)

        if schema is RenderedBody:
            return schema.model_validate(self._render(user, rng).model_dump())

        return schema.model_validate(self._select(user, rng).model_dump())

    def _select(self, user: str, rng: random.Random) -> SelectedNotice:
        facts = parse_prompt(user)
        repairing = REPAIR_MARKER in user

        # On a repair, a model usually - not always - fixes what it was told
        # about. The residual failure is what the escalation path is for.
        if repairing and rng.random() < self._profile.repair_success:
            return self._compliant(facts)

        return self._with_failures(facts, rng)

    def _compliant(self, facts: PromptFacts) -> SelectedNotice:
        reasons = [
            SelectedReason(
                factor_id=factor_id,
                text=f"Your {factor_id.replace('_', ' ').lower()} did not meet our requirement.",
            )
            for factor_id in facts.factor_ids[: facts.max_reasons]
        ]
        return SelectedNotice(
            principal_reasons=reasons,
            factual_claims=[],
            citations=[
                SelectedCitation(
                    document_id=facts.document_id,
                    section=facts.sections[0],
                    quoted_span=_quotable_span(facts.provision_texts[0]),
                )
            ],
            included_elements=list(facts.required_elements),
        )

    def _with_failures(self, facts: PromptFacts, rng: random.Random) -> SelectedNotice:
        notice = self._compliant(facts)
        reasons = list(notice.principal_reasons)
        claims: list[SelectedClaim] = []
        citations = list(notice.citations)
        elements = list(notice.included_elements)

        if rng.random() < self._profile.invent_factor:
            reasons.append(
                SelectedReason(
                    factor_id=rng.choice(INVENTED_FACTORS),
                    text="Our wider assessment of your profile raised concerns.",
                )
            )

        if rng.random() < self._profile.favourable_factor:
            # Names a plausible column the payload did not offer, which is what
            # citing a favourable factor looks like from the model's side.
            reasons.append(
                SelectedReason(
                    factor_id="AMT_INCOME_TOTAL",
                    text="Your income was a factor in this decision.",
                )
            )

        if rng.random() < self._profile.exceed_reason_cap:
            reasons.extend(
                SelectedReason(
                    factor_id=factor_id,
                    text=f"Your {factor_id.replace('_', ' ').lower()} also counted against you.",
                )
                for factor_id in facts.factor_ids
            )

        if rng.random() < self._profile.prohibited_content and reasons:
            reasons[0] = SelectedReason(
                factor_id=reasons[0].factor_id,
                text=rng.choice(PROHIBITED_SENTENCES),
            )

        if rng.random() < self._profile.misstate_value:
            numeric = [
                (name, value)
                for name, value in facts.factor_values.items()
                if isinstance(value, float)
            ]
            if numeric:
                name, value = rng.choice(numeric)
                claims.append(SelectedClaim(field_name=name, stated_value=value * 2.7 + 11.0))

        if rng.random() < self._profile.fabricate_citation:
            citations = [
                SelectedCitation(
                    document_id=facts.document_id,
                    section="99.9",
                    quoted_span="a lender may decline at its sole discretion",
                )
            ]
        elif rng.random() < self._profile.misquote_citation:
            citations = [
                SelectedCitation(
                    document_id=facts.document_id,
                    section=facts.sections[0],
                    quoted_span="the lender shall communicate its reasoning promptly",
                )
            ]

        if rng.random() < self._profile.omit_element and len(elements) > 1:
            elements.pop(rng.randrange(len(elements)))

        return SelectedNotice(
            principal_reasons=reasons,
            factual_claims=claims,
            citations=citations,
            included_elements=elements,
        )

    def _render(self, user: str, rng: random.Random) -> RenderedBody:
        sentences = re.findall(r"^\d+\.\s*(.+)$", user, flags=re.MULTILINE)
        reasons = (
            " ".join(sentences) if sentences else "Your application did not meet our criteria."
        )

        body = (
            "Dear applicant,\n\n"
            "We are unable to approve your application at this time. "
            f"{reasons}\n\n"
            "If you would like to discuss this, please contact our grievance "
            "officer, who will review your case.\n\n"
            "Yours sincerely,\nCredit Operations"
        )

        if rng.random() < self._profile.prose_invents_figure:
            body += "\n\nOur records show a monthly obligation of 47,250 against your income."

        return RenderedBody(body=body)


def profile_from_name(name: str) -> FailureProfile:
    """Look up a named failure profile.

    Args:
        name: ``default`` or ``perfect``.

    Returns:
        The profile.

    Raises:
        ValueError: If the name is unknown.
    """
    profiles = {"default": FailureProfile(), "perfect": PERFECT}
    if name not in profiles:
        msg = f"Unknown failure profile {name!r}; choose from {sorted(profiles)}."
        raise ValueError(msg)
    return profiles[name]


def describe_profile(profile: FailureProfile) -> Sequence[tuple[str, float]]:
    """Render a profile for the report header.

    Args:
        profile: The profile in force.

    Returns:
        Name and rate pairs, in declaration order.
    """
    return tuple(
        (field, getattr(profile, field))
        for field in (
            "invent_factor",
            "favourable_factor",
            "misquote_citation",
            "fabricate_citation",
            "misstate_value",
            "omit_element",
            "exceed_reason_cap",
            "prohibited_content",
            "prose_invents_figure",
            "repair_success",
        )
    )
