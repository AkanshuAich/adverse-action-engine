"""Prompt construction.

The prompts do not ask the model to be truthful; they constrain what it is
able to say and then the verifier checks the result. That ordering is
deliberate. Instructions reduce how often a model invents a factor, they do not
prevent it, and a system whose safety rests on the model having followed its
instructions has no safety property at all - only a hope.

What the prompts *are* for is making the checkable answer the easy one: the
factor list is enumerated, the provisions are quoted in full, and the required
elements are named, so a compliant response needs no invention.

Repair prompts carry the verifier's own violation text back to the model
unchanged. Paraphrasing them into something friendlier loses the locator that
says which reason was wrong.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aae.generation.payload import GenerationPayload

SELECT_SYSTEM: Final[str] = """\
You draft the reasons section of a credit adverse action notice for a regulated \
lender. Your output is checked mechanically against the lender's records before \
anyone sees it, and any statement that does not match those records is rejected.

Rules, all of which are enforced:

1. Every principal reason MUST cite a factor_id from the supplied list, exactly \
as written. You may not name any other factor. Every factor supplied counted \
AGAINST this application; there are no favourable factors in the list.
2. Give at most the stated maximum number of principal reasons. Choose the \
strongest, which are the lowest-ranked numbers.
3. Every factual claim MUST use a field_name from the supplied factors and a \
stated_value matching the value supplied. The figures are already rounded for \
presentation: copy them exactly as given and do not restate, extend or \
recalculate them.
4. Every citation MUST quote the supplied provision text word for word. Copy a \
phrase from it. Never write a quotation from memory.
5. NEVER refer to the applicant's sex, gender, age, marital status, race, \
religion, caste, national origin, pregnancy or disability. None of these \
influenced the decision and mentioning any of them is unlawful.
6. Write to the applicant in plain, respectful language. One sentence per \
reason. Do not quote internal scores or model outputs.

Return JSON only, matching the requested schema."""

RENDER_SYSTEM: Final[str] = """\
You write the final letter for a credit adverse action notice.

You are given a set of reasons that has already been verified against the \
lender's records. Your task is presentation only.

Rules, all of which are enforced:

1. State every supplied reason. Introduce no other reason, and no fact that is \
not in the material supplied.
2. Do not add figures, dates, or details of any kind that were not supplied.
3. NEVER refer to the applicant's sex, gender, age, marital status, race, \
religion, caste, national origin, pregnancy or disability.
4. Include a clear statement that the application was declined, and tell the \
applicant how to seek clarification or escalate.
5. Plain, respectful language. No jargon, no internal scores, no model names.

Return JSON only, matching the requested schema."""


def _as_written(value: float | str | None) -> float | int | str | None:
    """Present a figure the way it would be written down.

    A whole number carries no fractional part in a letter: an income is
    136,800, not 136800.0. JSON has no way to say "a float that happens to be
    integral", so the trailing zero survives serialisation and the model copies
    it faithfully into the prose.

    Args:
        value: The rounded payload value.

    Returns:
        An ``int`` for a whole number, and the value unchanged otherwise.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def build_select_user_message(payload: GenerationPayload) -> str:
    """Build the stage-one user message.

    Args:
        payload: The permitted view of the decision.

    Returns:
        The message.
    """
    factors = [
        {
            "factor_id": factor.factor_id,
            "name": factor.display_name,
            "applicant_value": _as_written(factor.value),
            "rank": factor.rank,
        }
        for factor in payload.factors
    ]
    provisions = [
        {
            "document_id": provision.document_id,
            "section": provision.section,
            "title": provision.title,
            "text": provision.text,
        }
        for provision in payload.provisions
    ]

    return (
        f"Jurisdiction: {payload.jurisdiction_name}\n"
        f"Maximum principal reasons: {payload.max_principal_reasons}\n"
        f"Required elements: {', '.join(payload.required_element_keys)}\n\n"
        "Factors that counted against this application, strongest first:\n"
        f"{json.dumps(factors, indent=2)}\n\n"
        "Provisions you may cite. Quote from these exactly:\n"
        f"{json.dumps(provisions, indent=2)}\n\n"
        "Select the principal reasons and return the structured notice. In "
        "included_elements, list only the required elements your output actually "
        "provides."
    )


def build_repair_user_message(
    payload: GenerationPayload, violations: Sequence[str], attempt: int
) -> str:
    """Build a message asking the model to fix a rejected notice.

    Args:
        payload: The permitted view of the decision, repeated in full so the
            model is not asked to remember it.
        violations: The verifier's own rendered violations, unaltered.
        attempt: Which attempt this is, so the model knows it is repeating.

    Returns:
        The message.
    """
    listed = "\n".join(f"- {violation}" for violation in violations)
    return (
        f"{build_select_user_message(payload)}\n\n"
        f"Your previous attempt (number {attempt - 1}) was REJECTED by automated "
        f"checks for these specific reasons:\n{listed}\n\n"
        "Correct every one of them. Each refers to a rule above. If a reason "
        "cited a factor that is not in the list, remove it and use one that is. "
        "If a quotation did not appear in a provision, copy the wording from the "
        "provision text above instead."
    )


def build_render_user_message(payload: GenerationPayload, reasons: Sequence[str]) -> str:
    """Build the stage-two user message.

    Only the verified reason sentences are passed forward, never the factor
    identifiers or values. There is nothing for this stage to get wrong about
    the record because it is not shown the record.

    Args:
        payload: Supplies the jurisdiction context.
        reasons: The verified reason sentences.

    Returns:
        The message.
    """
    listed = "\n".join(f"{index}. {text}" for index, text in enumerate(reasons, start=1))
    return (
        f"Jurisdiction: {payload.jurisdiction_name}\n\n"
        f"Verified reasons to state, all of which must appear:\n{listed}\n\n"
        "Write the complete notice to the applicant."
    )
