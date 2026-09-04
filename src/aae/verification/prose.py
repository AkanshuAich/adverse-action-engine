"""Checking the customer-facing letter.

Stage one produces a typed object that can be checked field by field. Stage two
turns it into prose, and prose is where a model can quietly add something: a
figure nobody supplied, a characteristic nobody mentioned.

Two things are checkable here without asking a model to grade a model.

**Prohibited content**, reusing the jurisdiction's expressions. A model told
not to mention age in the structured stage can still mention it while writing
warmly.

**Invented figures.** Every number in the letter must correspond to something
the model was actually given. This is the check that catches "your income of
₹250,000" when the record says 149,900.

The numeric check has a deliberate blind spot, stated here rather than
discovered later. Small integers are permitted unconditionally, because list
numbering, a count of reasons, and a month all produce them and flagging those
would make every notice fail. A fabricated small integer therefore passes -
"you have 4 missed payments" would not be caught. That is acceptable *for this
payload*, which contains no small-integer facts about the applicant, and stops
being acceptable the moment one is added.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from aae.domain.models import Violation, ViolationCode
from aae.verification.rules import DEFAULT_POLICY, VerificationPolicy

if TYPE_CHECKING:
    from collections.abc import Iterable

    from aae.domain.models import AdverseActionNotice
    from aae.generation.payload import GenerationPayload
    from aae.jurisdiction.base import Jurisdiction

_NUMBER: Final[re.Pattern[str]] = re.compile(r"\d[\d,]*(?:\.\d+)?")

SMALL_INTEGER_CEILING: Final[int] = 12
"""Integers at or below this are permitted without matching a supplied value.

List numbering, a count of reasons, and a month number all land here. See the
module docstring for what this costs.
"""


def _parse_numbers(text: str) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for match in _NUMBER.finditer(text):
        token = match.group(0)
        try:
            found.append((token, float(token.replace(",", ""))))
        except ValueError:  # pragma: no cover - the pattern guarantees a float
            continue
    return found


def _permitted_values(notice: AdverseActionNotice, payload: GenerationPayload) -> list[float]:
    """Collect every number the model legitimately had access to."""
    permitted: list[float] = []

    for factor in payload.factors:
        if isinstance(factor.value, (int, float)):
            permitted.append(float(factor.value))

    for claim in notice.factual_claims:
        if isinstance(claim.stated_value, (int, float)):
            permitted.append(float(claim.stated_value))

    # Section references such as "2.3" are legitimately quoted in the letter.
    for citation in notice.citations:
        for _, value in _parse_numbers(citation.section):
            permitted.append(value)

    return permitted


def _is_supported(value: float, permitted: Iterable[float], policy: VerificationPolicy) -> bool:
    if value.is_integer() and abs(value) <= SMALL_INTEGER_CEILING:
        return True

    for allowed in permitted:
        tolerance = max(
            policy.value_absolute_tolerance,
            abs(allowed) * policy.value_relative_tolerance,
        )
        if abs(value - allowed) <= tolerance:
            return True

        # A letter may reasonably round 149,900 to 150,000, or state a ratio of
        # 0.3064 as 31%. Both are the same fact presented for a reader.
        if allowed != 0 and abs(value - allowed * 100.0) <= abs(allowed * 100.0) * 0.01:
            return True
        if _rounds_to(value, allowed):
            return True

    return False


def _rounds_to(stated: float, actual: float) -> bool:
    """Whether a stated figure is a plausible rounding of the real one."""
    return any(round(actual, places) == stated for places in (-4, -3, -2, -1, 0, 1, 2))


def check_prose(
    body: str,
    notice: AdverseActionNotice,
    payload: GenerationPayload,
    jurisdiction: Jurisdiction,
    policy: VerificationPolicy = DEFAULT_POLICY,
) -> tuple[Violation, ...]:
    """Check the rendered letter against what the model was given.

    Args:
        body: The customer-facing text.
        notice: The verified structured notice it was rendered from.
        payload: What the model was permitted to see.
        jurisdiction: Supplies the prohibited expressions.
        policy: Numeric tolerances.

    Returns:
        One violation per prohibited reference or unsupported figure.
    """
    violations: list[Violation] = []

    for pattern in jurisdiction.prohibited_patterns:
        match = pattern.search(body)
        if match is not None:
            violations.append(
                Violation(
                    code=ViolationCode.PROHIBITED_CONTENT,
                    detail=(
                        f"the letter refers to {match.group(0)!r}, which is "
                        f"{jurisdiction.prohibited_description}"
                    ),
                    locator="body",
                )
            )

    permitted = _permitted_values(notice, payload)
    for token, value in _parse_numbers(body):
        if not _is_supported(value, permitted, policy):
            violations.append(
                Violation(
                    code=ViolationCode.VALUE_ACCURACY,
                    detail=(
                        f"the letter states the figure {token!r}, which does not "
                        "correspond to anything in the applicant's record"
                    ),
                    locator="body",
                )
            )

    return tuple(violations)
