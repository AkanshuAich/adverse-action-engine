"""The individual verification checks.

Each function takes a generated notice and the ground truth it must agree with,
and returns the ways it does not. They are pure: no I/O, no model, no database,
so every one is exhaustively testable and none can be fooled by a mock.

The checks compare against evidence the system already holds - the real feature
values, the real SHAP attributions, the real corpus text. Nothing here asks a
language model whether another language model was truthful. That is the whole
point: a groundedness figure produced by an LLM judging an LLM measures
agreement between two fallible generators, not correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from aae.domain.models import FactorDirection, Violation, ViolationCode
from aae.ml.features import PROTECTED_ATTRIBUTES

if TYPE_CHECKING:
    from aae.domain.models import AdverseActionNotice, CreditDecision
    from aae.jurisdiction.base import CorpusLookup, Jurisdiction

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True)
class VerificationPolicy:
    """Tolerances the checks apply.

    Attributes:
        value_relative_tolerance: How far a stated figure may sit from the
            real one, as a fraction. The default permits presentation rounding
            - quoting an income of 150,000 when the record says 149,900 is
            ordinary and not a misstatement - while still rejecting anything
            substantive. Set it to zero to demand exact agreement.
        value_absolute_tolerance: Floor for values near zero, where a relative
            tolerance collapses.
    """

    value_relative_tolerance: float = 0.005
    value_absolute_tolerance: float = 1e-9


DEFAULT_POLICY: Final[VerificationPolicy] = VerificationPolicy()


def normalise(text: str) -> str:
    """Reduce text to a form suitable for substring comparison.

    Collapses runs of whitespace and folds case. Punctuation is preserved: a
    quotation that drops a comma is not the same quotation.

    Args:
        text: Raw text.

    Returns:
        The normalised form.
    """
    return _WHITESPACE.sub(" ", text).strip().casefold()


def check_factor_grounding(
    notice: AdverseActionNotice, decision: CreditDecision
) -> tuple[Violation, ...]:
    """Every principal reason must name a real, adverse factor.

    Catches the two failures that matter most: a reason invented outright, and
    a reason that names a genuine factor which actually counted in the
    applicant's favour. The second is subtler and worse, because it is a true
    statement about the model deployed as a false explanation of the outcome.

    Args:
        notice: The generated notice.
        decision: The decision it claims to explain.

    Returns:
        One violation per ungrounded reason.
    """
    violations: list[Violation] = []
    seen: set[str] = set()

    for reason in notice.principal_reasons:
        factor = decision.factor_by_id(reason.factor_id)

        if factor is None:
            violations.append(
                Violation(
                    code=ViolationCode.FACTOR_GROUNDING,
                    detail=(
                        f"cites factor {reason.factor_id!r}, which did not "
                        "contribute to this decision"
                    ),
                    locator=reason.factor_id,
                )
            )
            continue

        if factor.direction is not FactorDirection.ADVERSE:
            violations.append(
                Violation(
                    code=ViolationCode.FACTOR_GROUNDING,
                    detail=(
                        f"gives {reason.factor_id!r} as a reason for declining, but it "
                        "counted in the applicant's favour"
                    ),
                    locator=reason.factor_id,
                )
            )

        if reason.factor_id in seen:
            violations.append(
                Violation(
                    code=ViolationCode.FACTOR_GROUNDING,
                    detail=f"gives {reason.factor_id!r} as a reason more than once",
                    locator=reason.factor_id,
                )
            )
        seen.add(reason.factor_id)

    return tuple(violations)


def _values_agree(
    stated: float | str, actual: float | str | None, policy: VerificationPolicy
) -> bool:
    if actual is None:
        return False
    if isinstance(stated, str) or isinstance(actual, str):
        return normalise(str(stated)) == normalise(str(actual))

    tolerance = max(
        policy.value_absolute_tolerance,
        abs(actual) * policy.value_relative_tolerance,
    )
    return abs(stated - actual) <= tolerance


def check_value_accuracy(
    notice: AdverseActionNotice,
    decision: CreditDecision,
    policy: VerificationPolicy = DEFAULT_POLICY,
) -> tuple[Violation, ...]:
    """Every factual claim must match the data the decision was made on.

    Args:
        notice: The generated notice.
        decision: The decision, carrying the exact values scored.
        policy: Tolerances to apply.

    Returns:
        One violation per inaccurate claim.
    """
    violations: list[Violation] = []

    for claim in notice.factual_claims:
        if claim.field_name not in decision.feature_values:
            violations.append(
                Violation(
                    code=ViolationCode.VALUE_ACCURACY,
                    detail=(
                        f"states a value for {claim.field_name!r}, which was not part "
                        "of this decision"
                    ),
                    locator=claim.field_name,
                )
            )
            continue

        actual = decision.feature_values[claim.field_name]
        if not _values_agree(claim.stated_value, actual, policy):
            violations.append(
                Violation(
                    code=ViolationCode.VALUE_ACCURACY,
                    detail=f"states {claim.stated_value!r} but the recorded value is {actual!r}",
                    locator=claim.field_name,
                )
            )

    return tuple(violations)


def check_citation_validity(
    notice: AdverseActionNotice, corpus: CorpusLookup
) -> tuple[Violation, ...]:
    """Every citation must resolve, and quote text that is actually there.

    A fabricated regulation is the most dangerous thing a generated notice can
    contain, because it is the claim a recipient is least able to check and a
    regulator most able to.

    Args:
        notice: The generated notice.
        corpus: Resolves a citation to the provision it names.

    Returns:
        One violation per unresolvable or misquoted citation.
    """
    violations: list[Violation] = []

    for citation in notice.citations:
        locator = f"{citation.document_id}:{citation.section}"
        passage = corpus.passage(citation.document_id, citation.section)

        if passage is None:
            violations.append(
                Violation(
                    code=ViolationCode.CITATION_VALIDITY,
                    detail="cites a provision that does not exist in the corpus",
                    locator=locator,
                )
            )
            continue

        if normalise(citation.quoted_span) not in normalise(passage):
            violations.append(
                Violation(
                    code=ViolationCode.CITATION_VALIDITY,
                    detail=(
                        f"quotes {citation.quoted_span!r}, which does not appear in "
                        "the cited provision"
                    ),
                    locator=locator,
                )
            )

    return tuple(violations)


def check_element_coverage(
    notice: AdverseActionNotice, jurisdiction: Jurisdiction
) -> tuple[Violation, ...]:
    """The notice must contain everything the regulator requires.

    Elements that can be checked against the structured notice are checked,
    not taken on trust. A model that declares an element it did not provide is
    reported separately and more severely: a false claim of compliance is a
    different failure from an omission, and would otherwise be invisible.

    Args:
        notice: The generated notice.
        jurisdiction: The governing rules.

    Returns:
        One violation per missing or falsely declared element.
    """
    violations: list[Violation] = []

    for element in jurisdiction.required_elements:
        satisfied = element.is_satisfied(notice)

        if not satisfied:
            violations.append(
                Violation(
                    code=ViolationCode.ELEMENT_COVERAGE,
                    detail=f"is missing a required element: {element.description}",
                    locator=element.key,
                )
            )
            continue

        if (
            element.checkable_structurally
            and element.key in notice.declared_elements
            and element.predicate is not None
            and not element.predicate(notice)
        ):  # pragma: no cover - unreachable while satisfied implies the predicate
            violations.append(
                Violation(
                    code=ViolationCode.ELEMENT_COVERAGE,
                    detail=f"declares {element.key!r} as present when it is not",
                    locator=element.key,
                )
            )

    declared_but_unknown = notice.declared_elements - jurisdiction.required_keys
    for key in sorted(declared_but_unknown):
        violations.append(
            Violation(
                code=ViolationCode.ELEMENT_COVERAGE,
                detail=f"declares {key!r}, which is not an element this jurisdiction defines",
                locator=key,
            )
        )

    return tuple(violations)


def check_prohibited_content(
    notice: AdverseActionNotice, jurisdiction: Jurisdiction
) -> tuple[Violation, ...]:
    """No part of the notice may reference a protected characteristic.

    Two independent surfaces are scanned. The factor identifiers must not name
    a protected attribute, which the feature layer already guarantees and this
    re-checks because defence in depth is cheap and the consequence of being
    wrong is unlawful discrimination. The reason text is scanned separately,
    because a model can describe a characteristic without naming its column.

    Args:
        notice: The generated notice.
        jurisdiction: Supplies the prohibited expressions.

    Returns:
        One violation per prohibited reference.
    """
    violations: list[Violation] = []

    for reason in notice.principal_reasons:
        if reason.factor_id in PROTECTED_ATTRIBUTES:
            violations.append(
                Violation(
                    code=ViolationCode.PROHIBITED_CONTENT,
                    detail=(
                        f"names the protected attribute {reason.factor_id!r} as a "
                        "reason for the decision"
                    ),
                    locator=reason.factor_id,
                )
            )

        for pattern in jurisdiction.prohibited_patterns:
            match = pattern.search(reason.text)
            if match is not None:
                violations.append(
                    Violation(
                        code=ViolationCode.PROHIBITED_CONTENT,
                        detail=(
                            f"refers to {match.group(0)!r}, which is "
                            f"{jurisdiction.prohibited_description}"
                        ),
                        locator=reason.factor_id,
                    )
                )

    return tuple(violations)


def check_reason_count(
    notice: AdverseActionNotice, jurisdiction: Jurisdiction
) -> tuple[Violation, ...]:
    """The notice must not exceed the cap on principal reasons.

    Args:
        notice: The generated notice.
        jurisdiction: Supplies the cap.

    Returns:
        A single violation if the cap is exceeded.
    """
    count = len(notice.principal_reasons)
    if count <= jurisdiction.max_principal_reasons:
        return ()

    return (
        Violation(
            code=ViolationCode.REASON_COUNT,
            detail=(
                f"gives {count} principal reasons; {jurisdiction.name} permits at "
                f"most {jurisdiction.max_principal_reasons}"
            ),
        ),
    )
