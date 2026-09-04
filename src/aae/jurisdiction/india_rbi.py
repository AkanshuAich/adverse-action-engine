"""India: RBI fair-practice requirements for declining a credit application.

The Reserve Bank's Fair Practices Code requires a lender to convey the reasons
for rejection to the applicant in writing, and separately requires a disclosed
grievance redressal route. Those two obligations are what this module encodes.

The reason cap is set at four. RBI does not name a number, unlike Regulation B
in the United States, which fixes it at four. A cap is imposed here anyway,
because the purpose of stating reasons is defeated by listing everything that
counted against an applicant: a notice giving fifteen reasons has told them
nothing about which mattered. Four is borrowed from Reg B as a defensible
convention rather than invented, and it keeps the two jurisdiction modules
comparable.
"""

from __future__ import annotations

from typing import Final

from aae.jurisdiction.base import (
    COMMON_PROHIBITED_PATTERNS,
    HAS_CITATIONS,
    HAS_REASONS,
    Jurisdiction,
    RequiredElement,
    compile_prohibited,
)

JURISDICTION_CODE: Final[str] = "india_rbi"

MAX_PRINCIPAL_REASONS: Final[int] = 4

REQUIRED_ELEMENTS: Final[tuple[RequiredElement, ...]] = (
    RequiredElement(
        key="principal_reasons",
        description="The specific reasons the application was declined.",
        predicate=HAS_REASONS,
    ),
    RequiredElement(
        key="regulatory_basis",
        description="The provision under which the notice is issued.",
        predicate=HAS_CITATIONS,
    ),
    RequiredElement(
        key="decision_statement",
        description="A clear statement that the application was declined.",
        # Prose only: the structured notice carries reasons, not the sentence
        # that delivers the outcome. Checked against the text in the prose stage.
        predicate=None,
    ),
    RequiredElement(
        key="grievance_contact",
        description=(
            "How to seek clarification or escalate, per the grievance "
            "redressal mechanism the lender is required to disclose."
        ),
        predicate=None,
    ),
)

INDIA_RBI: Final[Jurisdiction] = Jurisdiction(
    code=JURISDICTION_CODE,
    name="India - RBI Fair Practices Code",
    max_principal_reasons=MAX_PRINCIPAL_REASONS,
    required_elements=REQUIRED_ELEMENTS,
    prohibited_patterns=compile_prohibited(COMMON_PROHIBITED_PATTERNS),
    prohibited_description=(
        "a protected characteristic, which may not lawfully influence a "
        "credit decision and therefore may not be given as a reason for one"
    ),
)
