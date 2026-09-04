"""United States: ECOA and Regulation B requirements for adverse action.

The second jurisdiction, and the reason the interface exists. If adding it had
required changing :mod:`aae.jurisdiction.base`, the abstraction would have
been wishful thinking; it did not, so the claim that requirements are
pluggable is demonstrated rather than asserted.

Regulation B is more prescriptive than the RBI Fair Practices Code in two ways
that show up directly in the configuration.

The reason cap is **four**, and that is a real number rather than a convention
borrowed for tidiness: the Official Interpretation to 12 CFR 1002.9 says a
creditor need not list more than four reasons, and listing more risks burying
the ones that decided the matter.

The **ECOA notice** is required text, not a summary. A notice must carry the
statutory statement naming the Act and the federal agency that enforces
compliance for that creditor. It is a prose element, so the structured stage
records that the model claims to have included it and the prose stage checks
the words are actually there.
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

JURISDICTION_CODE: Final[str] = "us_reg_b"

MAX_PRINCIPAL_REASONS: Final[int] = 4
"""12 CFR 1002.9, Official Interpretation: four is enough."""

US_PROHIBITED_PATTERNS: Final[tuple[str, ...]] = (
    *COMMON_PROHIBITED_PATTERNS,
    # Prohibited bases named in ECOA that the shared list does not cover.
    r"\bcolor\b",
    r"\bpublic assistance\b",
    r"\bwelfare\b",
    r"\bfood stamps?\b",
    r"\bexercised? (?:any )?rights? under\b",
)
"""ECOA's prohibited bases.

Receipt of public assistance is one of them, and it is the one most often
forgotten because it does not look like a demographic characteristic. So is
having exercised a right under the Consumer Credit Protection Act.
"""

REQUIRED_ELEMENTS: Final[tuple[RequiredElement, ...]] = (
    RequiredElement(
        key="principal_reasons",
        description="The specific principal reasons for the adverse action.",
        predicate=HAS_REASONS,
    ),
    RequiredElement(
        key="regulatory_basis",
        description="The provision under which the notice is issued.",
        predicate=HAS_CITATIONS,
    ),
    RequiredElement(
        key="decision_statement",
        description="A statement of the action taken on the application.",
        predicate=None,
    ),
    RequiredElement(
        key="ecoa_notice",
        description=(
            "The statutory ECOA notice, naming the Act and the federal agency "
            "that enforces compliance for this creditor."
        ),
        predicate=None,
    ),
    RequiredElement(
        key="creditor_contact",
        description="The name and address of the creditor.",
        predicate=None,
    ),
)

US_REG_B: Final[Jurisdiction] = Jurisdiction(
    code=JURISDICTION_CODE,
    name="United States - ECOA / Regulation B",
    max_principal_reasons=MAX_PRINCIPAL_REASONS,
    required_elements=REQUIRED_ELEMENTS,
    prohibited_patterns=compile_prohibited(US_PROHIBITED_PATTERNS),
    prohibited_description=(
        "a prohibited basis under the Equal Credit Opportunity Act, which may "
        "not influence a credit decision and therefore may not be given as a "
        "reason for one"
    ),
)
