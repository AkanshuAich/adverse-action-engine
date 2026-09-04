"""What a jurisdiction requires of an adverse action notice.

Regulators differ on the details - how many reasons may be given, what the
notice must contain, which characteristics may never be mentioned - but the
shape of the requirement is the same everywhere. This module is that shape;
:mod:`aae.jurisdiction.india_rbi` and its siblings fill it in.

Required elements carry a *predicate* wherever the structured notice can be
checked directly, rather than trusting the model's own declaration that it
complied. A model asserting "yes, I included the reasons" is not evidence it
did. Where an element only exists in the customer-facing prose - a grievance
contact, for instance - the structured stage falls back to the declaration and
the prose stage checks the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from aae.domain.models import AdverseActionNotice


@runtime_checkable
class CorpusLookup(Protocol):
    """Resolves a citation to the text it claims to quote.

    The verifier depends on this rather than on a vector store, so citation
    checking is testable without a database and the retrieval layer can be
    replaced without touching verification.
    """

    def passage(self, document_id: str, section: str) -> str | None:
        """Return the text of a provision.

        Args:
            document_id: Corpus document identifier.
            section: Section or clause reference within it.

        Returns:
            The passage text, or ``None`` if no such provision exists.
        """
        ...


@dataclass(frozen=True)
class RequiredElement:
    """Something a notice must contain to be lawful.

    Attributes:
        key: Stable identifier, also the value used in ``declared_elements``.
        description: What the regulator requires, in plain terms.
        predicate: Checks the structured notice actually contains it. ``None``
            means the element lives only in the prose, so the structured stage
            can do no better than trust the declaration.
    """

    key: str
    description: str
    predicate: Callable[[AdverseActionNotice], bool] | None = None

    @property
    def checkable_structurally(self) -> bool:
        """Whether this element can be verified without reading the prose."""
        return self.predicate is not None

    def is_satisfied(self, notice: AdverseActionNotice) -> bool:
        """Check whether a notice contains this element.

        Args:
            notice: The structured notice.

        Returns:
            True if present. Falls back to the model's own declaration for
            prose-only elements.
        """
        if self.predicate is not None:
            return self.predicate(notice)
        return self.key in notice.declared_elements


def _has_reasons(notice: AdverseActionNotice) -> bool:
    return len(notice.principal_reasons) > 0


def _has_citations(notice: AdverseActionNotice) -> bool:
    return len(notice.citations) > 0


HAS_REASONS: Final[Callable[[AdverseActionNotice], bool]] = _has_reasons
HAS_CITATIONS: Final[Callable[[AdverseActionNotice], bool]] = _has_citations


@dataclass(frozen=True)
class Jurisdiction:
    """The rules one regulator imposes on adverse action notices.

    Attributes:
        code: Identifier, matched against ``AdverseActionNotice.jurisdiction``.
        name: Human-readable name for reports and the model card.
        max_principal_reasons: Cap on how many reasons may be given. Regulators
            impose one so a notice states the decisive factors rather than
            burying them in a list of everything.
        required_elements: What the notice must contain.
        prohibited_patterns: Compiled expressions matching references to
            protected characteristics.
        prohibited_description: What the patterns are protecting against, for
            the violation message.
    """

    code: str
    name: str
    max_principal_reasons: int
    required_elements: tuple[RequiredElement, ...]
    prohibited_patterns: tuple[re.Pattern[str], ...]
    prohibited_description: str

    def element(self, key: str) -> RequiredElement | None:
        """Look up a required element by key.

        Args:
            key: The element identifier.

        Returns:
            The element, or ``None`` if this jurisdiction does not require it.
        """
        return next((e for e in self.required_elements if e.key == key), None)

    @property
    def required_keys(self) -> frozenset[str]:
        """Every element key this jurisdiction requires."""
        return frozenset(e.key for e in self.required_elements)


def compile_prohibited(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile prohibited-content expressions, case-insensitively.

    Args:
        patterns: Regular expression sources.

    Returns:
        Compiled patterns.
    """
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


# Shared across jurisdictions: every regime that prohibits deciding on a
# characteristic also prohibits citing it as the reason. Phrased as targeted
# expressions rather than bare words - a notice may legitimately say "a single
# payment", so "single" alone is not evidence of anything, while "marital
# status" and "not married" are.
COMMON_PROHIBITED_PATTERNS: Final[tuple[str, ...]] = (
    r"\bmarital status\b",
    r"\b(?:un)?married\b",
    r"\bnot married\b",
    r"\bdivorced?\b",
    r"\bwidow(?:ed|er)?\b",
    r"\bgender\b",
    r"\bsex\b",
    r"\b(?:fe)?male\b",
    r"\byour age\b",
    r"\baged?\s+\d+\b",
    r"\b\d+\s+years old\b",
    r"\bdate of birth\b",
    r"\brace\b",
    r"\bethnic(?:ity)?\b",
    r"\breligion\b",
    r"\bcaste\b",
    r"\bnational origin\b",
    r"\bpregnan(?:t|cy)\b",
    r"\bdisabilit(?:y|ies)\b",
)
