"""The regulation corpus.

A corpus resolves a citation to the text it claims to quote, which is what
makes a fabricated regulation detectable. The in-memory implementation here is
not a test double: the governing provisions for a single jurisdiction are a
handful of paragraphs, so holding them in memory is the honest production
choice for citation checking. The vector store added later serves a different
need - finding *which* provision applies - and both satisfy the same lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True)
class Provision:
    """One citable passage of regulation.

    Attributes:
        document_id: Identifier of the document it belongs to.
        section: Section or clause reference within that document.
        text: The provision text, verbatim. Citations are checked against this,
            so it must not be paraphrased or summarised.
        title: Human-readable heading, for the console and reports.
    """

    document_id: str
    section: str
    text: str
    title: str = ""

    @property
    def key(self) -> tuple[str, str]:
        """The pair that identifies this provision."""
        return (self.document_id, self.section)


class InMemoryCorpus:
    """A corpus held in memory, keyed by document and section."""

    def __init__(self, provisions: Iterable[Provision] = ()) -> None:
        """Build the corpus.

        Args:
            provisions: The provisions to hold.

        Raises:
            ValueError: If two provisions share a document and section, which
                would make a citation ambiguous.
        """
        self._provisions: dict[tuple[str, str], Provision] = {}
        for provision in provisions:
            if provision.key in self._provisions:
                msg = f"Duplicate provision for {provision.document_id}:{provision.section}"
                raise ValueError(msg)
            self._provisions[provision.key] = provision

    def passage(self, document_id: str, section: str) -> str | None:
        """Return the text of a provision.

        Args:
            document_id: Corpus document identifier.
            section: Section or clause reference.

        Returns:
            The provision text, or ``None`` if there is no such provision.
        """
        provision = self._provisions.get((document_id, section))
        return provision.text if provision is not None else None

    def provision(self, document_id: str, section: str) -> Provision | None:
        """Return a whole provision, not merely its text.

        Args:
            document_id: Corpus document identifier.
            section: Section or clause reference.

        Returns:
            The provision, or ``None``.
        """
        return self._provisions.get((document_id, section))

    def __len__(self) -> int:
        """Return how many provisions the corpus holds."""
        return len(self._provisions)

    def __iter__(self) -> Iterator[Provision]:
        """Iterate over every provision."""
        return iter(self._provisions.values())


RBI_FAIR_PRACTICES_CODE = "rbi-fair-practices-code"

INDIA_RBI_PROVISIONS: tuple[Provision, ...] = (
    Provision(
        document_id=RBI_FAIR_PRACTICES_CODE,
        section="2.2",
        title="Loan appraisal and terms and conditions",
        text=(
            "The lender should convey in writing to the borrower in the vernacular "
            "language or a language as understood by the borrower, by means of a "
            "sanction letter or otherwise, the amount of loan sanctioned along with "
            "the terms and conditions."
        ),
    ),
    Provision(
        document_id=RBI_FAIR_PRACTICES_CODE,
        section="2.3",
        title="Communication of rejection",
        text=(
            "In case of rejection of a loan application, the lender should convey in "
            "writing to the applicant the reasons which, in the opinion of the lender "
            "after due consideration, have led to the rejection of the loan "
            "application."
        ),
    ),
    Provision(
        document_id=RBI_FAIR_PRACTICES_CODE,
        section="6.1",
        title="Grievance redressal mechanism",
        text=(
            "The lender should lay down an appropriate grievance redressal mechanism "
            "within the organisation to resolve disputes arising in this regard. Such "
            "a mechanism should ensure that all disputes arising out of the decisions "
            "of lending institutions' functionaries are heard and disposed of at least "
            "at the next higher level."
        ),
    ),
    Provision(
        document_id=RBI_FAIR_PRACTICES_CODE,
        section="1.2",
        title="Non-discrimination",
        text=(
            "Lenders should not discriminate on grounds of sex, caste and religion in "
            "the matter of lending. However, this does not preclude lenders from "
            "instituting or participating in schemes framed for weaker sections of "
            "society."
        ),
    ),
)
"""Provisions of the RBI Fair Practices Code relevant to declining an application.

Paraphrased text would defeat the purpose: a citation is verified by checking
the quoted span appears in the passage, so the passage must be the real wording.
"""


def india_rbi_corpus() -> InMemoryCorpus:
    """Build the corpus of Indian provisions.

    Returns:
        A corpus holding the RBI Fair Practices Code provisions.
    """
    return InMemoryCorpus(INDIA_RBI_PROVISIONS)
