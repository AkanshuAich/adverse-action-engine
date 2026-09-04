"""The regulation corpus in Postgres.

Serves two different needs from one table, which is why it exists alongside the
in-memory corpus rather than replacing it.

**Lookup** resolves a citation to the exact text it claims to quote. This is
what makes a fabricated regulation detectable, and it is an exact-match read.

**Search** finds which provisions are relevant to a decision, so the model is
shown the two or three that apply rather than the whole code. This is a nearest
neighbour query over embeddings.

Vectors are stored with the name of the embedder that produced them. Vectors
from different models are not comparable - searching a corpus embedded by one
model with a query embedded by another returns confident nonsense - so a
mismatch is refused rather than silently answered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sqlalchemy import Index, String, Text, delete, func, select
from sqlalchemy.orm import Mapped, mapped_column

from aae.audit.models import Base
from aae.domain.errors import RetrievalError
from aae.logging import get_logger
from aae.retrieval.corpus import Provision
from aae.retrieval.embedding import EMBEDDING_DIMENSIONS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session, sessionmaker

    from aae.retrieval.embedding import Embedder

logger = get_logger(__name__)

VECTOR_AVAILABLE: bool
"""Whether pgvector's SQLAlchemy type is importable.

The corpus degrades to lookup-only without it rather than failing to import,
so the verifier - which needs exact citation resolution, not search - keeps
working when the vector extras are absent.
"""

try:  # pragma: no cover - exercised by whichever branch the install provides
    from pgvector.sqlalchemy import Vector

    VECTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    VECTOR_AVAILABLE = False


DEFAULT_SEARCH_LIMIT: Final[int] = 3


class RegulationChunk(Base):
    """One citable provision, with its embedding."""

    __tablename__ = "regulation_chunk"
    __table_args__ = (
        # Citation lookup is an exact match on this pair and happens on every
        # verification, so it is the index that matters most.
        Index("ix_regulation_chunk_reference", "document_id", "section", unique=True),
        Index("ix_regulation_chunk_jurisdiction", "jurisdiction"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    jurisdiction: Mapped[str] = mapped_column(String(64), nullable=False)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False)
    section: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Stored verbatim. A paraphrase would make every citation unverifiable.",
    )
    embedder: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="Which model produced the vector. Vectors from different models are not comparable.",
    )

    if VECTOR_AVAILABLE:
        embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)

    def to_provision(self) -> Provision:
        """Convert to the domain provision.

        Returns:
            The provision, without its embedding.
        """
        return Provision(
            document_id=self.document_id,
            section=self.section,
            text=self.text,
            title=self.title,
        )


class PgVectorCorpus:
    """A regulation corpus backed by Postgres and pgvector."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embedder: Embedder,
        *,
        jurisdiction: str,
    ) -> None:
        """Build the corpus.

        Args:
            session_factory: Produces database sessions.
            embedder: Embeds queries. Must be the one used to ingest.
            jurisdiction: Restricts every read to one regime's provisions.
        """
        self._session_factory = session_factory
        self._embedder = embedder
        self._jurisdiction = jurisdiction

    def ingest(self, provisions: Sequence[Provision], *, replace: bool = True) -> int:
        """Embed and store provisions.

        Args:
            provisions: The provisions to store, verbatim.
            replace: Remove this jurisdiction's existing provisions first.
                Defaults to true so re-ingesting after an edit does not leave
                a stale copy that a citation could still resolve against.

        Returns:
            How many provisions were stored.
        """
        if not provisions:
            return 0

        vectors = self._embedder.embed([self._embed_text(p) for p in provisions])

        with self._session_factory() as session, session.begin():
            if replace:
                session.execute(
                    delete(RegulationChunk).where(
                        RegulationChunk.jurisdiction == self._jurisdiction
                    )
                )
            for provision, vector in zip(provisions, vectors, strict=True):
                session.add(
                    RegulationChunk(
                        jurisdiction=self._jurisdiction,
                        document_id=provision.document_id,
                        section=provision.section,
                        title=provision.title,
                        text=provision.text,
                        embedder=self._embedder.name,
                        embedding=list(map(float, vector)),
                    )
                )

        logger.info(
            "corpus_ingested",
            jurisdiction=self._jurisdiction,
            provisions=len(provisions),
            embedder=self._embedder.name,
        )
        return len(provisions)

    @staticmethod
    def _embed_text(provision: Provision) -> str:
        """Combine title and text so a heading contributes to retrieval."""
        return f"{provision.title}. {provision.text}" if provision.title else provision.text

    def passage(self, document_id: str, section: str) -> str | None:
        """Return the text of a provision, for citation checking.

        Args:
            document_id: Corpus document identifier.
            section: Section or clause reference.

        Returns:
            The provision text, or ``None`` if there is no such provision.
        """
        with self._session_factory() as session:
            return session.execute(
                select(RegulationChunk.text).where(
                    RegulationChunk.jurisdiction == self._jurisdiction,
                    RegulationChunk.document_id == document_id,
                    RegulationChunk.section == section,
                )
            ).scalar_one_or_none()

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> tuple[Provision, ...]:
        """Find the provisions most relevant to a query.

        Args:
            query: What the notice needs to cite.
            limit: How many provisions to return.

        Returns:
            Provisions in order of relevance.

        Raises:
            RetrievalError: If the stored vectors came from a different
                embedder, which would make the distances meaningless.
        """
        self._assert_embedder_matches()
        vector = self._embedder.embed([query])[0]

        with self._session_factory() as session:
            rows = (
                session.execute(
                    select(RegulationChunk)
                    .where(RegulationChunk.jurisdiction == self._jurisdiction)
                    .order_by(RegulationChunk.embedding.cosine_distance(list(map(float, vector))))
                    .limit(limit)
                )
                .scalars()
                .all()
            )

        return tuple(row.to_provision() for row in rows)

    def _assert_embedder_matches(self) -> None:
        with self._session_factory() as session:
            stored = (
                session.execute(
                    select(RegulationChunk.embedder)
                    .where(RegulationChunk.jurisdiction == self._jurisdiction)
                    .distinct()
                )
                .scalars()
                .all()
            )

        if not stored:
            msg = (
                f"No provisions stored for {self._jurisdiction!r}. Ingest the corpus "
                "before searching it."
            )
            raise RetrievalError(msg)

        mismatched = [name for name in stored if name != self._embedder.name]
        if mismatched:
            msg = (
                f"Corpus was embedded with {sorted(set(mismatched))} but the query "
                f"embedder is {self._embedder.name!r}. Vectors from different models "
                "are not comparable; re-ingest the corpus."
            )
            raise RetrievalError(msg)

    def count(self) -> int:
        """Return how many provisions are stored for this jurisdiction.

        Returns:
            The provision count.
        """
        with self._session_factory() as session:
            return int(
                session.execute(
                    select(func.count())
                    .select_from(RegulationChunk)
                    .where(RegulationChunk.jurisdiction == self._jurisdiction)
                ).scalar_one()
            )
