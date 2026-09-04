"""The regulation corpus in Postgres, against a real database with pgvector.

These use the hashing embedder rather than the real one. A build that fails
because a model host was slow has told you nothing about the code, and every
property under test here - storage, exact lookup, ordering by distance,
refusing a mismatched embedder - is independent of which model produced the
vectors.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from aae.audit.session import create_session_factory
from aae.domain.errors import RetrievalError
from aae.retrieval.corpus import (
    INDIA_RBI_PROVISIONS,
    RBI_FAIR_PRACTICES_CODE,
    Provision,
)
from aae.retrieval.embedding import EMBEDDING_DIMENSIONS, HashingEmbedder
from aae.retrieval.store import PgVectorCorpus

pytestmark = pytest.mark.integration

JURISDICTION = "india_rbi"


@pytest.fixture
def corpus(migrated_db, owner_connection) -> PgVectorCorpus:
    owner_connection.execute("TRUNCATE regulation_chunk RESTART IDENTITY")
    url = (
        f"postgresql+psycopg://aae_app:app_test_password"
        f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
    )
    factory = create_session_factory(create_engine(url, pool_pre_ping=True))
    return PgVectorCorpus(factory, HashingEmbedder(), jurisdiction=JURISDICTION)


class TestIngest:
    def test_stores_every_provision(self, corpus: PgVectorCorpus):
        assert corpus.ingest(INDIA_RBI_PROVISIONS) == len(INDIA_RBI_PROVISIONS)
        assert corpus.count() == len(INDIA_RBI_PROVISIONS)

    def test_ingesting_nothing_is_a_no_op(self, corpus: PgVectorCorpus):
        assert corpus.ingest([]) == 0
        assert corpus.count() == 0

    def test_re_ingesting_replaces_rather_than_duplicates(self, corpus: PgVectorCorpus):
        """A stale copy would still resolve a citation against old wording."""
        corpus.ingest(INDIA_RBI_PROVISIONS)
        corpus.ingest(INDIA_RBI_PROVISIONS)
        assert corpus.count() == len(INDIA_RBI_PROVISIONS)


class TestCitationLookup:
    def test_resolves_a_stored_provision(self, corpus: PgVectorCorpus):
        corpus.ingest(INDIA_RBI_PROVISIONS)
        passage = corpus.passage(RBI_FAIR_PRACTICES_CODE, "2.3")
        assert passage is not None
        assert "convey in writing to the applicant the reasons" in passage

    def test_text_is_stored_verbatim(self, corpus: PgVectorCorpus):
        """Any drift and a genuine quotation would fail the citation check."""
        corpus.ingest(INDIA_RBI_PROVISIONS)
        original = next(p for p in INDIA_RBI_PROVISIONS if p.section == "6.1")
        assert corpus.passage(original.document_id, original.section) == original.text

    def test_returns_none_for_an_unknown_section(self, corpus: PgVectorCorpus):
        corpus.ingest(INDIA_RBI_PROVISIONS)
        assert corpus.passage(RBI_FAIR_PRACTICES_CODE, "99.9") is None

    def test_satisfies_the_lookup_protocol_the_verifier_needs(self, corpus: PgVectorCorpus):
        from aae.jurisdiction.base import CorpusLookup

        assert isinstance(corpus, CorpusLookup)


class TestSearch:
    def test_finds_the_provision_about_rejection(self, corpus: PgVectorCorpus):
        corpus.ingest(INDIA_RBI_PROVISIONS)
        results = corpus.search("reasons for rejection of a loan application", limit=2)

        assert results
        assert any(provision.section == "2.3" for provision in results)

    def test_finds_the_grievance_provision(self, corpus: PgVectorCorpus):
        corpus.ingest(INDIA_RBI_PROVISIONS)
        results = corpus.search("grievance redressal mechanism disputes", limit=2)
        assert any(provision.section == "6.1" for provision in results)

    def test_respects_the_limit(self, corpus: PgVectorCorpus):
        corpus.ingest(INDIA_RBI_PROVISIONS)
        assert len(corpus.search("lending", limit=2)) == 2

    def test_searching_an_empty_corpus_is_an_error_not_an_empty_answer(
        self, corpus: PgVectorCorpus
    ):
        """Silently returning nothing would produce a notice with no citation."""
        with pytest.raises(RetrievalError, match="Ingest the corpus"):
            corpus.search("anything")


class TestEmbedderMismatch:
    def test_searching_with_a_different_embedder_is_refused(self, migrated_db, owner_connection):
        """Vectors from different models are not comparable.

        Answering anyway would return confident nonsense, which is worse than
        an error because nothing downstream could tell.
        """
        owner_connection.execute("TRUNCATE regulation_chunk RESTART IDENTITY")
        url = (
            f"postgresql+psycopg://aae_app:app_test_password"
            f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
        )
        factory = create_session_factory(create_engine(url, pool_pre_ping=True))

        PgVectorCorpus(factory, HashingEmbedder(), jurisdiction=JURISDICTION).ingest(
            INDIA_RBI_PROVISIONS
        )

        other = PgVectorCorpus(
            factory,
            HashingEmbedder(dimensions=EMBEDDING_DIMENSIONS),
            jurisdiction=JURISDICTION,
        )
        # Same dimensions, different declared name.
        object.__setattr__(other, "_embedder", _RenamedEmbedder())

        with pytest.raises(RetrievalError, match="not comparable"):
            other.search("reasons for rejection")


class _RenamedEmbedder(HashingEmbedder):
    """A hashing embedder that claims to be a different model."""

    @property
    def name(self) -> str:
        """Deliberately different from what was ingested."""
        return "some-other-model"


class TestJurisdictionIsolation:
    def test_one_jurisdiction_cannot_see_another(self, migrated_db, owner_connection):
        owner_connection.execute("TRUNCATE regulation_chunk RESTART IDENTITY")
        url = (
            f"postgresql+psycopg://aae_app:app_test_password"
            f"@{migrated_db.get_container_host_ip()}:{migrated_db.get_exposed_port(5432)}/aae"
        )
        factory = create_session_factory(create_engine(url, pool_pre_ping=True))

        india = PgVectorCorpus(factory, HashingEmbedder(), jurisdiction="india_rbi")
        india.ingest(INDIA_RBI_PROVISIONS[:2])

        other = PgVectorCorpus(factory, HashingEmbedder(), jurisdiction="us_reg_b")
        other.ingest(
            [
                Provision(
                    document_id="ecoa-reg-b",
                    section="1002.9",
                    title="Notifications",
                    text="A creditor shall notify an applicant of action taken.",
                )
            ]
        )

        assert india.count() == 2
        assert other.count() == 1
        assert other.passage(RBI_FAIR_PRACTICES_CODE, "2.3") is None


class TestHashingEmbedder:
    def test_is_deterministic(self):
        left = HashingEmbedder().embed(["reasons for rejection"])
        right = HashingEmbedder().embed(["reasons for rejection"])
        assert (left == right).all()

    def test_produces_the_expected_width(self):
        assert HashingEmbedder().embed(["text"]).shape == (1, EMBEDDING_DIMENSIONS)

    def test_vectors_are_normalised(self):
        import numpy as np

        vectors = HashingEmbedder().embed(["one", "two words here"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_empty_text_does_not_produce_nan(self):
        import numpy as np

        assert not np.isnan(HashingEmbedder().embed([""])).any()
