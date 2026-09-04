"""Dependency wiring for the API.

Everything expensive - the engine, the connection pool, the booster, the SHAP
explainer - is built once and cached. Constructing a ``TreeExplainer`` walks
the whole ensemble, so doing it per request would dominate response time.

Nothing here is a module-level singleton. Each provider is a cached function,
which keeps the graph explicit and lets a test override any single piece
through FastAPI's ``dependency_overrides`` without monkeypatching imports.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aae.audit.repository import AuditRepository
from aae.audit.session import create_database_engine, create_session_factory
from aae.config import get_settings
from aae.generation.graph import NoticeGenerator
from aae.generation.providers.registry import build_provider
from aae.jurisdiction.india_rbi import INDIA_RBI
from aae.logging import get_logger
from aae.ml.decision import DecisionEngine
from aae.ml.registry import load_model, save_model
from aae.ml.train import TrainedModel, train_model
from aae.retrieval.corpus import InMemoryCorpus, india_rbi_corpus
from aae.verification.verifier import NoticeVerifier

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker

logger = get_logger(__name__)

MODEL_DIRECTORY: Final[Path] = Path("artifacts/model")
BOOTSTRAP_ROWS: Final[int] = 20_000


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory.

    Returns:
        A session factory bound to the application role.
    """
    return create_session_factory(create_database_engine(get_settings()))


@lru_cache(maxsize=1)
def get_audit_repository() -> AuditRepository:
    """Return the audit repository.

    Returns:
        Append-only access to the chain.
    """
    return AuditRepository(get_session_factory())


@lru_cache(maxsize=1)
def get_model() -> TrainedModel:
    """Load the trained model, training one if no artifact exists.

    Bootstrapping on synthetic data keeps a fresh clone runnable without a
    Kaggle download. Provenance travels with the model, so a decision made by
    a bootstrapped model is identifiable as such in its own audit record
    rather than being indistinguishable from one trained on real data.

    Returns:
        The trained, calibrated model.
    """
    if (MODEL_DIRECTORY / "metadata.json").is_file():
        return load_model(MODEL_DIRECTORY)

    logger.warning(
        "no_model_artifact_bootstrapping",
        directory=str(MODEL_DIRECTORY),
        detail="Training on synthetic data. Provenance is recorded on the model.",
    )
    # Imported here so the API does not pull the data layer when an artifact
    # already exists.
    from aae.data.loaders import load_applications

    model = train_model(load_applications(force_synthetic=True, n_synthetic=BOOTSTRAP_ROWS))
    save_model(model, MODEL_DIRECTORY)
    return model


@lru_cache(maxsize=1)
def get_decision_engine() -> DecisionEngine:
    """Return the decision engine.

    Returns:
        An engine holding the model and its explainer.
    """
    settings = get_settings()
    return DecisionEngine(
        get_model(),
        threshold=settings.decision_threshold,
        top_k=settings.top_k_factors,
    )


@lru_cache(maxsize=1)
def get_corpus() -> InMemoryCorpus:
    """Return the regulation corpus used for citation checking.

    Held in memory rather than read from the vector store. Citation checking
    is an exact lookup of a handful of provisions on every verification, and a
    round trip to Postgres for each buys nothing. The pgvector store exists for
    the other job - finding *which* provision applies once the corpus is a
    full Master Direction rather than four paragraphs.

    Returns:
        The corpus of provisions for the configured jurisdiction.
    """
    return india_rbi_corpus()


@lru_cache(maxsize=1)
def get_verifier() -> NoticeVerifier:
    """Return the notice verifier.

    Returns:
        A verifier for the configured jurisdiction.
    """
    return NoticeVerifier(INDIA_RBI, get_corpus())


@lru_cache(maxsize=1)
def get_notice_generator() -> NoticeGenerator:
    """Return the notice generator.

    Constructing the provider is where a missing LLM credential surfaces. That
    is deliberate: the scoring API must start without one, and only the
    endpoint that actually calls a model should fail for the want of a key.

    Returns:
        A generator wired to the configured provider and verifier.

    Raises:
        ConfigurationError: If a hosted provider is selected without its key.
    """
    settings = get_settings()
    return NoticeGenerator(
        provider=build_provider(settings),
        verifier=get_verifier(),
        max_attempts=settings.max_repair_attempts,
    )


def reset_caches() -> None:
    """Clear every cached provider.

    Used by tests that need a different database or model between cases.
    """
    get_session_factory.cache_clear()
    get_audit_repository.cache_clear()
    get_model.cache_clear()
    get_decision_engine.cache_clear()
    get_corpus.cache_clear()
    get_verifier.cache_clear()
    get_notice_generator.cache_clear()
