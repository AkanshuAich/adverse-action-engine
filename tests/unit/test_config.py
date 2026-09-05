"""Tests for settings loading and validation.

``_env_file=None`` is passed throughout so a developer's real ``.env`` can never
change the outcome of a test.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from aae.config import Environment, LLMProvider, Settings, get_settings
from aae.database_url import describe_target, is_local_default
from aae.domain.errors import ConfigurationError


def _settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "cerebras_api_key": SecretStr("test-key"),
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestDefaults:
    def test_sensible_defaults(self):
        settings = _settings()
        assert settings.env is Environment.DEVELOPMENT
        assert settings.llm_provider is LLMProvider.GROQ
        assert settings.jurisdiction == "india_rbi"
        assert settings.max_repair_attempts == 3

    def test_settings_are_frozen(self):
        settings = _settings()
        with pytest.raises(ValidationError):
            settings.decision_threshold = 0.9  # type: ignore[misc]

    def test_migration_url_differs_from_application_url(self):
        """The app must never connect with the role that can drop tables."""
        settings = _settings()
        assert settings.database_url != settings.migration_database_url
        assert "aae_app" in settings.database_url
        assert "aae_owner" in settings.migration_database_url


class TestProviderKeyValidation:
    def test_settings_load_without_an_llm_key(self):
        """Config must not demand a credential the caller may never use.

        The decision API, the migrations, and the training job all call no
        language model. Refusing to construct settings without an LLM key
        would stop every one of them from starting.
        """
        settings = Settings(
            _env_file=None, llm_provider=LLMProvider.CEREBRAS, cerebras_api_key=None
        )
        assert settings.llm_provider is LLMProvider.CEREBRAS

    def test_cerebras_without_key_fails_at_the_point_of_use(self):
        settings = Settings(
            _env_file=None, llm_provider=LLMProvider.CEREBRAS, cerebras_api_key=None
        )
        with pytest.raises(ConfigurationError, match="CEREBRAS_API_KEY"):
            settings.llm_api_key()

    def test_groq_without_key_fails_at_the_point_of_use(self):
        settings = Settings(_env_file=None, llm_provider=LLMProvider.GROQ, groq_api_key=None)
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            settings.llm_api_key()

    def test_groq_with_key_returns_it(self):
        settings = Settings(
            _env_file=None,
            llm_provider=LLMProvider.GROQ,
            groq_api_key=SecretStr("gsk-test"),
        )
        key = settings.llm_api_key()
        assert key is not None
        assert key.get_secret_value() == "gsk-test"

    def test_ollama_needs_no_key(self):
        """The local fallback must work with no credentials at all."""
        settings = Settings(_env_file=None, llm_provider=LLMProvider.OLLAMA)
        assert settings.llm_api_key() is None


class TestBounds:
    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_threshold_must_be_a_probability(self, threshold: float):
        with pytest.raises(ValidationError):
            _settings(decision_threshold=threshold)

    @pytest.mark.parametrize("top_k", [0, 21])
    def test_top_k_factors_is_bounded(self, top_k: int):
        with pytest.raises(ValidationError):
            _settings(top_k_factors=top_k)

    def test_repair_attempts_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            _settings(max_repair_attempts=0)


class TestSecretHandling:
    def test_api_key_is_not_exposed_by_repr(self):
        """A settings dump must never leak a credential into a log line."""
        settings = _settings(cerebras_api_key=SecretStr("super-secret-value"))
        assert "super-secret-value" not in repr(settings)
        assert settings.cerebras_api_key is not None
        assert settings.cerebras_api_key.get_secret_value() == "super-secret-value"


class TestGetSettings:
    def test_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AAE_CEREBRAS_API_KEY", "k")
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()


class TestShippedSampleMatchesShippedDefault:
    """The sample and the default threshold must agree.

    `samples/decline.json` is used in the deployment instructions and named
    for what it is supposed to do. It was silently approved for a while,
    because the API defaulted to 0.5 while the golden set that produced the
    sample was built at 0.15. This ties them together so they cannot diverge
    again without a test failing.
    """

    def test_the_default_threshold_matches_the_evaluation_harness(self):
        settings = _settings()
        assert settings.decision_threshold == 0.15

    def test_the_shipped_sample_declines_at_the_default_threshold(self):
        import json
        from pathlib import Path

        import pandas as pd

        from aae.data.loaders import load_applications
        from aae.data.schema import validate_for_scoring
        from aae.domain.models import Decision
        from aae.ml.decision import DecisionEngine
        from aae.ml.train import train_model

        sample = Path("samples/decline.json")
        if not sample.is_file():
            pytest.skip("sample not present")

        payload = json.loads(sample.read_text(encoding="utf-8"))
        frame = validate_for_scoring(pd.DataFrame([payload]))

        model = train_model(load_applications(force_synthetic=True, n_synthetic=20_000))
        engine = DecisionEngine(model, threshold=_settings().decision_threshold)

        assert engine.decide(frame).decision is Decision.DECLINE


class TestConnectionDiagnostics:
    """A failed connection must be diagnosable without exposing the password.

    Deployment platforms redact any error that might carry a credential, which
    withholds the useful half along with the dangerous half: a live console
    showed a stack trace ending in OperationalError and nothing about what it
    had dialled. Describing the target by host and database alone keeps the
    diagnosis and drops the secret.
    """

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("postgresql+psycopg://aae_app:pw@localhost:5433/aae", "localhost:5433/aae"),
            (
                "postgresql://u:p@db.example.neon.tech/aae?sslmode=require",
                "db.example.neon.tech/aae",
            ),
            ("postgresql://u:p@127.0.0.1:5432/aae", "127.0.0.1:5432/aae"),
        ],
    )
    def test_it_names_the_target(self, url: str, expected: str):
        assert describe_target(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "postgresql+psycopg://aae_app:aae_dev_password@localhost:5433/aae",
            "postgresql://user:hunter2@db.example.neon.tech/aae?sslmode=require",
        ],
    )
    def test_the_password_never_appears(self, url: str):
        described = describe_target(url)

        assert "aae_dev_password" not in described
        assert "hunter2" not in described
        assert "@" not in described

    def test_an_unreadable_url_does_not_raise(self):
        """A diagnostic that crashes replaces one unreadable error with another."""
        assert describe_target("not a url at all")

    @pytest.mark.parametrize(
        ("url", "local"),
        [
            ("postgresql://u:p@localhost:5433/aae", True),
            ("postgresql://u:p@127.0.0.1:5432/aae", True),
            ("postgresql://u:p@[::1]:5432/aae", True),
            ("postgresql://u:p@db.example.neon.tech/aae", False),
        ],
    )
    def test_it_recognises_the_local_default(self, url: str, local: bool):
        assert is_local_default(url) is local
