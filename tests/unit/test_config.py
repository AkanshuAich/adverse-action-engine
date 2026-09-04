"""Tests for settings loading and validation.

``_env_file=None`` is passed throughout so a developer's real ``.env`` can never
change the outcome of a test.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from aae.config import Environment, LLMProvider, Settings, get_settings
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
        assert settings.llm_provider is LLMProvider.CEREBRAS
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
    def test_cerebras_without_key_is_rejected(self):
        with pytest.raises(ConfigurationError, match="CEREBRAS_API_KEY"):
            Settings(_env_file=None, llm_provider=LLMProvider.CEREBRAS, cerebras_api_key=None)

    def test_groq_without_key_is_rejected(self):
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
            Settings(_env_file=None, llm_provider=LLMProvider.GROQ, groq_api_key=None)

    def test_groq_with_key_is_accepted(self):
        settings = Settings(
            _env_file=None,
            llm_provider=LLMProvider.GROQ,
            groq_api_key=SecretStr("gsk-test"),
        )
        assert settings.llm_provider is LLMProvider.GROQ

    def test_ollama_needs_no_key(self):
        """The local fallback must work with no credentials at all."""
        settings = Settings(_env_file=None, llm_provider=LLMProvider.OLLAMA)
        assert settings.llm_provider is LLMProvider.OLLAMA


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
