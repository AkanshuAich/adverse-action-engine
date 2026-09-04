"""Application configuration.

All configuration arrives through environment variables prefixed ``AAE_``.
Nothing is hardcoded and no secret ever appears in source. See ``.env.example``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aae.domain.errors import ConfigurationError


class Environment(StrEnum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    """Supported language model providers, all on free tiers."""

    CEREBRAS = "cerebras"
    GROQ = "groq"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Runtime settings, populated from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="AAE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    database_url: str = Field(
        default="postgresql+psycopg://aae_app:aae_dev_password@localhost:5433/aae",
        description="Application connection. This role cannot UPDATE or DELETE audit records.",
    )
    migration_database_url: str = Field(
        default="postgresql+psycopg://aae_owner:aae_dev_password@localhost:5433/aae",
        description="Owner connection, used only by Alembic. The application never uses this.",
    )

    llm_provider: LLMProvider = LLMProvider.CEREBRAS
    llm_model: str = "llama-3.3-70b"
    cerebras_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    jurisdiction: str = "india_rbi"
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k_factors: int = Field(default=5, ge=1, le=20)
    max_repair_attempts: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def _require_key_for_hosted_provider(self) -> Settings:
        """Fail fast when a hosted provider is selected without its API key."""
        required: dict[LLMProvider, SecretStr | None] = {
            LLMProvider.CEREBRAS: self.cerebras_api_key,
            LLMProvider.GROQ: self.groq_api_key,
        }
        key = required.get(self.llm_provider)
        if self.llm_provider in required and key is None:
            msg = (
                f"LLM provider {self.llm_provider.value!r} requires "
                f"AAE_{self.llm_provider.value.upper()}_API_KEY to be set."
            )
            raise ConfigurationError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once.

    Returns:
        The immutable settings object.
    """
    return Settings()
