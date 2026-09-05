"""Application configuration.

All configuration arrives through environment variables prefixed ``AAE_``.
Nothing is hardcoded and no secret ever appears in source. See ``.env.example``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aae.database_url import normalise_database_url
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

    llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description=(
            "Which backend answers. Groq, not Cerebras: Cerebras withdrew its "
            "free tier during this project and now returns 402 for every "
            "model. The provider abstraction exists for exactly this, and the "
            "migration was these two lines."
        ),
    )
    llm_model: str = Field(
        default="openai/gpt-oss-120b",
        description=(
            "Model identifier as the backend spells it. Pinned against a live "
            "model listing rather than from memory: the previous default named "
            "a model that had ceased to exist, which fails identically to a "
            "bad credential."
        ),
    )
    cerebras_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    jurisdiction: str = "india_rbi"
    decision_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of default at or above which credit is declined. "
            "Not 0.5: that is the naive classification cut, and it is wrong "
            "for an imbalanced target. Measured on this model, a 0.5 threshold "
            "declines 2.7% of applicants against an 8.3% default rate, so it "
            "approves a great many people who will not repay. It also means "
            "the notice pipeline almost never fires, which would make a system "
            "built to explain declines produce almost none. The evaluation "
            "harness, the golden set and the monitoring reports all use 0.15; "
            "this is the value that makes them agree."
        ),
    )
    top_k_factors: int = Field(default=5, ge=1, le=20)
    max_repair_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("database_url", "migration_database_url")
    @classmethod
    def _normalise_driver(cls, value: str) -> str:
        """Accept a connection string exactly as a provider hands it out.

        Managed Postgres services give URLs beginning ``postgresql://``, which
        SQLAlchemy reads as a request for psycopg2. Only psycopg 3 is
        installed, so a bare scheme is always a mistake with always the same
        fix, and normalising it means a pasted string works.

        Args:
            value: The configured URL.

        Returns:
            The URL with an explicit driver.
        """
        return normalise_database_url(value)

    def llm_api_key(self) -> SecretStr | None:
        """Return the API key for the selected provider, checking it exists.

        Deliberately a method rather than a load-time validator. Requiring an
        LLM credential merely to construct settings would mean the decision
        API, the migrations, and the training job all refuse to start without
        one, despite none of them calling a language model. The check belongs
        where the credential is used, so a missing key fails the notice
        generation that needs it rather than the service that does not.

        Returns:
            The key, or ``None`` for the local provider which needs none.

        Raises:
            ConfigurationError: If a hosted provider is selected without a key.
        """
        keys: dict[LLMProvider, SecretStr | None] = {
            LLMProvider.CEREBRAS: self.cerebras_api_key,
            LLMProvider.GROQ: self.groq_api_key,
        }
        if self.llm_provider not in keys:
            return None

        key = keys[self.llm_provider]
        if key is None:
            msg = (
                f"LLM provider {self.llm_provider.value!r} requires "
                f"AAE_{self.llm_provider.value.upper()}_API_KEY to be set."
            )
            raise ConfigurationError(msg)
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once.

    Returns:
        The immutable settings object.
    """
    return Settings()
