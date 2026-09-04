"""Selecting a provider from configuration.

Base URLs are recorded here rather than in settings so that switching backend
is a single environment variable and not a URL a deployer has to know. The
credential is still supplied per environment; only the endpoint is a constant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from aae.config import LLMProvider
from aae.generation.providers.base import (
    OpenAICompatibleProvider,
    ProviderConfig,
    StructuredProvider,
)
from aae.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from aae.config import Settings

logger = get_logger(__name__)

BASE_URLS: Final[dict[LLMProvider, str]] = {
    LLMProvider.CEREBRAS: "https://api.cerebras.ai/v1",
    LLMProvider.GROQ: "https://api.groq.com/openai/v1",
    # Ollama exposes an OpenAI-compatible surface alongside its own API, which
    # is what lets one adapter serve local and hosted backends alike.
    LLMProvider.OLLAMA: "http://localhost:11434/v1",
}


def build_provider(settings: Settings, client: httpx.Client | None = None) -> StructuredProvider:
    """Construct the configured provider.

    Args:
        settings: Supplies the provider choice, model, and credential.
        client: Injected HTTP client, for tests.

    Returns:
        A provider ready to call.

    Raises:
        ConfigurationError: If a hosted provider is selected without its key.
    """
    provider = settings.llm_provider
    base_url = (
        f"{settings.ollama_base_url.rstrip('/')}/v1"
        if provider is LLMProvider.OLLAMA
        else BASE_URLS[provider]
    )

    config = ProviderConfig(
        name=provider.value,
        base_url=base_url,
        model=settings.llm_model,
        # Raises here, at the point of use, rather than at settings load. The
        # scoring API and the migrations call no model and must not require a
        # credential to start.
        api_key=settings.llm_api_key(),
    )

    logger.info("llm_provider_selected", provider=config.name, model=config.model)
    return OpenAICompatibleProvider(config, client=client)
