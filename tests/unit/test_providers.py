"""The OpenAI-compatible provider adapter.

One adapter serves Cerebras, Groq, and a local Ollama, so its error handling is
the error handling for every backend. These tests drive it through a mock
transport rather than a network: the failures that matter - a rate limit, a
truncated response, prose where JSON was requested - are exactly the ones a
live call will not produce on demand.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from aae.config import LLMProvider, Settings
from aae.domain.errors import ConfigurationError, ProviderError
from aae.generation.providers.base import OpenAICompatibleProvider, ProviderConfig
from aae.generation.providers.registry import BASE_URLS, build_provider


class Answer(BaseModel):
    """A minimal schema for exercising the adapter."""

    verdict: str
    score: float


def _config(**overrides: Any) -> ProviderConfig:
    defaults: dict[str, Any] = {
        "name": "test-backend",
        "base_url": "https://api.example.test/v1",
        "model": "test-model-1",
        "api_key": SecretStr("secret-value"),
    }
    return ProviderConfig(**{**defaults, **overrides})


def _provider_returning(
    content: str, *, status: int = 200, capture: list[httpx.Request] | None = None
) -> OpenAICompatibleProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if status != 200:
            return httpx.Response(status, text=content)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return OpenAICompatibleProvider(
        _config(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )


class TestSuccessfulCall:
    def test_returns_a_validated_object(self):
        provider = _provider_returning(json.dumps({"verdict": "ok", "score": 0.5}))
        answer = provider.complete(system="s", user="u", schema=Answer)
        assert answer.verdict == "ok"
        assert answer.score == 0.5

    def test_sends_the_credential_as_a_bearer_token(self):
        captured: list[httpx.Request] = []
        provider = _provider_returning(
            json.dumps({"verdict": "ok", "score": 1.0}), capture=captured
        )
        provider.complete(system="s", user="u", schema=Answer)

        assert captured[0].headers["Authorization"] == "Bearer secret-value"

    def test_requests_json_and_a_deterministic_temperature(self):
        """A regulated notice should not vary between identical inputs."""
        captured: list[httpx.Request] = []
        provider = _provider_returning(
            json.dumps({"verdict": "ok", "score": 1.0}), capture=captured
        )
        provider.complete(system="s", user="u", schema=Answer)

        body = json.loads(captured[0].content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.0
        assert body["model"] == "test-model-1"

    def test_posts_to_the_chat_completions_path(self):
        captured: list[httpx.Request] = []
        provider = _provider_returning(
            json.dumps({"verdict": "ok", "score": 1.0}), capture=captured
        )
        provider.complete(system="s", user="u", schema=Answer)
        assert str(captured[0].url) == "https://api.example.test/v1/chat/completions"

    def test_carries_both_messages(self):
        captured: list[httpx.Request] = []
        provider = _provider_returning(
            json.dumps({"verdict": "ok", "score": 1.0}), capture=captured
        )
        provider.complete(system="the rules", user="the request", schema=Answer)

        messages = json.loads(captured[0].content)["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == "the rules"


class TestMalformedResponses:
    def test_prose_instead_of_json_is_rejected(self):
        """Models emit explanations around JSON. It must fail here, not later."""
        provider = _provider_returning("Certainly! Here is the answer you asked for.")
        with pytest.raises(ProviderError, match="not valid JSON"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_truncated_json_is_rejected(self):
        provider = _provider_returning('{"verdict": "ok", "sco')
        with pytest.raises(ProviderError, match="not valid JSON"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_valid_json_of_the_wrong_shape_is_rejected(self):
        """The dangerous case: plausible output that does not fit the schema."""
        provider = _provider_returning(json.dumps({"answer": "ok"}))
        with pytest.raises(ProviderError, match="did not match Answer"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_a_response_without_choices_is_rejected(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "quota exhausted"})

        provider = OpenAICompatibleProvider(
            _config(), client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with pytest.raises(ProviderError, match="no message content"):
            provider.complete(system="s", user="u", schema=Answer)


class TestTransportFailures:
    def test_rate_limiting_is_named_explicitly(self):
        """Free tiers cap requests per minute; the message should say so."""
        provider = _provider_returning("slow down", status=429)
        with pytest.raises(ProviderError, match="rate limited"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_a_server_error_carries_the_status(self):
        provider = _provider_returning("upstream exploded", status=503)
        with pytest.raises(ProviderError, match="HTTP 503"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_a_connection_failure_is_wrapped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            msg = "connection refused"
            raise httpx.ConnectError(msg, request=request)

        provider = OpenAICompatibleProvider(
            _config(), client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        with pytest.raises(ProviderError, match="request failed"):
            provider.complete(system="s", user="u", schema=Answer)


class TestLocalBackend:
    def test_no_authorization_header_without_a_key(self):
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        class Empty(BaseModel):
            pass

        provider = OpenAICompatibleProvider(
            _config(api_key=None),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        provider.complete(system="s", user="u", schema=Empty)
        assert "Authorization" not in captured[0].headers


class TestRegistry:
    def test_every_provider_has_an_endpoint(self):
        assert set(BASE_URLS) == set(LLMProvider)

    def test_builds_a_hosted_provider(self):
        settings = Settings(
            _env_file=None,
            llm_provider=LLMProvider.GROQ,
            groq_api_key=SecretStr("gsk-test"),
            llm_model="llama-3.3-70b",
        )
        provider = build_provider(settings)
        assert provider.name == "groq"
        assert provider.model == "llama-3.3-70b"

    def test_a_hosted_provider_without_a_key_fails_at_construction(self):
        """Not at settings load: the scoring API must start without a key."""
        settings = Settings(
            _env_file=None, llm_provider=LLMProvider.CEREBRAS, cerebras_api_key=None
        )
        with pytest.raises(ConfigurationError, match="CEREBRAS_API_KEY"):
            build_provider(settings)

    def test_the_local_provider_uses_the_configured_host(self):
        settings = Settings(
            _env_file=None,
            llm_provider=LLMProvider.OLLAMA,
            ollama_base_url="http://gpu-box:11434",
        )
        provider = build_provider(settings)
        assert provider.name == "ollama"


class TestEmbedderShapeGuard:
    def test_wrong_width_vectors_are_refused(self):
        """A silent width mismatch would corrupt every stored embedding."""
        import numpy as np

        from aae.domain.errors import RetrievalError
        from aae.retrieval.embedding import FastEmbedEmbedder

        class WrongWidthModel:
            @staticmethod
            def embed(texts: list[str]) -> list[Any]:
                return [np.zeros(128, dtype=np.float32) for _ in texts]

        embedder = FastEmbedEmbedder()
        embedder._model = WrongWidthModel()

        with pytest.raises(RetrievalError, match="expects width 384"):
            embedder.embed(["text"])
