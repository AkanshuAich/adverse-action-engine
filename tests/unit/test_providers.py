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
from aae.generation.providers.base import (
    OpenAICompatibleProvider,
    ProviderConfig,
    _parse_duration,
    _retry_after_seconds,
    strict_schema,
)
from aae.generation.providers.registry import (
    BASE_URLS,
    SUPPORTS_JSON_SCHEMA,
    build_provider,
)


class Answer(BaseModel):
    """A minimal schema for exercising the adapter."""

    verdict: str
    score: float


class Defaulted(BaseModel):
    """Defaulted fields are what Pydantic leaves out of ``required``."""

    name: str
    items: list[str] = []


class Nested(BaseModel):
    """A schema whose shape lives in ``$defs``."""

    inner: Defaulted


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

    def test_requests_the_schema_and_a_deterministic_temperature(self):
        """A regulated notice should not vary between identical inputs."""
        captured: list[httpx.Request] = []
        provider = _provider_returning(
            json.dumps({"verdict": "ok", "score": 1.0}), capture=captured
        )
        provider.complete(system="s", user="u", schema=Answer)

        body = json.loads(captured[0].content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "Answer"
        assert body["response_format"]["json_schema"]["strict"] is True
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


class TestRateLimits:
    """A 429 says when to ask again; discarding the work ignores it.

    An evaluation run against a free tier lost 71 of 100 cases to rate limits,
    and lost them inside fourteen seconds: a refused call returns immediately,
    so the throttle between cases never applied to the failures. Waiting is not
    an optimisation here - without it the harness cannot measure a free tier at
    all.
    """

    def test_a_caller_that_cannot_wait_still_fails_fast(self):
        """Zero retries is the default, because an HTTP handler must not block."""
        provider = _provider_returning("slow down", status=429)

        with pytest.raises(ProviderError, match="rate limited"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_a_batch_caller_waits_and_succeeds(self):
        responses = [
            httpx.Response(
                429,
                headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1.05s"},
            ),
            httpx.Response(
                200, json={"choices": [{"message": {"content": '{"verdict": "ok", "score": 1.0}'}}]}
            ),
        ]
        slept: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        provider = OpenAICompatibleProvider(
            _config(rate_limit_retries=3),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        provider._sleep = slept.append  # type: ignore[method-assign]

        answer = provider.complete(system="s", user="u", schema=Answer)

        assert answer.verdict == "ok"
        assert slept == [pytest.approx(1.05)]

    def test_it_gives_up_after_the_configured_attempts(self):
        slept: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1s"},
            )

        provider = OpenAICompatibleProvider(
            _config(rate_limit_retries=2),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        provider._sleep = slept.append  # type: ignore[method-assign]

        with pytest.raises(ProviderError, match="still limited after 2 attempts"):
            provider.complete(system="s", user="u", schema=Answer)

        assert len(slept) == 2

    def test_it_refuses_to_wait_beyond_the_ceiling(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": "1h43m40s",
                },
            )

        provider = OpenAICompatibleProvider(
            _config(rate_limit_retries=3),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(ProviderError, match="beyond the"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_the_backend_explanation_reaches_the_caller(self):
        """The headers describe per-minute buckets only.

        A daily allowance appears in none of them: an exhausted one still
        reports a full per-minute bucket, and every request fails anyway. The
        body names the limit that was actually enforced, so it must not be
        swallowed - diagnosing this from headers alone cost two wrong
        conclusions.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"x-ratelimit-remaining-tokens": "8000"},
                json={
                    "error": {
                        "message": (
                            "Rate limit reached for model `openai/gpt-oss-120b` on "
                            "tokens per day (TPD): Limit 200000, Used 197204."
                        )
                    }
                },
            )

        provider = OpenAICompatibleProvider(
            _config(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(ProviderError, match="tokens per day"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_a_body_that_is_not_json_is_still_reported(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="slow down please")

        provider = OpenAICompatibleProvider(
            _config(),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(ProviderError, match="slow down please"):
            provider.complete(system="s", user="u", schema=Answer)

    def test_it_refuses_to_guess_when_told_nothing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429)

        provider = OpenAICompatibleProvider(
            _config(rate_limit_retries=3),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(ProviderError, match="did not say for how long"):
            provider.complete(system="s", user="u", schema=Answer)


class TestRetryAfter:
    """The wait must come from the bucket that is empty.

    Taking the longest reset header looks safer and is wrong:
    x-ratelimit-reset-requests reports when a daily allowance returns to full,
    which is hours away even with most of it unspent. Waiting on that for a
    per-minute token limit turns a one-second pause into an abandoned run.
    """

    def test_an_empty_token_bucket_gives_a_short_wait(self):
        wait = _retry_after_seconds(
            httpx.Headers(
                {
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-tokens": "1.05s",
                    "x-ratelimit-remaining-requests": "928",
                    "x-ratelimit-reset-requests": "1h43m40s",
                }
            )
        )

        assert wait == pytest.approx(1.05)

    def test_retry_after_overrides_the_reset_headers(self):
        wait = _retry_after_seconds(
            httpx.Headers(
                {
                    "retry-after": "3",
                    "x-ratelimit-remaining-tokens": "0",
                    "x-ratelimit-reset-tokens": "60s",
                }
            )
        )

        assert wait == pytest.approx(3.0)

    def test_no_signal_at_all_returns_nothing(self):
        assert _retry_after_seconds(httpx.Headers({})) is None

    @pytest.mark.parametrize(
        ("raw", "seconds"),
        [
            ("1.05s", 1.05),
            ("7m12.799s", 432.799),
            ("1h43m40.799s", 6220.799),
            ("620ms", 0.62),
            ("2", 2.0),
        ],
    )
    def test_it_reads_the_duration_format_these_backends_send(self, raw: str, seconds: float):
        assert _parse_duration(raw) == pytest.approx(seconds)

    def test_an_unreadable_duration_is_not_invented(self):
        assert _parse_duration("soon") is None


class TestTransportFailures:
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


class TestConstrainedDecoding:
    """Asking for JSON is not asking for the right JSON.

    A bare ``json_object`` request buys syntactically valid JSON and nothing
    more. Against a live backend this produced a well-formed object whose
    fields were invented - ``reason`` and ``citation`` where the schema said
    ``text`` - which fails validation and burns a repair attempt for no reason.
    Sending the schema moves the shape from instruction to decoding constraint.
    """

    def test_the_schema_travels_with_the_request(self):
        captured: list[httpx.Request] = []
        provider = _provider_returning(
            json.dumps({"verdict": "ok", "score": 1.0}), capture=captured
        )
        provider.complete(system="s", user="u", schema=Answer)

        sent = json.loads(captured[0].content)["response_format"]["json_schema"]["schema"]
        assert set(sent["properties"]) == {"verdict", "score"}

    def test_every_property_is_required_because_strict_mode_demands_it(self):
        """Pydantic omits defaulted fields; strict mode rejects that schema."""
        schema = strict_schema(Defaulted)

        assert schema["required"] == ["items", "name"]
        assert schema["additionalProperties"] is False

    def test_nested_objects_are_tightened_too(self):
        """A nested definition left loose fails the whole request, not part."""
        schema = strict_schema(Nested)
        inner = schema["$defs"]["Defaulted"]

        assert inner["required"] == ["items", "name"]
        assert inner["additionalProperties"] is False

    def test_a_backend_without_support_asks_only_for_valid_json(self):
        """Ollama's compatibility surface has carried this unevenly."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"verdict": "ok", "score": 1.0}'}}]},
            )

        config = ProviderConfig(
            name="ollama",
            base_url="http://localhost:11434/v1",
            model="qwen2.5:3b",
            supports_json_schema=False,
        )
        provider = OpenAICompatibleProvider(
            config, client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        provider.complete(system="s", user="u", schema=Answer)

        body = json.loads(captured[0].content)
        assert body["response_format"] == {"type": "json_object"}

    def test_the_local_backend_is_the_only_one_declared_unsupported(self):
        assert SUPPORTS_JSON_SCHEMA[LLMProvider.GROQ] is True
        assert SUPPORTS_JSON_SCHEMA[LLMProvider.CEREBRAS] is True
        assert SUPPORTS_JSON_SCHEMA[LLMProvider.OLLAMA] is False
