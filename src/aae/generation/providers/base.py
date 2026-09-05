"""Language model providers.

All three supported backends - Cerebras, Groq, and a local Ollama - speak the
OpenAI chat-completions protocol, so there is one adapter parameterised by base
URL, model, and credential rather than three SDKs. That keeps the dependency
surface at ``httpx``, makes every request inspectable, and means swapping
provider is configuration rather than code.

Free tiers moved materially during 2026, which is the practical argument for
this abstraction: the graph above it never learns which backend answered.

Structured output is *constrained* to the expected schema at the decoder,
and then validated again on arrival. Asking for ``json_object`` only buys
syntactically valid JSON: it leaves the model free to invent field names, and
models do. Sending the schema itself makes the shape a decoding constraint
rather than an instruction the model may reinterpret.

Validation still runs afterwards. A backend that ignores the constraint, or
one configured without support for it, must fail as a provider error rather
than flow onward as a half-populated notice.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from aae.domain.errors import ProviderError
from aae.logging import get_logger

if TYPE_CHECKING:
    from pydantic import SecretStr

logger = get_logger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_RATE_LIMIT_WAIT_SECONDS: Final[float] = 65.0
"""Longest single pause honoured before giving up on a rate limit.

Slightly over a minute, because the limit that binds a free tier is measured
per minute and a longer reset means a daily allowance has run out - which
waiting will not fix.
"""

DEFAULT_TIMEOUT_SECONDS: Final[float] = 90.0
DEFAULT_MAX_TOKENS: Final[int] = 2_000


def strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic schema into one strict structured output accepts.

    Strict mode requires every property of every object to be listed in
    ``required`` and additional properties to be forbidden. Pydantic omits
    fields that have defaults, so the schema it generates is rejected as
    written. Listing them all is sound here because every optional field on
    these models defaults to an empty collection: the model is being asked to
    supply the key, not to invent content for it.

    Args:
        schema: The model the response must satisfy.

    Returns:
        A JSON schema suitable for ``strict: true``.
    """

    def tighten(node: Any) -> Any:
        if isinstance(node, dict):
            tightened = {key: tighten(value) for key, value in node.items()}
            if tightened.get("type") == "object" and "properties" in tightened:
                tightened["required"] = sorted(tightened["properties"])
                tightened["additionalProperties"] = False
            return tightened
        if isinstance(node, list):
            return [tighten(item) for item in node]
        return node

    return dict(tighten(schema.model_json_schema()))


_DURATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:(?P<hours>[\d.]+)h)?(?:(?P<minutes>[\d.]+)m(?!s))?(?:(?P<seconds>[\d.]+)s)?"
    r"(?:(?P<millis>[\d.]+)ms)?"
)


def _limit_detail(response: httpx.Response) -> str:
    """Quote what the backend said about the limit.

    The per-request headers describe only the per-minute buckets. A daily
    allowance does not appear in them at all, so a run that has exhausted one
    reads ``x-ratelimit-remaining-tokens: 8000`` - a full bucket - and every
    request still fails. The body is where the backend names the limit it
    actually enforced, in a sentence, and it was there the whole time while
    this code inferred the wrong cause from headers twice.

    Args:
        response: The 429 response.

    Returns:
        The backend's own explanation, or a note that it gave none.
    """
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        text = response.text.strip()
        return f"Backend said: {text[:300]}" if text else "The backend gave no explanation."
    return f"Backend said: {str(message)[:300]}"


def _parse_duration(value: str) -> float | None:
    """Read a duration in the form these backends actually send.

    ``retry-after`` is seconds, but the rate-limit reset headers are written
    like ``7m12.5s`` or ``1.05s`` or ``620ms``.

    Args:
        value: The header value.

    Returns:
        Seconds, or ``None`` if nothing could be read from it.
    """
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass

    match = _DURATION_PATTERN.fullmatch(text)
    if match is None or not any(match.groupdict().values()):
        return None

    parts = {key: float(raw) for key, raw in match.groupdict().items() if raw is not None}
    return (
        parts.get("hours", 0.0) * 3600.0
        + parts.get("minutes", 0.0) * 60.0
        + parts.get("seconds", 0.0)
        + parts.get("millis", 0.0) / 1000.0
    )


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    """Decide how long to wait before asking again.

    ``retry-after`` is authoritative when present. Otherwise the wait comes
    from the bucket that is actually empty, which the ``remaining`` headers
    identify.

    Reading the reset headers alone is not enough, and taking the longest of
    them is wrong: ``x-ratelimit-reset-requests`` reports when the request
    bucket returns to *full*, which on a daily allowance is hours away even
    with hundreds of requests still in hand. Waiting on that for a
    tokens-per-minute limit turns a one-second pause into an abandoned run.

    Args:
        headers: The response headers.

    Returns:
        Seconds to wait, or ``None`` if the response gave no usable signal.
    """
    direct = headers.get("retry-after")
    if direct is not None and (parsed := _parse_duration(direct)) is not None:
        return parsed

    waits = [
        reset
        for remaining_header, reset_header in (
            ("x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
            ("x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"),
        )
        if (raw_remaining := headers.get(remaining_header)) is not None
        and (remaining := _parse_duration(raw_remaining)) is not None
        and remaining <= 0
        and (raw_reset := headers.get(reset_header)) is not None
        and (reset := _parse_duration(raw_reset)) is not None
    ]
    if waits:
        return max(waits)

    # Nothing reported empty. The per-minute token bucket is the limit a free
    # tier hits in practice, so its reset is the best available guess.
    raw = headers.get("x-ratelimit-reset-tokens")
    return _parse_duration(raw) if raw is not None else None


@runtime_checkable
class StructuredProvider(Protocol):
    """Produces a validated object from a prompt."""

    @property
    def name(self) -> str:
        """Identifier recorded in the audit trail."""
        ...

    @property
    def model(self) -> str:
        """The model in use, recorded alongside the decision."""
        ...

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float = 0.0,
    ) -> ModelT:
        """Return a validated instance of ``schema``.

        Args:
            system: System instruction.
            user: User message.
            schema: The Pydantic model the response must satisfy.
            temperature: Sampling temperature.

        Returns:
            The validated object.

        Raises:
            ProviderError: If the call fails or the response does not validate.
        """
        ...


@dataclass(frozen=True)
class ProviderConfig:
    """Where and how to reach one backend.

    Attributes:
        name: Identifier recorded in the audit trail.
        base_url: OpenAI-compatible endpoint root, without ``/chat/completions``.
        model: Model identifier the backend expects.
        api_key: Credential, or ``None`` for a local backend.
        timeout_seconds: Wall-clock budget for one call.
        max_tokens: Ceiling on the response.
        supports_json_schema: Whether the backend can constrain decoding to a
            supplied schema. Declared per backend rather than discovered by
            catching a 400, so an unsupported backend is a known limitation
            recorded in configuration and not a silent downgrade.
        rate_limit_retries: How many times to wait out a rate limit and try
            again. Zero for anything serving a request: a caller waiting on an
            HTTP response wants a fast 502, not a worker blocked for a minute.
            Non-zero for batch work, where the run is long anyway and losing
            the case costs more than waiting for it.
    """

    name: str
    base_url: str
    model: str
    api_key: SecretStr | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    supports_json_schema: bool = True
    rate_limit_retries: int = 0


class OpenAICompatibleProvider:
    """One adapter for every backend that speaks chat-completions."""

    def __init__(self, config: ProviderConfig, client: httpx.Client | None = None) -> None:
        """Build the provider.

        Args:
            config: Endpoint, model, and credential.
            client: Injected HTTP client, for tests. One is created if absent.
        """
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    @property
    def name(self) -> str:
        """Identifier recorded in the audit trail."""
        return self._config.name

    @property
    def model(self) -> str:
        """The model in use."""
        return self._config.model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key.get_secret_value()}"
        return headers

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float = 0.0,
    ) -> ModelT:
        """Call the backend and validate the response.

        Args:
            system: System instruction.
            user: User message.
            schema: The Pydantic model the response must satisfy.
            temperature: Sampling temperature. Defaults to zero, because a
                regulated notice should not vary between identical inputs.

        Returns:
            The validated object.

        Raises:
            ProviderError: On transport failure, a non-2xx response, malformed
                JSON, or a response that does not satisfy the schema.
        """
        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": temperature,
            "max_tokens": self._config.max_tokens,
            "response_format": self._response_format(schema),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        response = self._post_honouring_rate_limits(url, payload)

        if response.status_code >= httpx.codes.BAD_REQUEST:
            msg = f"{self.name}: HTTP {response.status_code}: {response.text[:300]}"
            raise ProviderError(msg)

        return self._parse(response.json(), schema)

    def _post_honouring_rate_limits(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        """Send the request, waiting out a rate limit if one is configured.

        A 429 is not a failure of the request; it is the backend saying when to
        ask again, and it says so in a header. Treating it as fatal discarded
        71 of 100 cases in one evaluation run - and discarded them in fourteen
        seconds, because a refused call returns instantly and the caller's
        throttle never applies. Guessing a slower throttle does not fix that:
        the limit is on tokens, which vary per case, so any fixed spacing is
        either too slow for most cases or too fast for some.

        Args:
            url: The chat-completions endpoint.
            payload: The request body.

        Returns:
            The response, which may still carry an error status.

        Raises:
            ProviderError: On transport failure, or a rate limit that outlasts
                the configured retries.
        """
        for attempt in range(self._config.rate_limit_retries + 1):
            try:
                response = self._client.post(url, json=payload, headers=self._headers())
            except httpx.HTTPError as exc:
                msg = f"{self.name}: request failed: {exc}"
                raise ProviderError(msg) from exc

            if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
                return response

            remaining = self._config.rate_limit_retries - attempt
            if remaining <= 0:
                msg = (
                    f"{self.name}: rate limited, and still limited after "
                    f"{self._config.rate_limit_retries} attempts to wait it out. "
                    f"{_limit_detail(response)}"
                )
                raise ProviderError(msg)

            wait = _retry_after_seconds(response.headers)
            if wait is None:
                msg = (
                    f"{self.name}: rate limited, and the response did not say for how "
                    f"long. Waiting an arbitrary period would be guessing. "
                    f"{_limit_detail(response)}"
                )
                raise ProviderError(msg)
            if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                msg = (
                    f"{self.name}: rate limited for {wait:.0f}s, beyond the "
                    f"{MAX_RATE_LIMIT_WAIT_SECONDS:.0f}s ceiling. "
                    f"{_limit_detail(response)}"
                )
                raise ProviderError(msg)

            logger.warning(
                "rate_limited_waiting",
                provider=self.name,
                wait_seconds=round(wait, 2),
                attempts_left=remaining,
            )
            self._sleep(wait)

        raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover

    def _sleep(self, seconds: float) -> None:
        """Pause between attempts.

        Isolated so a test can drive the retry path without spending the wait.

        Args:
            seconds: How long to pause.
        """
        time.sleep(seconds)

    def _response_format(self, schema: type[BaseModel]) -> dict[str, Any]:
        """Describe the required response shape to the backend.

        Args:
            schema: The model the response must satisfy.

        Returns:
            A ``response_format`` object: the schema itself where the backend
            supports it, and otherwise a bare request for valid JSON.
        """
        if not self._config.supports_json_schema:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": strict_schema(schema),
                "strict": True,
            },
        }

    def _parse(self, body: dict[str, Any], schema: type[ModelT]) -> ModelT:
        """Extract and validate the structured payload.

        Args:
            body: Decoded response body.
            schema: The expected model.

        Returns:
            The validated object.

        Raises:
            ProviderError: If the shape is wrong at any level.
        """
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"{self.name}: response had no message content: {str(body)[:300]}"
            raise ProviderError(msg) from exc

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            # Routine, not exceptional: models emit prose, fenced code, or
            # truncated objects. It must fail here rather than downstream.
            msg = f"{self.name}: response was not valid JSON: {content[:300]}"
            raise ProviderError(msg) from exc

        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            msg = f"{self.name}: response did not match {schema.__name__}: {exc}"
            raise ProviderError(msg) from exc

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
