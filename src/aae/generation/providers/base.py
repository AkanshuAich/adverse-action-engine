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
    """

    name: str
    base_url: str
    model: str
    api_key: SecretStr | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    supports_json_schema: bool = True


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

        try:
            response = self._client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            msg = f"{self.name}: request failed: {exc}"
            raise ProviderError(msg) from exc

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            msg = (
                f"{self.name}: rate limited. Free tiers cap requests and tokens per "
                "minute; the eval runner throttles for this reason."
            )
            raise ProviderError(msg)

        if response.status_code >= httpx.codes.BAD_REQUEST:
            msg = f"{self.name}: HTTP {response.status_code}: {response.text[:300]}"
            raise ProviderError(msg)

        return self._parse(response.json(), schema)

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
