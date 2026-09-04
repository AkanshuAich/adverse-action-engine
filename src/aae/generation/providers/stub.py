"""A provider that returns scripted responses.

Tests must not depend on a network, a credential, or a model's mood. The repair
loop in particular can only be tested by making the first attempt fail in a
specific way and the second succeed, which no real provider will do on demand.

This is not a mock in the usual sense: it satisfies the same protocol and
performs the same validation, so a scripted response that does not match the
schema fails exactly as a real one would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError

from aae.domain.errors import ProviderError

if TYPE_CHECKING:
    from collections.abc import Sequence

STUB_NAME: Final[str] = "scripted"


class ScriptedProvider:
    """Returns pre-set responses in order, then repeats the last one.

    Repeating rather than exhausting is deliberate: a test that scripts one bad
    response and one good one should exercise a single repair, not fail on the
    third call with an unrelated error.
    """

    def __init__(
        self,
        responses: Sequence[BaseModel | ProviderError],
        *,
        name: str = STUB_NAME,
        model: str = "scripted-1",
    ) -> None:
        """Build the provider.

        Args:
            responses: Objects to return in order. A ``ProviderError`` entry is
                raised instead of returned, which is how transport failure and
                rate limiting are exercised.
            name: Provider name recorded in the audit trail.
            model: Model name recorded alongside it.

        Raises:
            ValueError: If no responses are supplied.
        """
        if not responses:
            msg = "ScriptedProvider needs at least one response."
            raise ValueError(msg)
        self._responses = list(responses)
        self._name = name
        self._model = model
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        """Provider name."""
        return self._name

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def call_count(self) -> int:
        """How many completions have been requested."""
        return len(self.calls)

    def complete[ModelT: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[ModelT],
        temperature: float = 0.0,
    ) -> ModelT:
        """Return the next scripted response.

        Args:
            system: Recorded so tests can assert on what was sent.
            user: Recorded so tests can assert the repair prompt carried the
                verifier's violations.
            schema: The expected model. Scripted responses are validated
                against it, as a real response would be.
            temperature: Ignored.

        Returns:
            The next scripted object.

        Raises:
            ProviderError: If the script says so, or if the scripted response
                does not satisfy the schema.
        """
        self.calls.append((system, user))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        response = self._responses[index]

        if isinstance(response, ProviderError):
            raise response

        try:
            return schema.model_validate(response.model_dump())
        except ValidationError as exc:
            msg = f"{self.name}: scripted response did not match {schema.__name__}: {exc}"
            raise ProviderError(msg) from exc
