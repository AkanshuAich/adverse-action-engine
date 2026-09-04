"""Turning text into vectors.

Two implementations behind one protocol, for a specific reason. The real
embedder downloads a model on first use, which is fine on a workstation and a
poor dependency for a test suite: a build that fails because a model host is
slow has told you nothing about your code. The hashing embedder needs no
download and is deterministic, so retrieval logic - schema, storage, ordering,
distance - is testable offline and in CI.

The hashing embedder is a real technique, not a stub: the hashing trick maps
tokens into fixed dimensions and gives genuine lexical similarity. What it does
not give is *semantic* similarity. It will not match "reasons for rejection" to
"grounds for refusal", which is the whole point of using a learned model. It is
for tests and for running without a download, never for serving.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import numpy as np

from aae.domain.errors import RetrievalError
from aae.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

EMBEDDING_DIMENSIONS: Final[int] = 384
"""Fixed by the model and by the database column, so both implementations match."""

DEFAULT_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"

_TOKEN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Produces vectors for text."""

    @property
    def name(self) -> str:
        """Identifier recorded alongside stored vectors.

        Vectors from different embedders are not comparable, so the name is
        what stops a corpus embedded by one being searched by another.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Vector width."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed.

        Returns:
            An array of shape ``(len(texts), dimensions)``, L2-normalised.
        """
        ...


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction; leaving the norm at 1 returns it
    # unchanged rather than producing NaN.
    norms[norms == 0.0] = 1.0
    normalised: np.ndarray = np.asarray(matrix / norms, dtype=np.float32)
    return normalised


class HashingEmbedder:
    """Deterministic lexical embeddings, with no model download.

    Uses the hashing trick: each token is hashed to a dimension and a sign,
    and contributions are summed. Identical text always produces an identical
    vector, on any machine, with no network.
    """

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        """Build the embedder.

        Args:
            dimensions: Vector width. Must match the database column.
        """
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        """Identifier recorded alongside stored vectors."""
        return f"hashing-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        """Vector width."""
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed.

        Returns:
            An L2-normalised array of shape ``(len(texts), dimensions)``.
        """
        matrix = np.zeros((len(texts), self._dimensions), dtype=np.float32)

        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.casefold()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self._dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign

        return _normalise_rows(matrix)


class FastEmbedEmbedder:
    """Semantic embeddings from a small sentence-transformer.

    Runs on ONNX Runtime rather than PyTorch, which keeps the install to tens
    of megabytes instead of a couple of gigabytes - the difference between
    fitting on a 12 GB machine alongside Postgres and not.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Build the embedder. The model is loaded lazily on first use.

        Args:
            model_name: A model fastembed can serve.
        """
        self._model_name = model_name
        self._model: object | None = None

    @property
    def name(self) -> str:
        """Identifier recorded alongside stored vectors."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Vector width."""
        return EMBEDDING_DIMENSIONS

    def _load(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - ai extras missing
                msg = "fastembed is not installed; install the 'ai' extras."
                raise RetrievalError(msg) from exc

            logger.info("loading_embedding_model", model=self._model_name)
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed.

        Returns:
            An L2-normalised array of shape ``(len(texts), dimensions)``.

        Raises:
            RetrievalError: If the model produces vectors of the wrong width,
                which would silently corrupt every stored embedding.
        """
        model = self._load()
        vectors = np.asarray(list(model.embed(list(texts))), dtype=np.float32)  # type: ignore[attr-defined]

        shape = tuple(int(size) for size in vectors.shape)
        if len(shape) != 2 or shape[1] != EMBEDDING_DIMENSIONS:
            msg = (
                f"{self._model_name} returned vectors of shape {shape}; "
                f"the store expects width {EMBEDDING_DIMENSIONS}."
            )
            raise RetrievalError(msg)

        return _normalise_rows(vectors)
