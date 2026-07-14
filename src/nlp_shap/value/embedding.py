"""Embedding-based value functions."""

from ..domain.enums import EmbeddingMode
from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult
from ..protocols.embedding import EmbeddingProvider
from ._metrics import cosine_similarity, euclidean_similarity


class CosineEmbeddingValue:
    """Cosine similarity between static or contextual embeddings (U1/U2)."""

    def __init__(
        self,
        embedding_mode: EmbeddingMode = EmbeddingMode.STATIC,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self._embedding_mode = embedding_mode
        self._provider = provider

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "embedding_cosine"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return cosine similarity between embedding vectors."""
        base_vector = _resolve_embedding(base, self._embedding_mode, self._provider)
        candidate_vector = _resolve_embedding(
            candidate,
            self._embedding_mode,
            self._provider,
        )
        return cosine_similarity(base_vector, candidate_vector)


class EuclideanEmbeddingValue:
    """Euclidean-derived similarity between embeddings (U4)."""

    def __init__(
        self,
        embedding_mode: EmbeddingMode = EmbeddingMode.STATIC,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self._embedding_mode = embedding_mode
        self._provider = provider

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "embedding_euclidean"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return ``1 / (1 + distance)`` between embedding vectors."""
        base_vector = _resolve_embedding(base, self._embedding_mode, self._provider)
        candidate_vector = _resolve_embedding(
            candidate,
            self._embedding_mode,
            self._provider,
        )
        return euclidean_similarity(base_vector, candidate_vector)


def _resolve_embedding(
    generation: GenerationResult,
    embedding_mode: EmbeddingMode,
    provider: EmbeddingProvider | None,
) -> tuple[float, ...]:
    if isinstance(generation, GenerationRecord):
        if embedding_mode == EmbeddingMode.CONTEXTUAL:
            if generation.contextual_embedding:
                return generation.contextual_embedding
            if generation.embedding:
                return generation.embedding
        elif generation.embedding:
            return generation.embedding
    if provider is not None:
        return provider.embed(generation)
    msg = "embedding vector or EmbeddingProvider is required"
    raise TypeError(msg)
