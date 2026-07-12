"""Generative backend protocol."""

from typing import Protocol

from ..domain.conversation import ConversationSnapshot


class GenerationResult(Protocol):
    """Minimal generation output consumed by value functions."""

    @property
    def text(self) -> str:
        """Return the generated text payload."""


class GenerativeBackend(Protocol):
    """Execute model generation for a masked conversation snapshot."""

    @property
    def model_id(self) -> str:
        """Return the backend model identifier."""

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> GenerationResult:
        """Generate model output for the given snapshot."""
