"""Value function protocol."""

from typing import Protocol

from .backend import GenerationResult


class ValueFunction(Protocol):
    """Score coalition utility relative to a base generation."""

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return utility for ``candidate`` relative to ``base``."""
