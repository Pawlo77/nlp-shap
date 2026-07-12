"""Estimator strategy protocol."""

from typing import Protocol

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet


class EstimatorStrategy(Protocol):
    """Sample coalitions and drive coalition evaluation for an explain run."""

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""

    def sample_masks(
        self,
        player_set: PlayerSet,
        *,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> tuple[CoalitionMask, ...]:
        """Return coalition masks to evaluate for the player set."""

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        """Attach the conversation snapshot under explanation."""
