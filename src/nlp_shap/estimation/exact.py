"""Exact coalition enumeration estimator."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator


class ExactEstimator:
    """Enumerate all coalitions and delegate attribution to an estimand plugin."""

    def __init__(self) -> None:
        self._snapshot: ConversationSnapshot | None = None

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "exact"

    @staticmethod
    def num_coalitions(num_players: int) -> int:
        """Return the number of coalitions evaluated by exact enumeration."""
        if num_players < 1:
            msg = "num_players must be at least 1"
            raise ValueError(msg)
        return int(2**num_players - 1)

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        """Attach the conversation snapshot under explanation."""
        self._snapshot = snapshot

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> tuple[CoalitionMask, ...]:
        """Return every coalition mask except the grand coalition."""
        if budget_fraction != 1.0:
            msg = "exact estimator requires budget_fraction == 1.0"
            raise ValueError(msg)
        _ = include_minimal_masks, seed
        return self.enumerate_masks(player_set)

    @staticmethod
    def enumerate_masks(player_set: PlayerSet) -> tuple[CoalitionMask, ...]:
        """Build all coalition masks except the all-present coalition."""
        num_players = player_set.num_players
        masks: list[CoalitionMask] = []
        for bits in product([False, True], repeat=num_players):
            if all(bits):
                continue
            mask = CoalitionMask.from_sequence(bits)
            mask.validate_against(player_set)
            masks.append(mask)
        return tuple(masks)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Aggregate coalition payoffs with the selected estimand plugin."""
        bool_masks = [mask.present for mask in masks]
        return aggregator.aggregate(bool_masks, list(payoffs))
