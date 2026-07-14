"""Exact coalition enumeration estimator."""

from collections.abc import Iterator, Sequence

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

    @staticmethod
    def iter_mask_ints(num_players: int) -> Iterator[int]:
        """Lazily yield coalition bitmasks except the all-present coalition."""
        grand_mask = (1 << num_players) - 1
        for value in range(1 << num_players):
            if value == grand_mask:
                continue
            yield value

    @staticmethod
    def mask_int_to_present(mask_int: int, num_players: int) -> tuple[bool, ...]:
        """Decode a coalition bitmask into presence flags."""
        return tuple(bool((mask_int >> index) & 1) for index in range(num_players))

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        """Attach the conversation snapshot under explanation."""
        self._snapshot = snapshot

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield every coalition mask except the grand coalition."""
        if budget_fraction != 1.0:
            msg = "exact estimator requires budget_fraction == 1.0"
            raise ValueError(msg)
        _ = include_minimal_masks, seed
        return self.iter_masks(player_set)

    @staticmethod
    def iter_masks(player_set: PlayerSet) -> Iterator[CoalitionMask]:
        """Lazily yield coalition masks except the all-present coalition."""
        num_players = player_set.num_players
        for mask_int in ExactEstimator.iter_mask_ints(num_players):
            present = ExactEstimator.mask_int_to_present(mask_int, num_players)
            yield CoalitionMask.from_sequence(present)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Aggregate coalition payoffs with the selected estimand plugin."""
        return aggregator.aggregate([mask.present for mask in masks], payoffs)
