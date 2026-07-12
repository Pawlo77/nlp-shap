"""Estimand aggregation protocol."""

from collections.abc import Sequence
from typing import Protocol

from ..domain.estimands import Estimand


class EstimandAggregator(Protocol):
    """Aggregate coalition payoffs into per-player attributions."""

    @property
    def estimand(self) -> Estimand:
        """Return the estimand label produced by this aggregator."""

    def coalition_weight(self, coalition_size: int, num_players: int) -> float:
        """Return the weight for a coalition of the given size.

        Args:
            coalition_size: Number of players in coalition ``S`` excluding the
                focal player ``i``.
            num_players: Total number of players ``n`` in the game.

        Returns:
            Coalition weight used when summing marginal contributions.
        """

    def aggregate(
        self,
        masks: Sequence[Sequence[bool]],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate coalition samples into per-player values.

        Args:
            masks: Coalition membership rows with shape ``(m, n)``.
            payoffs: Characteristic-function samples ``v(S)`` aligned with masks.

        Returns:
            Per-player attribution vector with length ``n``.
        """
