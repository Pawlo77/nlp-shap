"""Shapley estimand aggregation."""

from collections.abc import Sequence

from ...domain.estimands import Estimand
from ._weights import (
    aggregate_from_marginals,
    aggregate_shapley_exact,
    is_complete_characteristic_table,
    shapley_weight,
)


class ShapleyAggregator:
    """Aggregate coalition samples with Shapley coalition weights."""

    @property
    def estimand(self) -> Estimand:
        """Return the Shapley estimand label."""
        return Estimand.SHAPLEY

    def coalition_weight(self, coalition_size: int, num_players: int) -> float:
        """Return Shapley coalition weight k!(n-k-1)!/n!."""
        return shapley_weight(coalition_size, num_players)

    def aggregate(
        self,
        masks: Sequence[Sequence[bool]],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate coalition samples into Shapley values."""
        if is_complete_characteristic_table(masks):
            return aggregate_shapley_exact(masks, payoffs)
        return aggregate_from_marginals(masks, payoffs, weight_fn=shapley_weight)
