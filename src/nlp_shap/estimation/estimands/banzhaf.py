"""Banzhaf estimand aggregation."""

from collections.abc import Sequence

from ...domain.estimands import Estimand
from ._weights import aggregate_from_marginals, banzhaf_weight


class BanzhafAggregator:
    """Aggregate coalition samples with uniform Banzhaf coalition weights."""

    @property
    def estimand(self) -> Estimand:
        """Return the Banzhaf estimand label."""
        return Estimand.BANZHAF

    def coalition_weight(self, coalition_size: int, num_players: int) -> float:
        """Return uniform Banzhaf coalition weight 1/2^(n-1)."""
        return banzhaf_weight(coalition_size, num_players)

    def aggregate(
        self,
        masks: Sequence[Sequence[bool]],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate coalition samples into Banzhaf indices."""
        return aggregate_from_marginals(masks, payoffs, weight_fn=banzhaf_weight)
