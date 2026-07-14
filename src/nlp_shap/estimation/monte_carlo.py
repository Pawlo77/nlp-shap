"""Monte Carlo coalition sampling estimator."""

from collections.abc import Iterator, Sequence

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator
from ._shared import (
    compute_mc_num_samples,
    is_empty_or_grand,
    iter_minimal_masks,
    present_to_mask_int,
    random_present,
)


class MonteCarloEstimator:
    """Sample random coalitions and delegate attribution to an estimand plugin."""

    def __init__(self) -> None:
        self._snapshot: ConversationSnapshot | None = None

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "mc"

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
        """Yield random coalition masks up to the configured budget."""
        num_players = player_set.num_players
        num_samples = compute_mc_num_samples(
            num_players,
            budget_fraction,
            include_minimal_masks,
        )
        rng = np.random.default_rng(seed)
        seen: set[int] = set()
        generated = 0
        grand_mask = present_to_mask_int((True,) * num_players)

        if include_minimal_masks:
            for present in iter_minimal_masks(num_players):
                mask_int = present_to_mask_int(present)
                if mask_int in seen:
                    continue
                seen.add(mask_int)
                generated += 1
                yield CoalitionMask.from_sequence(present)
                if generated >= num_samples:
                    return

        while generated < num_samples:
            present = random_present(rng, num_players)
            if is_empty_or_grand(present):
                continue
            mask_int = present_to_mask_int(present)
            if mask_int in seen:
                continue
            if mask_int == grand_mask:
                continue
            seen.add(mask_int)
            generated += 1
            yield CoalitionMask.from_sequence(present)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Aggregate sampled coalition payoffs with the selected estimand plugin."""
        return aggregator.aggregate([mask.present for mask in masks], payoffs)
