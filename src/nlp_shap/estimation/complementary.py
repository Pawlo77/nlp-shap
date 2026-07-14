"""Complementary pair sampling estimator."""

from collections.abc import Iterator, Sequence

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ._shared import (
    aggregate_complementary_shapley,
    build_c_matrix_from_pairs,
    complementary_base_present,
    compute_complementary_num_splits,
    increment_m_counts,
    is_empty_or_grand,
    present_to_mask_int,
    random_present,
)


class ComplementaryEstimator:
    """Sample complementary coalition pairs and aggregate CC attributions."""

    def __init__(self) -> None:
        self._snapshot: ConversationSnapshot | None = None
        self._m_counts: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "complementary"

    @property
    def m_counts(self) -> np.ndarray | None:
        """Return the latest complementary M-matrix counts from sampling."""
        return None if self._m_counts is None else self._m_counts.copy()

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        """Attach the conversation snapshot under explanation."""
        self._snapshot = snapshot

    def reset_sampling_state(self, num_players: int) -> None:
        """Reset complementary M-matrix counts before a new sampling run."""
        self._m_counts = np.zeros((num_players, num_players + 1), dtype=np.int64)

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield complementary coalition pairs up to the configured budget."""
        num_players = player_set.num_players
        num_splits = compute_complementary_num_splits(
            num_players,
            budget_fraction,
            include_minimal_masks,
        )
        self.reset_sampling_state(num_players)
        m_counts = self._m_counts
        if m_counts is None:
            msg = "M matrix must be initialized before sampling."
            raise RuntimeError(msg)
        rng = np.random.default_rng(seed)
        seen: set[int] = set()
        generated = 0
        base_index = 0

        while generated < num_splits:
            if include_minimal_masks and base_index < num_players:
                present = complementary_base_present(rng, num_players, base_index)
                base_index += 1
            else:
                present = random_present(rng, num_players)

            if is_empty_or_grand(present):
                continue

            mask_int = present_to_mask_int(present)
            complement_int = present_to_mask_int(tuple(not value for value in present))
            if mask_int in seen or complement_int in seen:
                continue

            if generated + 2 > num_splits:
                return

            complement = tuple(not value for value in present)
            seen.add(mask_int)
            seen.add(complement_int)

            increment_m_counts(m_counts, present)
            increment_m_counts(m_counts, complement)
            generated += 2
            yield CoalitionMask.from_sequence(present)
            yield CoalitionMask.from_sequence(complement)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate complementary pair payoffs into CC Shapley attributions."""
        if self._m_counts is None:
            msg = "M matrix must be initialized before calculating attributions."
            raise RuntimeError(msg)
        c_matrix = build_c_matrix_from_pairs(self._m_counts, masks, payoffs)
        return aggregate_complementary_shapley(self._m_counts, c_matrix)
