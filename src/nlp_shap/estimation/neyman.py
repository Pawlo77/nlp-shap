"""Neyman complementary allocation estimator."""

import math
from collections.abc import Iterator, Sequence
from enum import IntEnum
from typing import cast

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ._shared import (
    aggregate_complementary_shapley,
    build_c_matrix_from_pairs,
    build_c_squared_from_pairs,
    compute_complementary_num_splits,
    increment_m_counts,
    is_empty_or_grand,
    present_to_mask_int,
    random_present,
)
from .complementary import ComplementaryEstimator


class _NeymanStep(IntEnum):
    """Phases in the Neyman allocation process."""

    INITIAL_SAMPLING = 1
    NEYMAN_ALLOCATION = 2


class NeymanEstimator(ComplementaryEstimator):
    """Two-phase complementary estimator with Neyman coalition-size allocation."""

    def __init__(
        self,
        initial_fraction: float | None = None,
        initial_num_samples: int | None = None,
        use_standard_method: bool = False,
    ) -> None:
        super().__init__()
        self.initial_fraction = initial_fraction
        self.initial_num_samples = initial_num_samples
        self.use_standard_method = use_standard_method
        self._use_default_initial_formula = (
            initial_fraction is None and initial_num_samples is None
        )
        self._step = _NeymanStep.INITIAL_SAMPLING
        self._initial_num_splits = 0
        self._position_i = 0
        self._position_j = 0
        self._first_call = True
        self._m_hat: np.ndarray | None = None
        self._c_matrix: np.ndarray | None = None
        self._c_squared: np.ndarray | None = None
        self._total_num_splits = 0
        self._initial_masks_count = 0
        self._num_players = 0
        self._rng: np.random.Generator | None = None
        self._seen: set[int] | None = None

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "neyman_cc"

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        """Attach the conversation snapshot under explanation."""
        self._snapshot = snapshot

    def _resolve_initial_num_splits(self, num_players: int, total_splits: int) -> int:
        if self._use_default_initial_formula:
            initial = max(2, math.ceil(total_splits / (2 * num_players * num_players)))
        elif self.initial_num_samples is not None:
            initial = self.initial_num_samples
        elif self.initial_fraction is not None:
            initial = int(
                compute_complementary_num_splits(
                    num_players,
                    self.initial_fraction,
                    include_minimal_masks=False,
                )
            )
        else:
            msg = "initial_fraction or initial_num_samples must be set."
            raise ValueError(msg)
        return max(2, initial)

    def _reset_neyman_state(
        self,
        num_players: int,
        budget_fraction: float,
        seed: int,
    ) -> None:
        self.reset_sampling_state(num_players)
        self._num_players = num_players
        self._total_num_splits = compute_complementary_num_splits(
            num_players,
            budget_fraction,
            include_minimal_masks=False,
        )
        self._initial_num_splits = self._resolve_initial_num_splits(
            num_players,
            self._total_num_splits,
        )
        self._step = _NeymanStep.INITIAL_SAMPLING
        self._position_i = 0
        self._position_j = 0
        self._first_call = True
        self._m_hat = None
        self._c_matrix = None
        self._c_squared = None
        self._initial_masks_count = 0
        self._rng = np.random.default_rng(seed)
        self._seen = set()

    def _allocation_start_index(self) -> int:
        m_counts = self._m_counts
        if m_counts is None:
            msg = "M matrix must be initialized before allocation."
            raise RuntimeError(msg)
        return cast(int, math.ceil(m_counts.shape[0] / 2))

    def _update_m_position(self) -> bool:
        """Advance the initial-phase cursor.

        Return True when initial sampling completes.
        """
        m_counts = self._m_counts
        if m_counts is None:
            msg = "M matrix must be initialized before sampling."
            raise RuntimeError(msg)
        incomplete = m_counts < self._initial_num_splits
        if not np.any(incomplete):
            return True

        flat = incomplete.reshape(-1)
        start_index = self._position_i * m_counts.shape[1] + self._position_j + 1
        next_indices = np.flatnonzero(flat[start_index:])
        if next_indices.size == 0:
            next_indices = np.flatnonzero(flat)
            start_index = 0
        next_index = int(next_indices[0]) + start_index
        self._position_i, self._position_j = divmod(
            next_index,
            m_counts.shape[1],
        )
        return False

    def _next_initial_split(self) -> tuple[bool, ...] | None:
        rng = self._rng
        m_counts = self._m_counts
        if rng is None or m_counts is None:
            msg = "Neyman sampling state must be initialized."
            raise RuntimeError(msg)
        coalition_size = self._position_j
        player_index = self._position_i

        if self.use_standard_method:
            return random_present(
                rng,
                self._num_players,
                true_count=coalition_size,
            )

        include_index = player_index if coalition_size > 0 else None
        present = random_present(
            rng,
            self._num_players,
            true_count=coalition_size,
            include_index=include_index,
        )
        if coalition_size > 0 and not present[player_index]:
            msg = "Generated mask does not include the required player."
            raise RuntimeError(msg)
        return present

    def _next_allocation_split(self) -> tuple[bool, ...] | None:
        rng = self._rng
        m_hat = self._m_hat
        if rng is None or m_hat is None:
            msg = "Neyman allocation state must be initialized."
            raise RuntimeError(msg)
        while self._position_j < m_hat.shape[0]:
            remaining = int(m_hat[self._position_j])
            if remaining > 0:
                m_hat[self._position_j] = remaining - 1
                return random_present(
                    rng,
                    self._num_players,
                    true_count=self._position_j,
                )
            self._position_j += 1
        return None

    def _yield_complementary_pair(
        self,
        present: tuple[bool, ...],
    ) -> Iterator[CoalitionMask]:
        m_counts = self._m_counts
        seen = self._seen
        if m_counts is None or seen is None:
            msg = "Neyman sampling state must be initialized."
            raise RuntimeError(msg)
        complement = tuple(not value for value in present)
        mask_int = present_to_mask_int(present)
        complement_int = present_to_mask_int(complement)
        if mask_int in seen or complement_int in seen:
            return
        seen.add(mask_int)
        seen.add(complement_int)
        increment_m_counts(m_counts, present)
        increment_m_counts(m_counts, complement)
        yield CoalitionMask.from_sequence(present)
        yield CoalitionMask.from_sequence(complement)

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield phase-one Neyman masks until the initial M-matrix grid is filled."""
        if include_minimal_masks:
            msg = "Neyman estimator does not support include_minimal_masks=True"
            raise ValueError(msg)
        self._reset_neyman_state(player_set.num_players, budget_fraction, seed)
        generated = 0

        while generated < self._total_num_splits:
            if self._step == _NeymanStep.INITIAL_SAMPLING:
                if generated >= self._total_num_splits:
                    break
                if not self._first_call and self._update_m_position():
                    self._initial_masks_count = generated
                    return
                self._first_call = False
                present = self._next_initial_split()
                if present is None:
                    break
                if is_empty_or_grand(present):
                    continue
                for mask in self._yield_complementary_pair(present):
                    generated += 1
                    yield mask
                continue
            break

    def begin_allocation(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
    ) -> None:
        """Estimate Neyman allocation after phase-one payoffs are available."""
        if self._m_counts is None:
            msg = "M matrix must be initialized before allocation."
            raise RuntimeError(msg)
        self._c_matrix = build_c_matrix_from_pairs(self._m_counts, masks, payoffs)
        self._c_squared = build_c_squared_from_pairs(self._m_counts, masks, payoffs)
        remaining_pairs = (self._total_num_splits - len(masks)) / 2
        if remaining_pairs < 0 or int(remaining_pairs) != remaining_pairs:
            msg = "Remaining samples for Neyman allocation must be an even integer."
            raise RuntimeError(msg)
        self._estimate_m_hat(int(remaining_pairs))
        self._step = _NeymanStep.NEYMAN_ALLOCATION
        self._position_j = self._allocation_start_index()
        self._position_i = 0

    def sample_allocation_masks(self) -> Iterator[CoalitionMask]:
        """Yield phase-two masks according to the estimated Neyman allocation."""
        if self._step != _NeymanStep.NEYMAN_ALLOCATION:
            msg = "Neyman allocation has not been initialized."
            raise RuntimeError(msg)
        if self._m_hat is None:
            msg = "M_hat must be initialized before allocation sampling."
            raise RuntimeError(msg)

        planned = int(self._m_hat.sum())
        generated = 0
        while generated < planned:
            present = self._next_allocation_split()
            if present is None:
                break
            if is_empty_or_grand(present):
                continue
            for mask in self._yield_complementary_pair(present):
                generated += 1
                yield mask

    def _estimate_m_hat(self, remaining_pairs: int) -> None:
        m_counts = self._m_counts
        c_matrix = self._c_matrix
        c_squared = self._c_squared
        if m_counts is None or c_matrix is None or c_squared is None:
            msg = "Neyman allocation inputs must be initialized."
            raise RuntimeError(msg)
        m_matrix = m_counts.astype(np.float64)
        m_small = m_matrix - 1.0
        sigma_squared = np.zeros_like(m_matrix, dtype=np.float64)
        valid = m_small > 0
        sigma_squared[valid] = (
            c_squared[valid] - np.square(c_matrix[valid]) / m_matrix[valid]
        ) / m_small[valid]
        sigma_squared = np.clip(sigma_squared, a_min=0.0, a_max=None)

        left = self._allocation_start_index()
        right = sigma_squared.shape[0]
        coalition_indices = np.arange(left, right, dtype=np.int64)
        k_left = coalition_indices.astype(np.float64)
        k_right = (right - coalition_indices - 1).astype(np.float64)
        sigma_left = sigma_squared[:, coalition_indices]
        sigma_right = sigma_squared[:, k_right.astype(np.int64)]
        inner = np.sqrt(
            np.sum(sigma_left / (k_left + 1.0), axis=0)
            + np.sum(sigma_right / (k_right + 1.0), axis=0)
        )
        m_hat = np.zeros(m_counts.shape[0], dtype=np.float64)
        denominator = inner.sum()
        if denominator > 0:
            m_hat[left:right] = np.ceil((remaining_pairs / denominator) * inner)
        self._m_hat = m_hat

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate Neyman CC payoffs into Shapley-style attributions."""
        if self._m_counts is None:
            msg = "M matrix must be initialized before calculating attributions."
            raise RuntimeError(msg)
        c_matrix = build_c_matrix_from_pairs(self._m_counts, masks, payoffs)
        allocation_complete = self._step == _NeymanStep.NEYMAN_ALLOCATION
        return aggregate_complementary_shapley(
            self._m_counts,
            c_matrix,
            allocation_complete=allocation_complete,
        )
