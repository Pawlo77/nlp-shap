"""Shared sampling utilities for approximate estimators."""

import math
from collections.abc import Iterator, Sequence

import numpy as np

from ..domain.coalition import CoalitionMask


def present_to_mask_int(present: Sequence[bool]) -> int:
    """Encode a coalition presence vector as a little-endian bitmask."""
    value = 0
    for index, is_present in enumerate(present):
        if is_present:
            value |= 1 << index
    return value


def mask_int_to_present(mask_int: int, num_players: int) -> tuple[bool, ...]:
    """Decode a coalition bitmask into presence flags."""
    return tuple(bool((mask_int >> index) & 1) for index in range(num_players))


def mc_coalition_budget(num_players: int) -> int:
    """Return the Monte Carlo coalition budget excluding the grand coalition."""
    if num_players < 1:
        msg = "num_players must be at least 1"
        raise ValueError(msg)
    return (1 << num_players) - 1


def complementary_coalition_budget(num_players: int) -> int:
    """Return the complementary sampling budget excluding empty and grand masks."""
    if num_players < 1:
        msg = "num_players must be at least 1"
        raise ValueError(msg)
    return (1 << num_players) - 2


def compute_mc_num_samples(
    num_players: int,
    budget_fraction: float,
    include_minimal_masks: bool,
) -> int:
    """Return the Monte Carlo sample count capped by the coalition budget."""
    maximum = mc_coalition_budget(num_players)
    minimum = num_players + 1 if include_minimal_masks else 1
    target = int(maximum * budget_fraction)
    target = max(minimum, min(target, maximum))
    return target


def compute_complementary_num_splits(
    num_players: int,
    budget_fraction: float,
    include_minimal_masks: bool,
) -> int:
    """Return an even complementary mask budget derived from ``budget_fraction``."""
    maximum = complementary_coalition_budget(num_players)
    minimum = 2 * num_players
    target = int(maximum * budget_fraction)
    if target < minimum:
        target = minimum
    if target > maximum:
        return maximum
    if target % 2 == 1:
        target -= 1
    if target < minimum:
        return minimum if minimum % 2 == 0 else minimum - 1
    _ = include_minimal_masks
    return target


def iter_minimal_masks(num_players: int) -> Iterator[tuple[bool, ...]]:
    """Yield the empty coalition followed by each singleton coalition."""
    yield (False,) * num_players
    for index in range(num_players):
        present = [False] * num_players
        present[index] = True
        yield tuple(present)


def random_present(
    rng: np.random.Generator,
    num_players: int,
    true_count: int | None = None,
    include_index: int | None = None,
) -> tuple[bool, ...]:
    """Sample a random coalition, optionally fixing size and one included player."""
    if true_count is None:
        bits = rng.integers(0, 2, size=num_players, dtype=np.int8)
        return tuple(bool(value) for value in bits)

    if include_index is not None:
        other_count = num_players - 1
        adjusted_count = true_count - 1
        if adjusted_count < 0 or adjusted_count > other_count:
            msg = "true_count incompatible with include_index"
            raise ValueError(msg)
        other_indices = [
            index for index in range(num_players) if index != include_index
        ]
        chosen = rng.choice(other_indices, size=adjusted_count, replace=False)
        present = [False] * num_players
        present[include_index] = True
        for index in chosen:
            present[int(index)] = True
        return tuple(present)

    if true_count < 0 or true_count > num_players:
        msg = "true_count out of range"
        raise ValueError(msg)
    chosen = rng.choice(num_players, size=true_count, replace=False)
    present = [False] * num_players
    for index in chosen:
        present[int(index)] = True
    return tuple(present)


def complementary_base_present(
    rng: np.random.Generator,
    num_players: int,
    include_index: int,
) -> tuple[bool, ...]:
    """Sample a complementary base coalition with one fixed included player."""
    true_count = math.ceil(num_players / 3)
    return random_present(
        rng,
        num_players,
        true_count=true_count,
        include_index=include_index,
    )


def is_empty_or_grand(present: Sequence[bool]) -> bool:
    """Return whether the coalition is empty or includes every player."""
    coalition_size = sum(present)
    return coalition_size == 0 or coalition_size == len(present)


def increment_m_counts(m_counts: np.ndarray, present: Sequence[bool]) -> None:
    """Increment complementary M-matrix counts for one coalition."""
    coalition_size = sum(present)
    if coalition_size == 0:
        m_counts[:, 0] += 1
        return
    for index, is_present in enumerate(present):
        if is_present:
            m_counts[index, coalition_size] += 1


def accumulate_complementary_contribution(
    c_matrix: np.ndarray,
    s_present: Sequence[bool],
    ns_present: Sequence[bool],
    contribution: float,
) -> None:
    """Accumulate one complementary pair contribution into matrix C."""
    s_size = sum(s_present)
    ns_size = sum(ns_present)
    if s_size == 0:
        c_matrix[:, 0] += contribution
    else:
        for index, is_present in enumerate(s_present):
            if is_present:
                c_matrix[index, s_size] += contribution
    if ns_size == 0:
        c_matrix[:, 0] -= contribution
    else:
        for index, is_present in enumerate(ns_present):
            if is_present:
                c_matrix[index, ns_size] -= contribution


def accumulate_complementary_contribution_squared(
    c_squared: np.ndarray,
    s_present: Sequence[bool],
    ns_present: Sequence[bool],
    contribution: float,
) -> None:
    """Accumulate squared complementary contributions for Neyman variance estimates."""
    contribution_squared = contribution * contribution
    s_size = sum(s_present)
    ns_size = sum(ns_present)
    if s_size == 0:
        c_squared[:, 0] += contribution_squared
    else:
        for index, is_present in enumerate(s_present):
            if is_present:
                c_squared[index, s_size] += contribution_squared
    if ns_size == 0:
        c_squared[:, 0] += contribution_squared
    else:
        for index, is_present in enumerate(ns_present):
            if is_present:
                c_squared[index, ns_size] += contribution_squared


def aggregate_complementary_shapley(
    m_counts: np.ndarray,
    c_matrix: np.ndarray,
    allocation_complete: bool = False,
) -> list[float]:
    """Aggregate complementary M/C matrices into Shapley-style attributions."""
    m_work = m_counts[:, 1:]
    c_work = c_matrix[:, 1:]
    num_players = m_counts.shape[0]
    values = np.zeros(num_players, dtype=np.float64)
    for player_index in range(num_players):
        counts = m_work[player_index]
        contributions = c_work[player_index]
        if allocation_complete:
            positive = counts > 0
            values[player_index] = (
                np.sum(
                    np.divide(
                        contributions[positive],
                        counts[positive],
                        dtype=np.float64,
                    )
                )
                / num_players
            )
            continue
        ratio = np.zeros_like(counts, dtype=np.float64)
        positive = counts > 0
        ratio[positive] = contributions[positive] / counts[positive]
        values[player_index] = ratio.sum() / num_players
    return values.tolist()


def build_c_matrix_from_pairs(
    m_counts: np.ndarray,
    masks: Sequence[CoalitionMask],
    payoffs: Sequence[float],
) -> np.ndarray:
    """Build matrix C from complementary mask pairs and coalition payoffs."""
    if len(masks) % 2 != 0:
        msg = "Masks should be in complementary pairs."
        raise ValueError(msg)
    c_matrix = np.zeros_like(m_counts, dtype=np.float64)
    pair_count = len(masks) // 2
    for pair_index in range(pair_count):
        s_mask = masks[2 * pair_index]
        ns_mask = masks[2 * pair_index + 1]
        s_present = s_mask.present
        ns_present = ns_mask.present
        if tuple(not value for value in s_present) != ns_present:
            msg = "Masks are not complementary pairs."
            raise ValueError(msg)
        contribution = payoffs[2 * pair_index] - payoffs[2 * pair_index + 1]
        accumulate_complementary_contribution(
            c_matrix,
            s_present,
            ns_present,
            contribution,
        )
    return c_matrix


def build_c_squared_from_pairs(
    m_counts: np.ndarray,
    masks: Sequence[CoalitionMask],
    payoffs: Sequence[float],
) -> np.ndarray:
    """Build the squared complementary matrix used by Neyman allocation."""
    if len(masks) % 2 != 0:
        msg = "Masks should be in complementary pairs."
        raise ValueError(msg)
    c_squared = np.zeros_like(m_counts, dtype=np.float64)
    pair_count = len(masks) // 2
    for pair_index in range(pair_count):
        s_mask = masks[2 * pair_index]
        ns_mask = masks[2 * pair_index + 1]
        s_present = s_mask.present
        ns_present = ns_mask.present
        if tuple(not value for value in s_present) != ns_present:
            msg = "Masks are not complementary pairs."
            raise ValueError(msg)
        contribution = payoffs[2 * pair_index] - payoffs[2 * pair_index + 1]
        accumulate_complementary_contribution_squared(
            c_squared,
            s_present,
            ns_present,
            contribution,
        )
    return c_squared
