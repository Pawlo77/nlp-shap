"""Shared helpers for estimand aggregation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def _factorial_table(num_players: int) -> list[float]:
    """Return factorials 0! … num_players! as floats."""
    values = [1.0]
    for value in range(1, num_players + 1):
        values.append(values[-1] * float(value))
    return values


def _mask_as_int(mask: Sequence[bool]) -> int:
    """Encode a coalition mask as a little-endian bitset."""
    value = 0
    for index, present in enumerate(mask):
        if present:
            value |= 1 << index
    return value


def is_complete_characteristic_table(masks: Sequence[Sequence[bool]]) -> bool:
    """Return whether masks include every coalition for n players."""
    num_players = len(masks[0])
    expected = 1 << num_players
    if len(masks) != expected:
        return False
    seen = {_mask_as_int(mask) for mask in masks}
    return len(seen) == expected


def shapley_weight(coalition_size: int, num_players: int) -> float:
    """Return Shapley coalition weight k!(n-k-1)!/n!."""
    coalition_size_without_player = coalition_size
    return (
        math.factorial(coalition_size_without_player)
        * math.factorial(num_players - coalition_size_without_player - 1)
        / math.factorial(num_players)
    )


def banzhaf_weight(_coalition_size: int, num_players: int) -> float:
    """Return uniform Banzhaf coalition weight 1/2^(n-1)."""
    return 1.0 / float(2 ** (num_players - 1))


def aggregate_shapley_exact(
    masks: Sequence[Sequence[bool]],
    payoffs: Sequence[float],
) -> list[float]:
    """Compute Shapley values from a complete characteristic-function table."""
    if not masks:
        raise ValueError("masks must not be empty")
    if len(masks) != len(payoffs):
        raise ValueError("masks and payoffs must have the same length")

    array = np.asarray(masks, dtype=np.bool_)
    if array.ndim != 2:
        msg = "masks must be a 2D coalition table"
        raise ValueError(msg)

    num_features = int(array.shape[1])
    pay = np.asarray(payoffs, dtype=np.float64)
    shap_values = np.zeros(num_features, dtype=np.float64)
    indices = np.arange(num_features + 1, dtype=np.float64)
    indices[0] = 1.0
    factorials = np.cumprod(indices)
    power_of_2 = 2 ** np.arange(num_features, dtype=np.int64)
    subset_hashes = (array.astype(np.int64) * power_of_2).sum(axis=1)
    sorted_hashes = np.sort(subset_hashes)
    sort_idx = np.argsort(subset_hashes)
    sorted_outputs = pay[sort_idx]
    subset_sizes = array.sum(axis=1)

    for player_index in range(num_features):
        include_mask = array[:, player_index]
        included_outputs = pay[include_mask]
        included_hashes = subset_hashes[include_mask]
        excluded_hash = included_hashes - power_of_2[player_index]
        excluded_outputs = sorted_outputs[np.searchsorted(sorted_hashes, excluded_hash)]
        excluded_subset_sizes = subset_sizes[include_mask] - 1
        weights = (
            factorials[excluded_subset_sizes]
            * factorials[num_features - excluded_subset_sizes - 1]
            / factorials[num_features]
        )
        shap_values[player_index] = np.sum(
            weights * (included_outputs - excluded_outputs)
        )
    return shap_values.tolist()


def aggregate_from_marginals(
    masks: Sequence[Sequence[bool]],
    payoffs: Sequence[float],
    weight_fn: Callable[[int, int], float],
) -> list[float]:
    """Sum weighted marginal contributions for each player.

    Args:
        masks: Coalition membership rows with shape ``(m, n)``.
        payoffs: Characteristic-function samples ``v(S)`` aligned with masks.
        weight_fn: Coalition weight callback receiving ``(coalition_size, n)``.

    Returns:
        Per-player attribution vector with length ``n``.
    """
    if not masks:
        raise ValueError("masks must not be empty")
    if len(masks) != len(payoffs):
        raise ValueError("masks and payoffs must have the same length")

    num_players = len(masks[0])
    if num_players == 0:
        raise ValueError("masks must include at least one player")
    if any(len(mask) != num_players for mask in masks):
        raise ValueError("all masks must have the same number of players")

    payoff_lookup: dict[int, float] = {}
    mask_ints: list[int] = []
    for mask, payoff in zip(masks, payoffs, strict=True):
        mask_int = _mask_as_int(mask)
        mask_ints.append(mask_int)
        payoff_lookup[mask_int] = payoff

    factorials = _factorial_table(num_players)
    values = [0.0] * num_players

    for player_index in range(num_players):
        player_bit = 1 << player_index
        total = 0.0
        for mask_int, payoff in zip(mask_ints, payoffs, strict=True):
            if mask_int & player_bit:
                continue

            coalition_size = mask_int.bit_count()
            extended_int = mask_int | player_bit
            extended_payoff = payoff_lookup.get(extended_int)
            if extended_payoff is None:
                continue

            marginal = extended_payoff - payoff
            if weight_fn is shapley_weight:
                weight = (
                    factorials[coalition_size]
                    * factorials[num_players - coalition_size - 1]
                    / factorials[num_players]
                )
            else:
                weight = weight_fn(coalition_size, num_players)
            total += weight * marginal
        values[player_index] = total

    return values
