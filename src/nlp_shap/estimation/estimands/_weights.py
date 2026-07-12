"""Shared helpers for estimand aggregation."""

import math
from collections.abc import Callable, Sequence


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

    payoff_lookup = {
        tuple(mask): payoff for mask, payoff in zip(masks, payoffs, strict=True)
    }
    values = [0.0] * num_players

    for player_index in range(num_players):
        total = 0.0
        for mask, payoff in zip(masks, payoffs, strict=True):
            if mask[player_index]:
                continue

            coalition_size = sum(mask)
            extended_mask = list(mask)
            extended_mask[player_index] = True
            extended_payoff = payoff_lookup.get(tuple(extended_mask))
            if extended_payoff is None:
                continue

            marginal = extended_payoff - payoff
            total += weight_fn(coalition_size, num_players) * marginal
        values[player_index] = total

    return values
