"""Cooperative-game domain types."""

from __future__ import annotations

from dataclasses import dataclass

from .players import PlayerSet


@dataclass(frozen=True, slots=True)
class CooperativeGame:
    """Player set and reference values for a characteristic function."""

    player_set: PlayerSet
    """Ordered explainability players that define the game."""

    empty_value: float = 0.0
    """Reference payoff :math:`v(\\emptyset)` for the characteristic function."""
