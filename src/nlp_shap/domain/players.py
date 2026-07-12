"""Player indexing for cooperative-game explainability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerSet:
    """Ordered explainability players aligned with coalition masks."""

    player_ids: tuple[str, ...]
    """Stable player identifiers in coalition-mask order."""

    def __post_init__(self) -> None:
        if not self.player_ids:
            msg = "player set must contain at least one player"
            raise ValueError(msg)
        if len(set(self.player_ids)) != len(self.player_ids):
            msg = "player ids must be unique"
            raise ValueError(msg)

    @property
    def num_players(self) -> int:
        """Return the number of players in the set."""
        return len(self.player_ids)
