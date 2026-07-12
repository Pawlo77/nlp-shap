"""Tests for cooperative-game domain types."""

from nlp_shap.domain.game import CooperativeGame
from nlp_shap.domain.players import PlayerSet


def test_cooperative_game_stores_player_set_and_empty_value() -> None:
    """Cooperative games retain the player set and empty-coalition reference."""
    players = PlayerSet(player_ids=("t0", "t1"))
    game = CooperativeGame(player_set=players, empty_value=0.0)
    assert game.player_set is players
    assert game.empty_value == 0.0
