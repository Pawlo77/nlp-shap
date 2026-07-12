"""Tests for token partition plugins."""

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.masking.partitions import TokenPartitioner


def test_token_partitioner_splits_whitespace_tokens(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Whitespace tokens become ordered explainability players."""
    partitioner = TokenPartitioner()
    players = partitioner.partition(sample_snapshot)
    assert partitioner.name == "tokens"
    assert players.num_players == 6
    assert players.player_ids[0].startswith(sample_snapshot.snapshot_id)


def test_token_partitioner_rejects_empty_text() -> None:
    """Snapshots without explainable tokens are rejected."""
    turn = Turn(messages=(Message(role=Role.USER, text="   "),))
    snapshot = ConversationSnapshot.from_turns((turn,))
    with pytest.raises(ValueError, match="at least one explainable token"):
        TokenPartitioner().partition(snapshot)
