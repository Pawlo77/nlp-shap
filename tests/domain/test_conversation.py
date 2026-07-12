"""Tests for conversation snapshots."""

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role


def test_conversation_snapshot_from_turns_is_stable() -> None:
    """Identical turns produce the same snapshot identifier."""
    turn = Turn(
        messages=(
            Message(role=Role.USER, text="Who"),
            Message(role=Role.USER, text="are"),
            Message(role=Role.USER, text="you?"),
        )
    )
    first = ConversationSnapshot.from_turns((turn,))
    second = ConversationSnapshot.from_turns((turn,))
    assert first.snapshot_id == second.snapshot_id
    assert first == second


def test_conversation_snapshot_rejects_empty_turns() -> None:
    """Snapshots require at least one turn."""
    with pytest.raises(ValueError, match="at least one turn"):
        ConversationSnapshot(turns=(), snapshot_id="abc")


def test_turn_rejects_empty_messages() -> None:
    """Turns require at least one message."""
    with pytest.raises(ValueError, match="at least one message"):
        Turn(messages=())
