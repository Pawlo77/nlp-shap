"""Shared conversation fixtures for masking tests."""

from __future__ import annotations

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role


@pytest.fixture
def sample_snapshot() -> ConversationSnapshot:
    """Return a short multi-token conversation snapshot."""
    turn = Turn(
        messages=(
            Message(role=Role.USER, text="Who are you?"),
            Message(role=Role.ASSISTANT, text="I am helpful."),
        )
    )
    return ConversationSnapshot.from_turns((turn,))
