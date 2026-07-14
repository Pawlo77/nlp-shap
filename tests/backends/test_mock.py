"""Tests for the deterministic mock backend."""

import asyncio

from nlp_shap.backends.mock import MockBackend
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def test_mock_backend_is_deterministic() -> None:
    """Identical inputs produce identical generation records."""
    backend = MockBackend(model_id="stub")
    snapshot = _snapshot("hello world")

    async def run() -> tuple[str, str]:
        first = await backend.generate(snapshot, 8, 0.0, 1)
        second = await backend.generate(snapshot, 8, 0.0, 1)
        return first.text, second.text

    first_text, second_text = asyncio.run(run())
    assert first_text == second_text


def test_mock_backend_varies_with_masked_text() -> None:
    """Different masked snapshots produce different generations."""
    backend = MockBackend(model_id="stub")
    full = _snapshot("alpha beta gamma")
    partial = _snapshot("alpha gamma")

    async def run() -> tuple[str, str]:
        full_record = await backend.generate(full, 8, 0.0, 1)
        partial_record = await backend.generate(partial, 8, 0.0, 1)
        return full_record.text, partial_record.text

    full_text, partial_text = asyncio.run(run())
    assert full_text != partial_text


def test_mock_backend_exposes_model_id() -> None:
    """Backend reports the configured model identifier."""
    backend = MockBackend(model_id="custom-mock")
    assert backend.model_id == "custom-mock"
