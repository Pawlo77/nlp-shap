"""Tests for conversation snapshots."""

import hashlib
import json

import pytest

from nlp_shap.domain.conversation import (
    AudioPayload,
    ConversationSnapshot,
    Message,
    Turn,
)
from nlp_shap.domain.enums import ModalityFlag, Role


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


def test_text_message_defaults_to_text_modality() -> None:
    """Text-only constructors keep modality TEXT and no audio payload."""
    message = Message(role=Role.USER, text="hello")
    assert message.modality is ModalityFlag.TEXT
    assert message.audio is None


def test_audio_message_requires_payload() -> None:
    """AUDIO modality messages must carry a non-empty audio payload."""
    with pytest.raises(ValueError, match="require an audio payload"):
        Message(role=Role.USER, text="", modality=ModalityFlag.AUDIO)


def test_text_message_rejects_audio_payload() -> None:
    """TEXT modality messages must not carry audio bytes."""
    payload = AudioPayload(data=b"RIFF", sample_rate_hz=16000, audio_format="wav")
    with pytest.raises(ValueError, match="must not carry audio"):
        Message(
            role=Role.USER,
            text="hello",
            modality=ModalityFlag.TEXT,
            audio=payload,
        )


def test_audio_payload_rejects_empty_data() -> None:
    """Audio payloads require non-empty encoded bytes."""
    with pytest.raises(ValueError, match="non-empty"):
        AudioPayload(data=b"", sample_rate_hz=16000, audio_format="wav")


def test_audio_payload_rejects_non_positive_sample_rate() -> None:
    """Audio payloads require a positive sample rate."""
    with pytest.raises(ValueError, match="sample_rate_hz"):
        AudioPayload(data=b"RIFF", sample_rate_hz=0, audio_format="wav")


def test_multimodal_snapshot_round_trip_preserves_audio() -> None:
    """Multimodal snapshots retain audio payloads and stable ids."""
    payload = AudioPayload(
        data=b"RIFF....wav-bytes",
        sample_rate_hz=16000,
        audio_format="wav",
    )
    turn = Turn(
        messages=(
            Message(
                role=Role.USER,
                text="hello",
                modality=ModalityFlag.AUDIO,
                audio=payload,
            ),
        )
    )
    first = ConversationSnapshot.from_turns((turn,))
    second = ConversationSnapshot.from_turns((turn,))
    assert first.snapshot_id == second.snapshot_id
    assert first.has_audio() is True
    assert first.turns[0].messages[0].audio == payload


def test_audio_bytes_change_snapshot_id() -> None:
    """Distinct audio payloads produce distinct snapshot identifiers."""
    base = Message(
        role=Role.USER,
        text="hello",
        modality=ModalityFlag.AUDIO,
        audio=AudioPayload(data=b"aaa", sample_rate_hz=16000, audio_format="wav"),
    )
    other = Message(
        role=Role.USER,
        text="hello",
        modality=ModalityFlag.AUDIO,
        audio=AudioPayload(data=b"bbb", sample_rate_hz=16000, audio_format="wav"),
    )
    first = ConversationSnapshot.from_turns((Turn(messages=(base,)),))
    second = ConversationSnapshot.from_turns((Turn(messages=(other,)),))
    assert first.snapshot_id != second.snapshot_id


def test_text_only_snapshot_id_matches_legacy_digest() -> None:
    """Text-only from_turns ids stay identical to the pre-multimodal digest."""
    turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
    snapshot = ConversationSnapshot.from_turns((turn,))
    legacy_payload = [[{"role": "user", "text": "Who are you?"}]]
    encoded = json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    assert snapshot.snapshot_id == expected


def test_text_only_snapshot_reports_no_audio() -> None:
    """Text-only snapshots report has_audio as false."""
    turn = Turn(messages=(Message(role=Role.USER, text="plain text"),))
    snapshot = ConversationSnapshot.from_turns((turn,))
    assert snapshot.has_audio() is False
