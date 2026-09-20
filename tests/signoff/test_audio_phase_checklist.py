"""Sign-off checklist for audio-track Phases 15-16."""

from nlp_shap import (
    AudioPayload,
    ConversationSnapshot,
    Message,
    ModalityFlag,
    ModelHistoryTrackingMode,
    Role,
    Turn,
)
from nlp_shap.alignment import AudioSegment, SpectrogramGuidedAligner, TorchAudioHandler
from nlp_shap.domain.enums import SystemRolesSetup


def test_phase15_multimodal_domain_enums_exported() -> None:
    """Phase 15 domain enums are public and stable on the wire."""
    assert ModalityFlag.AUDIO.value == "audio"
    assert ModelHistoryTrackingMode.TEXT_AUDIO.value == "text_audio"
    assert SystemRolesSetup.SYSTEM_ASSISTANT.value == "system_assistant"


def test_phase15_multimodal_snapshot_round_trip() -> None:
    """Audio Message payloads survive snapshot construction and has_audio."""
    payload = AudioPayload(data=b"RIFFDATA", sample_rate_hz=16_000, audio_format="wav")
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(
                    role=Role.USER,
                    text="hello",
                    modality=ModalityFlag.AUDIO,
                    audio=payload,
                ),
            )
        ),
    ))
    assert snapshot.has_audio() is True
    assert snapshot.turns[0].messages[0].audio == payload


def test_phase15_text_only_api_unchanged() -> None:
    """Text-only Message construction still defaults to TEXT modality."""
    message = Message(role=Role.USER, text="plain")
    assert message.modality is ModalityFlag.TEXT
    assert message.audio is None
    snapshot = ConversationSnapshot.from_turns((Turn(messages=(message,)),))
    assert snapshot.has_audio() is False


def test_phase16_alignment_surface_present() -> None:
    """Phase 16 alignment package exports SGPA building blocks."""
    assert AudioSegment.__name__ == "AudioSegment"
    assert SpectrogramGuidedAligner.__name__ == "SpectrogramGuidedAligner"
    assert TorchAudioHandler.__name__ == "TorchAudioHandler"
    assert hasattr(TorchAudioHandler, "combine")
    assert hasattr(SpectrogramGuidedAligner, "normalize_text")
