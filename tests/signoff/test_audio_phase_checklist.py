"""Sign-off checklist for audio-track Phases 15-17."""

from nlp_shap import (
    AudioPayload,
    ConversationSnapshot,
    Message,
    ModalityFlag,
    ModelHistoryTrackingMode,
    Role,
    Turn,
)
from nlp_shap.alignment import (
    AudioSegment,
    SgpaSegmentPartitioner,
    SpectrogramGuidedAligner,
    TorchAudioHandler,
)
from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.enums import SystemRolesSetup
from nlp_shap.masking import (
    ExcludePunctuationTokensFilter,
    KeepAllTokens,
    MaskBuilder,
    SegmentDeletePolicy,
)
from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins


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


def test_phase17_filters_and_sgpa_partition_plugin() -> None:
    """Phase 17 token filters and sgpa_segments partition are wired."""
    assert KeepAllTokens().phrases_to_exclude == frozenset()
    punct = ExcludePunctuationTokensFilter()
    assert punct.keeps(".") is False
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.PARTITIONS)
    partitioner = registry.resolve(PluginGroup.PARTITIONS, "sgpa_segments")
    assert isinstance(partitioner, SgpaSegmentPartitioner)


def test_phase17_audio_masking_produces_masked_snapshot() -> None:
    """Masking SGPA segments yields a valid multimodal MaskedSnapshot."""
    payload = AudioPayload(data=b"RIFFDATA", sample_rate_hz=16_000, audio_format="wav")
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(
                    role=Role.USER,
                    text="hello world",
                    modality=ModalityFlag.AUDIO,
                    audio=payload,
                ),
            )
        ),
    ))
    segments = (
        AudioSegment(token="hello", start_time=0.0, end_time=0.1, confidence=1.0),
        AudioSegment(token="world", start_time=0.1, end_time=0.2, confidence=1.0),
    )
    partitioner = SgpaSegmentPartitioner(segments=segments)
    players = partitioner.partition(snapshot)
    mask = CoalitionMask.from_sequence((True, False))
    view = MaskBuilder(SegmentDeletePolicy(segments=segments)).view(
        snapshot, players, mask
    )
    assert view.base.has_audio() is True
    assert view.players.num_players == 2
