"""Tests for SGPA segment player partition plugin."""

import pytest

from nlp_shap.alignment.partitions import SgpaSegmentPartitioner
from nlp_shap.alignment.segments import AudioSegment
from nlp_shap.domain.conversation import (
    AudioPayload,
    ConversationSnapshot,
    Message,
    Turn,
)
from nlp_shap.domain.enums import ModalityFlag, Role
from nlp_shap.masking.filters import ExcludePunctuationTokensFilter, KeepAllTokens
from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins


def _audio_snapshot(text: str = "hello world") -> ConversationSnapshot:
    payload = AudioPayload(data=b"RIFFDATA", sample_rate_hz=16_000, audio_format="wav")
    return ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(
                    role=Role.USER,
                    text=text,
                    modality=ModalityFlag.AUDIO,
                    audio=payload,
                ),
            )
        ),
    ))


def _segment(token: str, start: float, end: float) -> AudioSegment:
    return AudioSegment(token=token, start_time=start, end_time=end, confidence=0.9)


def test_sgpa_partitioner_name_and_players() -> None:
    """Aligned segments become ordered explainability players."""
    snapshot = _audio_snapshot()
    segments = (_segment("hello", 0.0, 0.2), _segment("world", 0.2, 0.4))
    partitioner = SgpaSegmentPartitioner(segments=segments)
    players = partitioner.partition(snapshot)
    assert partitioner.name == "sgpa_segments"
    assert players.num_players == 2
    assert players.player_ids[0] == f"{snapshot.snapshot_id}:sgpa:0"
    assert players.player_ids[1] == f"{snapshot.snapshot_id}:sgpa:1"


def test_sgpa_partitioner_filters_punctuation_tokens() -> None:
    """Punctuation segments are dropped before building the player set."""
    snapshot = _audio_snapshot("hello , world")
    segments = (
        _segment("hello", 0.0, 0.2),
        _segment(",", 0.2, 0.25),
        _segment("world", 0.25, 0.5),
    )
    partitioner = SgpaSegmentPartitioner(
        segments=segments,
        token_filter=ExcludePunctuationTokensFilter(),
    )
    players = partitioner.partition(snapshot)
    assert players.num_players == 2
    assert partitioner.filtered_segments() == (segments[0], segments[2])


def test_sgpa_partitioner_keep_all_retains_punctuation() -> None:
    """KeepAllTokens retains punctuation segments as players."""
    snapshot = _audio_snapshot()
    segments = (_segment("hello", 0.0, 0.2), _segment("!", 0.2, 0.3))
    partitioner = SgpaSegmentPartitioner(
        segments=segments,
        token_filter=KeepAllTokens(),
    )
    assert partitioner.partition(snapshot).num_players == 2


def test_sgpa_partitioner_rejects_text_only_snapshot() -> None:
    """SGPA partition requires at least one audio payload."""
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="hello world"),)),
    ))
    partitioner = SgpaSegmentPartitioner(segments=(_segment("hello", 0.0, 0.1),))
    with pytest.raises(ValueError, match="audio"):
        partitioner.partition(snapshot)


def test_sgpa_partitioner_rejects_empty_filtered_segments() -> None:
    """All-punctuation segment layouts are rejected."""
    snapshot = _audio_snapshot()
    partitioner = SgpaSegmentPartitioner(
        segments=(_segment(".", 0.0, 0.1),),
        token_filter=ExcludePunctuationTokensFilter(),
    )
    with pytest.raises(ValueError, match="at least one"):
        partitioner.partition(snapshot)


def test_sgpa_segments_plugin_registered() -> None:
    """Builtin + entry-point registration exposes sgpa_segments."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.PARTITIONS)
    partitioner = registry.resolve(PluginGroup.PARTITIONS, "sgpa_segments")
    assert isinstance(partitioner, SgpaSegmentPartitioner)
    assert partitioner.name == "sgpa_segments"
