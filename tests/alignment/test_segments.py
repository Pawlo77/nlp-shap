"""Tests for aligned audio segments."""

import pytest

from nlp_shap.alignment.segments import AudioSegment


def test_audio_segment_duration() -> None:
    """Duration is end_time minus start_time."""
    segment = AudioSegment(
        token="hello",
        start_time=0.25,
        end_time=1.0,
        confidence=0.9,
    )
    assert segment.duration == pytest.approx(0.75)


def test_audio_segment_rejects_inverted_times() -> None:
    """end_time must be greater than start_time."""
    with pytest.raises(ValueError, match="end_time"):
        AudioSegment(token="x", start_time=1.0, end_time=0.5, confidence=0.5)


def test_audio_segment_add_merges_matching_tokens() -> None:
    """Adding matching tokens merges ranges, confidence, and audio bytes."""
    left = AudioSegment(
        token="a",
        start_time=0.2,
        end_time=0.3,
        confidence=0.6,
        audio=b"x",
        boundary_refined=True,
    )
    right = AudioSegment(
        token="a",
        start_time=0.1,
        end_time=0.4,
        confidence=0.8,
        audio=b"y",
        boundary_refined=False,
    )
    merged = left + right
    assert merged.token == "a"
    assert merged.start_time == pytest.approx(0.1)
    assert merged.end_time == pytest.approx(0.4)
    assert merged.confidence == pytest.approx(0.7)
    assert merged.audio == b"xy"
    assert merged.boundary_refined is False


def test_audio_segment_add_rejects_mismatched_tokens() -> None:
    """Segments with different tokens cannot be combined."""
    left = AudioSegment(token="a", start_time=0.0, end_time=0.1, confidence=0.7)
    right = AudioSegment(token="b", start_time=0.1, end_time=0.2, confidence=0.8)
    with pytest.raises(ValueError, match="different tokens"):
        _ = left + right
