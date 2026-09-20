"""Tests for SpectrogramGuidedAligner."""

from unittest.mock import MagicMock, patch

import pytest

from nlp_shap.alignment.segments import AudioSegment
from nlp_shap.alignment.sgpa import SpectrogramGuidedAligner


def _patched_transformers() -> tuple[MagicMock, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {"A": 0, "B": 1, "|": 2}
    tokenizer.pad_token_id = 3
    tokenizer.convert_tokens_to_ids.side_effect = lambda char: {"A": 0, "B": 1, "|": 2}[
        char
    ]
    tokenizer.convert_ids_to_tokens.side_effect = lambda token_id: {
        0: "A",
        1: "B",
        2: "|",
    }[token_id]

    processor = MagicMock()
    processor.tokenizer = tokenizer
    processor_cls = MagicMock()
    processor_cls.from_pretrained.return_value = processor

    model = MagicMock()
    model.to.return_value = model
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model
    return processor_cls, model_cls


def test_normalize_text_strips_diacritics_and_punct() -> None:
    """normalize_text uppercases and drops non-alnum marks."""
    assert SpectrogramGuidedAligner.normalize_text("café!") == "CAFE"


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_init_loads_model_and_vocab(mock_require: MagicMock) -> None:
    """Constructor loads processor/model and stores vocab + blank id."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)

    aligner = SpectrogramGuidedAligner(device="cpu", model_name="dummy-model")

    processor_cls.from_pretrained.assert_called_once_with(
        "dummy-model", revision="main"
    )
    model_cls.from_pretrained.assert_called_once_with("dummy-model", revision="main")
    assert aligner.vocab == {"A": 0, "B": 1, "|": 2}
    assert aligner.blank_id == 3
    assert aligner.boundary_energy_weight == pytest.approx(0.8)


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_init_rejects_invalid_boundary_weights(mock_require: MagicMock) -> None:
    """Boundary weights must be non-negative and not both zero."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)

    with pytest.raises(ValueError, match="non-negative"):
        SpectrogramGuidedAligner(
            device="cpu",
            boundary_energy_weight=-0.1,
            boundary_flux_weight=1.0,
        )
    with pytest.raises(ValueError, match="greater than zero"):
        SpectrogramGuidedAligner(
            device="cpu",
            boundary_energy_weight=0.0,
            boundary_flux_weight=0.0,
        )


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_init_raises_on_oserror(mock_require: MagicMock) -> None:
    """Model load OSError becomes a ValueError with a clear message."""
    processor_cls = MagicMock()
    processor_cls.from_pretrained.side_effect = OSError("missing")
    model_cls = MagicMock()
    mock_require.return_value = (processor_cls, model_cls)

    with pytest.raises(ValueError, match="Could not load"):
        SpectrogramGuidedAligner(device="cpu", model_name="bad")


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_merge_tokens_skips_blanks(mock_require: MagicMock) -> None:
    """Blank frames are excluded from character spans."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    path = MagicMock()
    path.tolist.return_value = [0, 0, 3, 3, 1, 1]
    spans = aligner._merge_tokens(path, blank_id=3)
    assert spans == [(0, 0, 2), (1, 4, 6)]


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_prepare_transcript_rejects_empty_vocab_hit(mock_require: MagicMock) -> None:
    """Transcripts with no vocab characters raise ValueError."""
    processor_cls, model_cls = _patched_transformers()
    processor = processor_cls.from_pretrained.return_value
    processor.tokenizer.get_vocab.return_value = {"A": 0}
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    with pytest.raises(ValueError, match="no valid characters"):
        aligner._prepare_transcript("@@@")


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_call_orchestrates_pipeline(mock_require: MagicMock) -> None:
    """__call__ wires prepare → align → merge → refine → aggregate → attach."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")

    waveform = object()
    final = [
        AudioSegment(token="AB", start_time=0.0, end_time=1.0, confidence=0.9),
    ]
    with (
        patch.object(
            aligner,
            "_prepare_transcript",
            return_value=("AB", ["AB"], "AB", [0, 1]),
        ) as mock_prepare,
        patch.object(
            aligner,
            "_perform_forced_alignment",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_align,
        patch.object(
            aligner,
            "_merge_tokens",
            return_value=[(0, 0, 2), (1, 2, 4)],
        ) as mock_merge,
        patch.object(
            aligner,
            "_refine_token_spans",
            return_value=[
                {
                    "char": "A",
                    "start": 0.0,
                    "end": 0.5,
                    "confidence": 0.9,
                    "boundary_refined": True,
                },
                {
                    "char": "B",
                    "start": 0.5,
                    "end": 1.0,
                    "confidence": 0.8,
                    "boundary_refined": True,
                },
            ],
        ) as mock_refine,
        patch.object(
            aligner,
            "_aggregate_chars_to_segments",
            return_value=final,
        ) as mock_agg,
        patch.object(
            aligner,
            "_attach_audio_to_segments",
            return_value=final,
        ) as mock_attach,
    ):
        out = aligner(
            transcript="AB",
            waveform=waveform,
            original_sr=16_000,
            attach_audio=True,
        )

    mock_prepare.assert_called_once()
    mock_align.assert_called_once()
    mock_merge.assert_called_once()
    mock_refine.assert_called_once()
    mock_agg.assert_called_once()
    mock_attach.assert_called_once_with(final, waveform, 16_000, attach_audio=True)
    assert out is final


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_call_requires_audio_input(mock_require: MagicMock) -> None:
    """Missing audio input raises ValueError."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    with pytest.raises(ValueError, match="audio_content or both waveform"):
        aligner(transcript="hello")


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_attach_audio_writes_wav_bytes(mock_require: MagicMock) -> None:
    """attach_audio_to_segments returns segments with non-empty WAV payloads."""
    torch = pytest.importorskip("torch")
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    sr = 16_000
    waveform = torch.zeros(1, int(0.2 * sr))
    segments = [
        AudioSegment(token="a", start_time=0.0, end_time=0.05, confidence=1.0),
        AudioSegment(token="b", start_time=0.05, end_time=0.1, confidence=0.5),
    ]
    updated = aligner.attach_audio_to_segments(
        segments, waveform=waveform, original_sr=sr
    )
    assert len(updated) == 2
    for segment in updated:
        assert segment.audio.startswith(b"RIFF")
        assert segment.sample_rate == sr
        assert segment.start_sample is not None
        assert segment.end_sample is not None
        assert segment.end_sample > segment.start_sample


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_refine_boundary_short_region_returns_candidate(
    mock_require: MagicMock,
) -> None:
    """Short search regions skip refinement (0.x parity)."""
    import numpy as np

    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    out, refined = aligner._refine_boundary_smart(
        waveform=np.zeros(10, dtype=np.float32),
        sr=16_000,
        candidate_time=0.1,
    )
    assert out == pytest.approx(0.1)
    assert refined is False


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_refine_boundary_no_silence_keeps_candidate(mock_require: MagicMock) -> None:
    """No quiet frame → candidate time unchanged (0.x parity)."""
    import numpy as np

    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    librosa = MagicMock()
    librosa.feature.rms.return_value = np.ones((1, 10))
    librosa.stft.return_value = np.ones((4, 10))
    with (
        patch("nlp_shap.alignment.sgpa._require_librosa", return_value=librosa),
        patch("nlp_shap.alignment.sgpa._require_numpy", return_value=np),
    ):
        out, refined = aligner._refine_boundary_smart(
            waveform=np.ones(4000, dtype=np.float32),
            sr=1000,
            candidate_time=1.0,
            left_time=0.8,
            right_time=1.2,
        )
    assert out == pytest.approx(1.0)
    assert refined is False


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_refine_boundary_silence_matches_legacy_timestamp(
    mock_require: MagicMock,
) -> None:
    """Quiet RMS minimum yields the same refined time as 0.x (±1 ms)."""
    import numpy as np

    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    librosa = MagicMock()
    librosa.feature.rms.return_value = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    librosa.stft.return_value = np.ones((4, 3))
    with (
        patch("nlp_shap.alignment.sgpa._require_librosa", return_value=librosa),
        patch("nlp_shap.alignment.sgpa._require_numpy", return_value=np),
    ):
        out, refined = aligner._refine_boundary_smart(
            waveform=np.ones(4000, dtype=np.float32),
            sr=1000,
            candidate_time=1.0,
            left_time=0.6,
            right_time=1.4,
        )
    # 0.x expected ~0.624-0.625 s for this synthetic RMS pattern
    assert out == pytest.approx(0.624, abs=0.002)
    assert refined is True


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_prepare_transcript_accepts_token_list(mock_require: MagicMock) -> None:
    """List transcripts join while preserving per-token segments."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    vocab_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ|"
    aligner.vocab = {char: index for index, char in enumerate(vocab_chars)}
    aligner.tokenizer.convert_tokens_to_ids = lambda char: (
        vocab_chars.index(char) if char in vocab_chars else -1
    )
    full, segments, clean, tokens = aligner._prepare_transcript(["Hi", "all"])
    assert full == "Hi all"
    assert segments == ["Hi", "all"]
    assert clean == "HI|ALL"
    assert len(tokens) == len(clean)


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_aggregate_chars_skips_empty_and_propagates_refine_flag(
    mock_require: MagicMock,
) -> None:
    """Aggregation skips empty targets and ANDs boundary_refined."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    char_segments = [
        {
            "char": "H",
            "start": 0.0,
            "end": 0.1,
            "confidence": 1.0,
            "boundary_refined": True,
        },
        {
            "char": "E",
            "start": 0.1,
            "end": 0.2,
            "confidence": 0.5,
            "boundary_refined": False,
        },
    ]
    out = aligner._aggregate_chars_to_segments(
        char_segments, target_segments=["", "HE", "MISS"]
    )
    assert len(out) == 1
    assert out[0].token == "HE"
    assert out[0].start_time == pytest.approx(0.0)
    assert out[0].end_time == pytest.approx(0.2)
    assert out[0].confidence == pytest.approx(0.75)
    assert out[0].boundary_refined is False


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_golden_segment_boundaries_within_tolerance(mock_require: MagicMock) -> None:
    """Controlled refine+aggregate fixture locks 0.x-compatible boundaries."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(device="cpu")
    char_segments = [
        {
            "char": "H",
            "start": 0.10,
            "end": 0.25,
            "confidence": 0.9,
            "boundary_refined": True,
        },
        {
            "char": "I",
            "start": 0.25,
            "end": 0.40,
            "confidence": 0.8,
            "boundary_refined": True,
        },
        {
            "char": "A",
            "start": 0.45,
            "end": 0.70,
            "confidence": 0.7,
            "boundary_refined": True,
        },
    ]
    segments = aligner._aggregate_chars_to_segments(
        char_segments, target_segments=["HI", "A"]
    )
    assert len(segments) == 2
    assert segments[0].token == "HI"
    assert segments[0].start_time == pytest.approx(0.10, abs=1e-3)
    assert segments[0].end_time == pytest.approx(0.40, abs=1e-3)
    assert segments[1].token == "A"
    assert segments[1].start_time == pytest.approx(0.45, abs=1e-3)
    assert segments[1].end_time == pytest.approx(0.70, abs=1e-3)
