"""Tests for alignment audio I/O helpers."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from nlp_shap.alignment.io import TorchAudioHandler  # noqa: E402


@pytest.fixture
def mono_waveform() -> "torch.Tensor":
    """Return a short mono waveform tensor."""
    return torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)


@patch("nlp_shap.alignment.io._require_soundfile")
def test_from_bytes_returns_mono_waveform(
    mock_require_sf: MagicMock,
    mono_waveform: "torch.Tensor",
) -> None:
    """from_bytes loads bytes into a mono [1, T] tensor."""
    sf = MagicMock()
    sf.read.return_value = (mono_waveform.squeeze(0).numpy(), 16_000)
    mock_require_sf.return_value = sf

    waveform, sample_rate = TorchAudioHandler.from_bytes(b"fake", audio_format="wav")

    assert waveform.shape == (1, 4)
    assert sample_rate == 16_000
    assert isinstance(sf.read.call_args.args[0], BytesIO)


@patch("nlp_shap.alignment.io._require_soundfile")
def test_from_bytes_averages_stereo(
    mock_require_sf: MagicMock,
) -> None:
    """Stereo arrays are collapsed to mono by channel mean."""
    stereo = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        dtype=torch.float32,
    )
    sf = MagicMock()
    sf.read.return_value = (stereo.T.numpy(), 16_000)
    mock_require_sf.return_value = sf

    waveform, sample_rate = TorchAudioHandler.from_bytes(b"fake")

    expected = stereo.mean(dim=0, keepdim=True)
    torch.testing.assert_close(waveform, expected)
    assert sample_rate == 16_000


@patch("nlp_shap.alignment.io._require_soundfile")
def test_to_bytes_writes_wav(
    mock_require_sf: MagicMock,
    mono_waveform: "torch.Tensor",
) -> None:
    """to_bytes encodes mono waveforms as WAV via soundfile."""
    sf = MagicMock()
    mock_require_sf.return_value = sf

    def _write(buf: BytesIO, *_args: object, **_kwargs: object) -> None:
        buf.write(b"RIFF")

    sf.write.side_effect = _write

    encoded = TorchAudioHandler.to_bytes(mono_waveform, sample_rate=16_000)
    assert encoded == b"RIFF"
    assert sf.write.called


def test_to_bytes_rejects_non_wav(mono_waveform: "torch.Tensor") -> None:
    """Non-WAV formats raise ValueError."""
    with pytest.raises(ValueError, match="unsupported audio_format"):
        TorchAudioHandler.to_bytes(mono_waveform, audio_format="mp3")


@patch("nlp_shap.alignment.io._require_soundfile")
@patch("nlp_shap.alignment.io._require_torchaudio")
def test_combine_concatenates_segment_waveforms(
    mock_require_ta: MagicMock,
    mock_require_sf: MagicMock,
    mono_waveform: "torch.Tensor",
) -> None:
    """combine joins segment payloads along the time axis."""
    from types import SimpleNamespace

    sf = MagicMock()
    sf.read.side_effect = [
        (mono_waveform.squeeze(0).numpy(), 16_000),
        (mono_waveform.squeeze(0).numpy(), 16_000),
    ]

    def _write(buf: BytesIO, *_args: object, **_kwargs: object) -> None:
        buf.write(b"COMBINED")

    sf.write.side_effect = _write
    mock_require_sf.return_value = sf
    mock_require_ta.return_value = MagicMock()

    segments = [
        SimpleNamespace(audio=b"a", audio_format="wav"),
        SimpleNamespace(audio=b"b", audio_format="wav"),
    ]
    combined = TorchAudioHandler.combine(segments)
    assert combined == b"COMBINED"
    assert sf.read.call_count == 2
