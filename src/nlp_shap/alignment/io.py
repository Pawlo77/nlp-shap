"""Audio byte ↔ waveform helpers for SGPA alignment."""

from io import BytesIO
from typing import Any

TARGET_SAMPLE_RATE: int = 24_000
"""Default sample rate for serialization helpers."""

_MONO_CHANNELS: int = 1
"""Channel count for mono waveforms."""

_WAVEFORM_2D_DIM: int = 2
"""Rank of channel-first waveform tensors ``[C, T]``."""

_AUDIO_FORMAT_WAV: str = "wav"
"""WAV format label."""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required for alignment I/O; install nlp-shap[audio]"
        raise ImportError(msg) from exc
    return torch


def _require_soundfile() -> Any:
    try:
        import soundfile as sf
    except ImportError as exc:
        msg = "soundfile is required for alignment I/O; install nlp-shap[audio]"
        raise ImportError(msg) from exc
    return sf


class TorchAudioHandler:
    """Convert between encoded audio bytes and mono float waveforms."""

    @staticmethod
    def from_bytes(
        audio_content: bytes,
        audio_format: str = "wav",
    ) -> tuple[Any, int]:
        """Decode ``audio_content`` into a mono ``[1, T]`` waveform tensor."""
        del audio_format
        torch = _require_torch()
        sf = _require_soundfile()
        try:
            waveform_np, sample_rate = sf.read(BytesIO(audio_content))
        except Exception as exc:
            msg = "failed to decode audio bytes"
            raise ValueError(msg) from exc

        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == _WAVEFORM_2D_DIM:
            waveform = waveform.T
        if waveform.shape[0] > _MONO_CHANNELS:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, int(sample_rate)

    @staticmethod
    def to_bytes(
        waveform: Any,
        sample_rate: int = TARGET_SAMPLE_RATE,
        audio_format: str = "wav",
    ) -> bytes:
        """Encode a waveform tensor as audio bytes."""
        torch = _require_torch()
        sf = _require_soundfile()
        with torch.no_grad():
            wf = waveform.detach().cpu()
            if wf.dim() == _WAVEFORM_2D_DIM and wf.size(0) > 1:
                wf = wf.mean(dim=0, keepdim=True)
            if wf.dim() == _WAVEFORM_2D_DIM and wf.size(0) == 1:
                wf = wf.squeeze(0)
            elif wf.dim() == 1:
                pass
            else:
                wf = wf.mean(dim=0)
            wf = wf.to(torch.float32)
            wf = torch.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-1.0, 1.0)
            wf = wf.contiguous()

        fmt = audio_format.lower()
        if fmt != _AUDIO_FORMAT_WAV:
            msg = f"unsupported audio_format: {audio_format!r}"
            raise ValueError(msg)
        buf = BytesIO()
        sf.write(buf, wf.numpy(), int(sample_rate), format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
