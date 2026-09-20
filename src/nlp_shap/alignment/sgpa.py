"""Spectrogram-guided forced aligner using Wav2Vec2 CTC."""

import io
import logging
import unicodedata
import warnings
import wave
from typing import Any, TypedDict, cast

from .io import TorchAudioHandler
from .segments import AudioSegment

logger = logging.getLogger(__name__)

ASCII_SPACE: str = " "
"""ASCII space used in transcript normalization and CTC separators."""

_UNICODE_NONSPACING_MARK: str = "Mn"
"""Unicode category stripped when removing diacritics."""

_SILENCE_THRESHOLD_RATIO: float = 0.5
"""Max RMS / mean RMS ratio accepted as a silence cut."""

_MIN_REFINE_SAMPLES: int = 256
"""Minimum samples required to refine a boundary."""

_MIN_SEGMENT_DURATION_S: float = 0.05
"""Floor duration applied when writing sample indices."""


class CharSpan(TypedDict):
    """Character-level alignment span after boundary refinement."""

    char: str
    start: float
    end: float
    confidence: float
    boundary_refined: bool


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        msg = "torch is required for SGPA alignment; install nlp-shap[audio]"
        raise ImportError(msg) from exc
    return torch


def _require_numpy() -> Any:
    import numpy as np

    return np


def _require_librosa() -> Any:
    try:
        import librosa
    except ImportError as exc:
        msg = "librosa is required for SGPA alignment; install nlp-shap[audio]"
        raise ImportError(msg) from exc
    return librosa


def _require_torchaudio() -> Any:
    try:
        import torchaudio
    except ImportError as exc:
        msg = "torchaudio is required for SGPA alignment; install nlp-shap[audio]"
        raise ImportError(msg) from exc
    return torchaudio


def _require_transformers() -> tuple[Any, Any]:
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from transformers import logging as hf_logging
    except ImportError as exc:
        msg = "transformers is required for SGPA alignment; install nlp-shap[audio]"
        raise ImportError(msg) from exc
    hf_logging.set_verbosity_error()
    return Wav2Vec2Processor, Wav2Vec2ForCTC


class SpectrogramGuidedAligner:
    """Align transcript tokens to audio via Wav2Vec2 CTC and spectrogram refine."""

    def __init__(
        self,
        device: Any,
        model_name: str = "facebook/wav2vec2-large-960h",
        model_revision: str = "main",
        sample_rate: int = 16_000,
        ctc_separator: str = "|",
        boundary_energy_weight: float = 0.8,
        boundary_flux_weight: float = 0.2,
    ) -> None:
        if boundary_energy_weight < 0 or boundary_flux_weight < 0:
            msg = "Boundary refinement weights must be non-negative."
            raise ValueError(msg)
        if boundary_energy_weight + boundary_flux_weight <= 0:
            msg = "At least one boundary refinement weight must be greater than zero."
            raise ValueError(msg)

        processor_cls, model_cls = _require_transformers()
        self.device = device
        self.sample_rate = sample_rate
        self.ctc_separator = ctc_separator
        self.boundary_energy_weight = float(boundary_energy_weight)
        self.boundary_flux_weight = float(boundary_flux_weight)

        logger.debug("Loading alignment model %s on %s", model_name, device)
        try:
            self.processor = processor_cls.from_pretrained(
                model_name, revision=model_revision
            )
            model = model_cls.from_pretrained(model_name, revision=model_revision)
            self.model = model.to(device)
        except OSError as exc:
            msg = f"Could not load '{model_name}'. Ensure it is a valid CTC model."
            raise ValueError(msg) from exc

        self.tokenizer = self.processor.tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.blank_id = self.tokenizer.pad_token_id or 0

        warnings.filterwarnings(
            "ignore", message=".*forced_align has been deprecated.*"
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        """Strip diacritics and non-alnum characters; uppercase."""
        text_nfd = unicodedata.normalize("NFD", text)
        no_marks = "".join(
            char
            for char in text_nfd
            if unicodedata.category(char) != _UNICODE_NONSPACING_MARK
        )
        return "".join(filter(str.isalnum, no_marks)).upper()

    def _compute_emissions(self, waveform: Any, original_sr: int) -> Any:
        torch = _require_torch()
        torchaudio = _require_torchaudio()
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.sample_rate
            ).to(self.device)
            waveform = resampler(waveform.to(self.device))
        else:
            waveform = waveform.to(self.device)
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits
            return torch.log_softmax(logits, dim=-1)

    def _refine_boundary_smart(
        self,
        waveform: Any,
        sr: int,
        candidate_time: float,
        left_time: float | None = None,
        right_time: float | None = None,
    ) -> tuple[float, bool]:
        np = _require_numpy()
        librosa = _require_librosa()
        if left_time is not None and right_time is not None:
            center_time = (left_time + right_time) / 2.0
            half_window = (right_time - left_time) / 2.0 + 0.04
        else:
            center_time = candidate_time
            half_window = 0.08

        window_samples = int(half_window * sr)
        center_sample = int(center_time * sr)
        start_idx = max(0, center_sample - window_samples)
        end_idx = min(len(waveform), center_sample + window_samples)
        search_region = waveform[start_idx:end_idx]
        if len(search_region) < _MIN_REFINE_SAMPLES:
            return candidate_time, False

        rms = librosa.feature.rms(y=search_region, frame_length=256, hop_length=64)[0]
        stft = np.abs(librosa.stft(search_region, n_fft=256, hop_length=64))
        flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
        flux = np.pad(flux, (0, len(rms) - len(flux)), mode="constant")
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-9)
        flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-9)
        cost = (
            self.boundary_energy_weight * rms_norm
            + self.boundary_flux_weight * flux_norm
        )
        min_idx = int(np.argmin(cost))
        if rms[min_idx] > _SILENCE_THRESHOLD_RATIO * float(np.mean(rms)):
            return candidate_time, False
        refined_sample = start_idx + (min_idx * 64)
        return refined_sample / sr, True

    def _save_wav_mem(self, tensor: Any, sample_rate: int) -> bytes:
        torch = _require_torch()
        src = tensor.cpu()
        if src.dim() == 1:
            src = src.unsqueeze(0)
        n_channels = src.shape[0]
        src = (src * 32767).clamp(-32768, 32767).to(torch.int16)
        src = src.t().numpy()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(src.tobytes())
        return buffer.getvalue()

    def _merge_tokens(
        self, alignment_path: Any, blank_id: int
    ) -> list[tuple[int, int, int]]:
        path = alignment_path.tolist()
        spans: list[tuple[int, int, int]] = []
        current_token: int | None = None
        start_frame = 0
        for index, token in enumerate(path):
            if token != current_token:
                if current_token is not None and current_token != blank_id:
                    spans.append((current_token, start_frame, index))
                current_token = token
                start_frame = index
        if current_token is not None and current_token != blank_id:
            spans.append((current_token, start_frame, len(path)))
        return spans

    def _prepare_transcript(
        self, transcript: str | list[str]
    ) -> tuple[str, list[str], str, list[int]]:
        if isinstance(transcript, str):
            full_transcript = transcript
            target_segments = transcript.split()
        else:
            target_segments = list(transcript)
            full_transcript = " ".join(transcript)

        text_upper = full_transcript.upper()
        text_nfd = unicodedata.normalize("NFD", text_upper)
        text_no_diacritics = "".join(
            char
            for char in text_nfd
            if unicodedata.category(char) != _UNICODE_NONSPACING_MARK
        )
        text_clean = "".join(
            char for char in text_no_diacritics if char.isalnum() or char == ASCII_SPACE
        )
        clean_text = text_clean.replace(ASCII_SPACE, self.ctc_separator)
        valid_tokens = [
            cast(int, self.tokenizer.convert_tokens_to_ids(char))
            for char in clean_text
            if char in self.vocab
        ]
        if not valid_tokens:
            msg = "Transcript contains no valid characters for this model."
            raise ValueError(msg)
        return full_transcript, target_segments, clean_text, valid_tokens

    def _perform_forced_alignment(
        self, waveform: Any, original_sr: int, valid_tokens: list[int]
    ) -> tuple[Any, Any]:
        torch = _require_torch()
        torchaudio = _require_torchaudio()
        emissions_gpu = self._compute_emissions(waveform, original_sr).squeeze(0)
        emissions_cpu = emissions_gpu.unsqueeze(0).cpu()
        targets_cpu = torch.tensor([valid_tokens], dtype=torch.int32).cpu()
        emission_lens_cpu = torch.tensor([emissions_gpu.size(0)]).cpu()
        target_lens_cpu = torch.tensor([len(valid_tokens)]).cpu()
        aligned_tokens, _scores = torchaudio.functional.forced_align(
            emissions_cpu,
            targets_cpu,
            emission_lens_cpu,
            target_lens_cpu,
            blank=self.blank_id,
        )
        return aligned_tokens[0], emissions_gpu

    def _refine_token_spans(
        self,
        token_spans: list[tuple[int, int, int]],
        emissions_gpu: Any,
        waveform: Any,
        original_sr: int,
    ) -> list[CharSpan]:
        np = _require_numpy()
        torch = _require_torch()
        if not token_spans:
            return []

        ratio = waveform.size(1) / emissions_gpu.size(0)
        numpy_wave = waveform.cpu().numpy().squeeze()
        total_samples = waveform.size(1)
        n_spans = len(token_spans)

        raw_starts: list[float] = []
        raw_ends: list[float] = []
        confidences: list[float] = []
        for token_id, start_frame, end_frame in token_spans:
            raw_starts.append((start_frame * ratio) / original_sr)
            raw_ends.append((end_frame * ratio) / original_sr)
            conf = (
                torch.exp(emissions_gpu[start_frame:end_frame, token_id]).mean().item()
            )
            confidences.append(float(conf))

        refined_boundaries: list[tuple[float, bool]] = []
        for index in range(n_spans - 1):
            gap_left = raw_ends[index]
            gap_right = raw_starts[index + 1]
            gap_mid = (gap_left + gap_right) / 2.0
            refined_time, was_refined = self._refine_boundary_smart(
                numpy_wave,
                original_sr,
                candidate_time=gap_mid,
                left_time=gap_left,
                right_time=gap_right,
            )
            refined_time = float(np.clip(refined_time, gap_left, gap_right))
            refined_boundaries.append((refined_time, was_refined))

        first_start, first_refined = self._refine_boundary_smart(
            numpy_wave, original_sr, raw_starts[0]
        )
        last_end, last_refined = self._refine_boundary_smart(
            numpy_wave,
            original_sr,
            min(raw_ends[-1], total_samples / original_sr),
        )

        refined_chars: list[CharSpan] = []
        for index, (token_id, _start, _end) in enumerate(token_spans):
            if index == 0:
                start_time = first_start
                start_refined = first_refined
            else:
                start_time, start_refined = refined_boundaries[index - 1]
            if index == n_spans - 1:
                end_time = last_end
                end_refined = last_refined
            else:
                end_time, end_refined = refined_boundaries[index]
            if end_time <= start_time:
                end_time = start_time + _MIN_SEGMENT_DURATION_S
            char = cast(str, self.tokenizer.convert_ids_to_tokens(token_id))
            refined_chars.append({
                "char": char,
                "start": float(start_time),
                "end": float(end_time),
                "confidence": confidences[index],
                "boundary_refined": start_refined and end_refined,
            })
        return refined_chars

    def _aggregate_chars_to_segments(
        self,
        char_segments: list[CharSpan],
        target_segments: list[str],
    ) -> list[AudioSegment]:
        final_segments: list[AudioSegment] = []
        current_char_idx = 0
        for segment_text in target_segments:
            clean_target = self.normalize_text(segment_text)
            if not clean_target:
                continue
            start_time: float | None = None
            end_time: float | None = None
            seg_confs: list[float] = []
            all_refined = True
            found_chars = 0
            while found_chars < len(clean_target) and current_char_idx < len(
                char_segments
            ):
                span = char_segments[current_char_idx]
                seg_char = span["char"].replace(self.ctc_separator, "")
                if seg_char == clean_target[found_chars]:
                    if start_time is None:
                        start_time = span["start"]
                    end_time = span["end"]
                    seg_confs.append(span["confidence"])
                    if not span["boundary_refined"]:
                        all_refined = False
                    found_chars += 1
                current_char_idx += 1
            if start_time is None:
                continue
            if end_time is None or end_time <= start_time:
                end_time = start_time + _MIN_SEGMENT_DURATION_S
            avg_conf = sum(seg_confs) / len(seg_confs) if seg_confs else 0.0
            final_segments.append(
                AudioSegment(
                    token=segment_text,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=avg_conf,
                    audio_format="wav",
                    boundary_refined=all_refined,
                )
            )
        return final_segments

    def _with_sample_indices(
        self,
        segments: list[AudioSegment],
        waveform: Any,
        original_sr: int,
    ) -> tuple[list[AudioSegment], Any]:
        cpu_waveform = waveform.cpu()
        if cpu_waveform.dim() == 1:
            cpu_waveform = cpu_waveform.unsqueeze(0)
        min_duration = int(_MIN_SEGMENT_DURATION_S * original_sr)
        updated: list[AudioSegment] = []
        for segment in segments:
            start_sample = int(segment.start_time * original_sr)
            end_sample = int(segment.end_time * original_sr)
            if end_sample - start_sample < min_duration:
                end_sample = start_sample + min_duration
            start_sample = max(0, start_sample)
            end_sample = min(cpu_waveform.size(1), end_sample)
            if end_sample <= start_sample:
                end_sample = min(cpu_waveform.size(1), start_sample + min_duration)
            updated.append(
                AudioSegment(
                    token=segment.token,
                    start_time=segment.start_time,
                    end_time=max(
                        segment.end_time,
                        segment.start_time + _MIN_SEGMENT_DURATION_S,
                    ),
                    confidence=segment.confidence,
                    audio=segment.audio,
                    audio_format=segment.audio_format,
                    sample_rate=original_sr,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    boundary_refined=segment.boundary_refined,
                )
            )
        return updated, cpu_waveform

    def _attach_audio_to_segments(
        self,
        segments: list[AudioSegment],
        waveform: Any,
        original_sr: int,
        attach_audio: bool = True,
    ) -> list[AudioSegment]:
        indexed, cpu_waveform = self._with_sample_indices(
            segments, waveform, original_sr
        )
        if not attach_audio:
            return indexed
        with_audio: list[AudioSegment] = []
        for segment in indexed:
            start_sample = segment.start_sample
            end_sample = segment.end_sample
            if start_sample is None or end_sample is None:
                msg = "segment sample indices are required before attaching audio"
                raise ValueError(msg)
            chunk = cpu_waveform[:, start_sample:end_sample]
            with_audio.append(
                AudioSegment(
                    token=segment.token,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    confidence=segment.confidence,
                    audio=self._save_wav_mem(chunk, original_sr),
                    audio_format="wav",
                    sample_rate=segment.sample_rate,
                    start_sample=segment.start_sample,
                    end_sample=segment.end_sample,
                    boundary_refined=segment.boundary_refined,
                )
            )
        return with_audio

    def attach_audio_to_segments(
        self,
        segments: list[AudioSegment],
        audio_content: bytes | None = None,
        waveform: Any | None = None,
        original_sr: int | None = None,
        audio_format: str = "wav",
    ) -> list[AudioSegment]:
        """Return copies of ``segments`` with WAV bytes attached from audio input."""
        waveform, original_sr = self._resolve_audio_input(
            audio_content=audio_content,
            waveform=waveform,
            original_sr=original_sr,
            audio_format=audio_format,
        )
        return self._attach_audio_to_segments(
            segments, waveform, original_sr, attach_audio=True
        )

    def _resolve_audio_input(
        self,
        audio_content: bytes | None,
        waveform: Any | None,
        original_sr: int | None,
        audio_format: str,
    ) -> tuple[Any, int]:
        if audio_content is None and (waveform is None or original_sr is None):
            msg = (
                "Either audio_content or both waveform and original_sr must be "
                "provided."
            )
            raise ValueError(msg)
        if audio_content is not None:
            return TorchAudioHandler.from_bytes(
                audio_content, audio_format=audio_format
            )
        if waveform is None or original_sr is None:
            msg = (
                "Either audio_content or both waveform and original_sr must be "
                "provided."
            )
            raise ValueError(msg)
        return waveform, original_sr

    def __call__(
        self,
        transcript: str | list[str],
        audio_content: bytes | None = None,
        waveform: Any | None = None,
        original_sr: int | None = None,
        audio_format: str = "wav",
        attach_audio: bool = False,
    ) -> list[AudioSegment]:
        """Align ``transcript`` to audio and return refined token segments."""
        waveform, original_sr = self._resolve_audio_input(
            audio_content=audio_content,
            waveform=waveform,
            original_sr=original_sr,
            audio_format=audio_format,
        )
        _full, target_segments, clean_text, valid_tokens = self._prepare_transcript(
            transcript
        )
        logger.debug("Aligning to transcript %r", clean_text)
        alignment_path, emissions_gpu = self._perform_forced_alignment(
            waveform, original_sr, valid_tokens
        )
        token_spans = self._merge_tokens(alignment_path, self.blank_id)
        refined_chars = self._refine_token_spans(
            token_spans, emissions_gpu, waveform, original_sr
        )
        final_segments = self._aggregate_chars_to_segments(
            refined_chars, target_segments
        )
        return self._attach_audio_to_segments(
            final_segments, waveform, original_sr, attach_audio=attach_audio
        )
