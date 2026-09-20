"""Tests for multimodal masking over SGPA audio segments."""

import io
import struct
import wave

from nlp_shap.alignment.partitions import SgpaSegmentPartitioner
from nlp_shap.alignment.segments import AudioSegment
from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.conversation import (
    AudioPayload,
    ConversationSnapshot,
    Message,
    Turn,
)
from nlp_shap.domain.enums import ModalityFlag, Role
from nlp_shap.masking.builder import MaskBuilder
from nlp_shap.masking.filters import ExcludePunctuationTokensFilter
from nlp_shap.masking.policies import SegmentDeletePolicy


def _pcm_wav(num_samples: int, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return buffer.getvalue()


def _segment(
    token: str,
    start: float,
    end: float,
    audio: bytes,
) -> AudioSegment:
    return AudioSegment(
        token=token,
        start_time=start,
        end_time=end,
        confidence=0.9,
        audio=audio,
        audio_format="wav",
        sample_rate=16_000,
        start_sample=0,
        end_sample=max(1, int((end - start) * 16_000)),
    )


def test_masking_sgpa_segments_yields_multimodal_masked_snapshot() -> None:
    """Coalition views over SGPA players share the multimodal base snapshot."""
    hello = _pcm_wav(1_600)
    world = _pcm_wav(1_600)
    payload = AudioPayload(
        data=hello + world, sample_rate_hz=16_000, audio_format="wav"
    )
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
        _segment("hello", 0.0, 0.1, hello),
        _segment("world", 0.1, 0.2, world),
    )
    partitioner = SgpaSegmentPartitioner(
        segments=segments,
        token_filter=ExcludePunctuationTokensFilter(),
    )
    players = partitioner.partition(snapshot)
    mask = CoalitionMask.from_sequence((True, False))
    policy = SegmentDeletePolicy(segments=partitioner.filtered_segments())
    builder = MaskBuilder(policy)
    view = builder.view(snapshot, players, mask)

    assert view.base is snapshot
    assert view.base.has_audio() is True
    assert view.players.num_players == 2
    assert view.mask.present == (True, False)
    assert view.policy_name == "sgpa_delete"


def test_segment_delete_policy_renders_kept_audio_message() -> None:
    """Absent SGPA segments are dropped; kept audio is recombined."""
    hello = _pcm_wav(1_600)
    world = _pcm_wav(2_400)
    payload = AudioPayload(
        data=hello + world, sample_rate_hz=16_000, audio_format="wav"
    )
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
        _segment("hello", 0.0, 0.1, hello),
        _segment("world", 0.1, 0.25, world),
    )
    partitioner = SgpaSegmentPartitioner(segments=segments)
    players = partitioner.partition(snapshot)
    mask = CoalitionMask.from_sequence((True, False))
    policy = SegmentDeletePolicy(segments=segments)
    rendered = MaskBuilder(policy).render(
        MaskBuilder(policy).view(snapshot, players, mask)
    )

    message = rendered.turns[0].messages[0]
    assert message.text == "hello"
    assert message.modality is ModalityFlag.AUDIO
    assert message.audio is not None
    assert message.audio.data == hello
    assert message.audio.sample_rate_hz == 16_000
