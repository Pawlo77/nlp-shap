"""Immutable conversation snapshots without backend or IO."""

import hashlib
import json
from dataclasses import dataclass
from typing import Self

from .enums import ModalityFlag, Role


@dataclass(frozen=True, slots=True)
class AudioPayload:
    """Encoded audio bytes attached to a multimodal message."""

    data: bytes
    """Raw encoded audio (for example WAV or PCM)."""

    sample_rate_hz: int
    """Sample rate in hertz."""

    audio_format: str
    """Container or codec label such as ``wav`` or ``pcm_s16le``."""

    def __post_init__(self) -> None:
        if not self.data:
            msg = "audio payload data must be non-empty"
            raise ValueError(msg)
        if self.sample_rate_hz <= 0:
            msg = "sample_rate_hz must be positive"
            raise ValueError(msg)
        if not self.audio_format:
            msg = "audio_format must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Message:
    """A single explainable unit with role, modality, and optional audio."""

    role: Role
    """Participant role for this text unit."""

    text: str
    """Token or span text; may be empty for audio-only messages."""

    modality: ModalityFlag = ModalityFlag.TEXT
    """Modality of this message content."""

    audio: AudioPayload | None = None
    """Optional encoded audio payload when ``modality`` is AUDIO."""

    def __post_init__(self) -> None:
        match self.modality:
            case ModalityFlag.TEXT:
                if self.audio is not None:
                    msg = "text messages must not carry audio payloads"
                    raise ValueError(msg)
            case ModalityFlag.AUDIO:
                if self.audio is None:
                    msg = "audio messages require an audio payload"
                    raise ValueError(msg)
            case ModalityFlag.IGNORE:
                pass
            case _ as unsupported:
                msg = f"unsupported modality: {unsupported!r}"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversational turn composed of ordered messages."""

    messages: tuple[Message, ...]
    """Ordered messages that make up this turn."""

    def __post_init__(self) -> None:
        if not self.messages:
            msg = "turn must contain at least one message"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    """Frozen conversation state used as the explainability input."""

    turns: tuple[Turn, ...]
    """Ordered turns that define the conversation under study."""

    snapshot_id: str
    """Stable identifier used for deduplication and run archives."""

    def __post_init__(self) -> None:
        if not self.turns:
            msg = "snapshot must contain at least one turn"
            raise ValueError(msg)
        if not self.snapshot_id:
            msg = "snapshot_id must be non-empty"
            raise ValueError(msg)

    @classmethod
    def from_turns(cls, turns: tuple[Turn, ...]) -> Self:
        """Build a snapshot with a stable content-derived identifier."""
        return cls(turns=turns, snapshot_id=_digest_turns(turns))

    def has_audio(self) -> bool:
        """Return whether any message carries an audio payload."""
        return any(
            message.modality is ModalityFlag.AUDIO or message.audio is not None
            for turn in self.turns
            for message in turn.messages
        )


def _digest_turns(turns: tuple[Turn, ...]) -> str:
    payload = [
        [_message_digest(message) for message in turn.messages] for turn in turns
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _message_digest(message: Message) -> dict[str, str | int | None]:
    if message.modality is ModalityFlag.TEXT and message.audio is None:
        return {
            "role": message.role.value,
            "text": message.text,
        }
    audio = message.audio
    audio_digest: str | None = None
    sample_rate: int | None = None
    audio_format: str | None = None
    if audio is not None:
        audio_digest = hashlib.sha256(audio.data).hexdigest()[:16]
        sample_rate = audio.sample_rate_hz
        audio_format = audio.audio_format
    return {
        "role": message.role.value,
        "text": message.text,
        "modality": message.modality.value,
        "audio_sha256": audio_digest,
        "sample_rate_hz": sample_rate,
        "audio_format": audio_format,
    }
