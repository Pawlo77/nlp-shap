"""Absence policies for rendering masked conversation snapshots."""

from dataclasses import dataclass

from ..alignment.io import TorchAudioHandler
from ..alignment.segments import AudioSegment
from ..domain.coalition import CoalitionMask
from ..domain.conversation import (
    AudioPayload,
    ConversationSnapshot,
    Message,
    Turn,
)
from ..domain.enums import ModalityFlag
from ..domain.players import PlayerSet
from .tokens import rebuild_snapshot, tokenize_snapshot


@dataclass(frozen=True, slots=True)
class DeletePolicy:
    """Remove absent tokens from the rendered conversation."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "delete"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Drop absent tokens while preserving turn and message structure."""
        mask.validate_against(players)
        spans = tokenize_snapshot(snapshot)
        if len(spans) != players.num_players:
            msg = "token layout does not match the supplied player set"
            raise ValueError(msg)
        rendered = tuple(
            span.text
            for span, present in zip(spans, mask.present, strict=True)
            if present
        )
        kept_spans = tuple(
            span for span, present in zip(spans, mask.present, strict=True) if present
        )
        # Empty coalition is valid for Shapley (v(∅)): keep turn/message
        # structure with empty message text.
        return rebuild_snapshot(snapshot, kept_spans, rendered)


@dataclass(frozen=True, slots=True)
class PadPolicy:
    """Replace absent tokens with a fixed mask placeholder."""

    placeholder: str = "[MASK]"
    """Placeholder text inserted for absent tokens."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "pad"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Substitute absent tokens with :attr:`placeholder`."""
        mask.validate_against(players)
        spans = tokenize_snapshot(snapshot)
        if len(spans) != players.num_players:
            msg = "token layout does not match the supplied player set"
            raise ValueError(msg)
        rendered = tuple(
            span.text if present else self.placeholder
            for span, present in zip(spans, mask.present, strict=True)
        )
        return rebuild_snapshot(snapshot, spans, rendered)


@dataclass(frozen=True, slots=True)
class NeutralPolicy:
    """Replace absent tokens with neutral fillers that preserve token width."""

    fill_char: str = "_"
    """Single character repeated to match each absent token length."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "neutral"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Substitute absent tokens with width-matched neutral fillers."""
        mask.validate_against(players)
        if len(self.fill_char) != 1:
            msg = "fill_char must be exactly one character"
            raise ValueError(msg)
        spans = tokenize_snapshot(snapshot)
        if len(spans) != players.num_players:
            msg = "token layout does not match the supplied player set"
            raise ValueError(msg)
        rendered = tuple(
            span.text if present else self.fill_char * len(span.text)
            for span, present in zip(spans, mask.present, strict=True)
        )
        return rebuild_snapshot(snapshot, spans, rendered)


@dataclass(frozen=True, slots=True)
class SegmentDeletePolicy:
    """Drop absent SGPA segments and rebuild multimodal messages."""

    segments: tuple[AudioSegment, ...]
    """Filtered SGPA segments aligned with the coalition player set."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "sgpa_delete"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Keep present segments; recombine audio for multimodal messages."""
        mask.validate_against(players)
        if len(self.segments) != players.num_players:
            msg = "segment layout does not match the supplied player set"
            raise ValueError(msg)
        kept = tuple(
            segment
            for segment, present in zip(self.segments, mask.present, strict=True)
            if present
        )
        return _rebuild_snapshot_from_segments(snapshot, kept)


def _rebuild_snapshot_from_segments(
    base: ConversationSnapshot,
    kept: tuple[AudioSegment, ...],
) -> ConversationSnapshot:
    audio_message_seen = False
    rebuilt_turns: list[Turn] = []
    for turn in base.turns:
        rebuilt_messages: list[Message] = []
        for message in turn.messages:
            if message.modality is ModalityFlag.AUDIO and not audio_message_seen:
                audio_message_seen = True
                rebuilt_messages.append(_render_audio_message(message, kept))
            else:
                rebuilt_messages.append(message)
        rebuilt_turns.append(Turn(messages=tuple(rebuilt_messages)))
    return ConversationSnapshot.from_turns(tuple(rebuilt_turns))


def _render_audio_message(
    message: Message,
    kept: tuple[AudioSegment, ...],
) -> Message:
    if not kept:
        return Message(role=message.role, text="")
    text = " ".join(segment.token for segment in kept)
    audio_bytes = _combine_segment_audio(kept)
    if not audio_bytes:
        return Message(role=message.role, text=text)
    sample_rate = next(
        (segment.sample_rate for segment in kept if segment.sample_rate is not None),
        message.audio.sample_rate_hz if message.audio is not None else 16_000,
    )
    audio_format = kept[0].audio_format or (
        message.audio.audio_format if message.audio is not None else "wav"
    )
    return Message(
        role=message.role,
        text=text,
        modality=ModalityFlag.AUDIO,
        audio=AudioPayload(
            data=audio_bytes,
            sample_rate_hz=int(sample_rate),
            audio_format=audio_format,
        ),
    )


def _combine_segment_audio(segments: tuple[AudioSegment, ...]) -> bytes:
    payloads = tuple(segment.audio for segment in segments if segment.audio)
    if not payloads:
        return b""
    if len(payloads) == 1:
        return payloads[0]
    return TorchAudioHandler.combine(list(segments), target_audio_format="wav")
