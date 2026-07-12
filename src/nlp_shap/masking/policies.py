"""Absence policies for rendering masked conversation snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
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
        if not rendered:
            msg = "delete policy cannot remove every token"
            raise ValueError(msg)
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
