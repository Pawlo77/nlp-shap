"""Immutable conversation snapshots without backend or IO."""

import hashlib
import json
from dataclasses import dataclass
from typing import Self

from .enums import Role


@dataclass(frozen=True, slots=True)
class Message:
    """A single explainable text unit with an attached role."""

    role: Role
    """Participant role for this text unit."""

    text: str
    """Token or span text included in the explainability input."""


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


def _digest_turns(turns: tuple[Turn, ...]) -> str:
    payload = [
        [
            {"role": message.role.value, "text": message.text}
            for message in turn.messages
        ]
        for turn in turns
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]
