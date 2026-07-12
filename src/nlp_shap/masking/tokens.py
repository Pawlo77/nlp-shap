"""Whitespace token layout shared by partition and absence policies."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.conversation import ConversationSnapshot, Message, Turn


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """One explainable token and its location inside a snapshot."""

    turn_index: int
    """Zero-based turn index containing the token."""

    message_index: int
    """Zero-based message index inside the turn."""

    token_index: int
    """Zero-based token index inside the message text."""

    text: str
    """Whitespace-delimited token text."""


def tokenize_snapshot(snapshot: ConversationSnapshot) -> tuple[TokenSpan, ...]:
    """Split snapshot messages into ordered whitespace tokens."""
    spans: list[TokenSpan] = []
    for turn_index, turn in enumerate(snapshot.turns):
        for message_index, message in enumerate(turn.messages):
            for token_index, token in enumerate(_split_tokens(message.text)):
                spans.append(
                    TokenSpan(
                        turn_index=turn_index,
                        message_index=message_index,
                        token_index=token_index,
                        text=token,
                    )
                )
    return tuple(spans)


def player_id_for_span(snapshot_id: str, span: TokenSpan) -> str:
    """Build a stable player identifier for one token span."""
    return f"{snapshot_id}:{span.turn_index}:{span.message_index}:{span.token_index}"


def rebuild_snapshot(
    base: ConversationSnapshot,
    spans: tuple[TokenSpan, ...],
    rendered_tokens: tuple[str, ...],
) -> ConversationSnapshot:
    """Rebuild a snapshot from rendered token strings."""
    if len(spans) != len(rendered_tokens):
        msg = "rendered token count must match span count"
        raise ValueError(msg)

    token_cursor = 0
    rebuilt_turns: list[Turn] = []
    for turn_index, turn in enumerate(base.turns):
        rebuilt_messages: list[Message] = []
        for message_index, message in enumerate(turn.messages):
            message_tokens: list[str] = []
            while token_cursor < len(spans):
                span = spans[token_cursor]
                if span.turn_index != turn_index or span.message_index != message_index:
                    break
                message_tokens.append(rendered_tokens[token_cursor])
                token_cursor += 1
            rebuilt_messages.append(
                Message(role=message.role, text=" ".join(message_tokens))
            )
        rebuilt_turns.append(Turn(messages=tuple(rebuilt_messages)))

    return ConversationSnapshot.from_turns(tuple(rebuilt_turns))


def _split_tokens(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    return tuple(stripped.split())
