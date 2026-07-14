"""Helpers for building generation records from mock outputs."""

from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...masking.tokens import tokenize_snapshot


def generation_record_from_snapshot(
    text: str,
    snapshot: ConversationSnapshot,
) -> GenerationRecord:
    """Build a generation record with token rows derived from ``snapshot``."""
    spans = tokenize_snapshot(snapshot)
    rows = tuple((_stable_token_id(span.text),) for span in spans)
    return GenerationRecord(text=text, text_token_rows=rows)


def _stable_token_id(token: str) -> int:
    digest = token.encode("utf-8")
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % 10_000
