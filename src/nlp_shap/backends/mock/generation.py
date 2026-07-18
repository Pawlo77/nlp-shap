"""Helpers for building generation records from mock outputs."""

import hashlib
import struct

from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...masking.tokens import tokenize_snapshot

_EMBED_DIM = 8


def generation_record_from_snapshot(
    text: str,
    snapshot: ConversationSnapshot,
) -> GenerationRecord:
    """Build a generation record with token rows derived from ``snapshot``.

    Populates deterministic ``embedding`` / ``contextual_embedding`` vectors so
    embedding-based value functions work under the mock backend (CI / smoke).
    """
    spans = tokenize_snapshot(snapshot)
    rows = tuple((_stable_token_id(span.text),) for span in spans)
    static = _hash_embedding(text)
    # Slightly different vector for contextual mode so U1 vs U2 paths diverge.
    contextual = _hash_embedding(f"{text}\0{snapshot.snapshot_id}")
    return GenerationRecord(
        text=text,
        text_token_rows=rows,
        embedding=static,
        contextual_embedding=contextual,
    )


def _stable_token_id(token: str) -> int:
    digest = token.encode("utf-8")
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % 10_000


def _hash_embedding(payload: str) -> tuple[float, ...]:
    """Map ``payload`` to a fixed-size float vector via SHA-256."""
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(_EMBED_DIM):
        (raw,) = struct.unpack_from("<I", digest, index * 4)
        # Map to [-1, 1]
        values.append((raw / 0xFFFFFFFF) * 2.0 - 1.0)
    return tuple(values)
