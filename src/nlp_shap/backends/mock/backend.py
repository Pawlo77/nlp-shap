"""Deterministic mock backend for CI-safe E2E tests."""

import hashlib
import struct

from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from .generation import generation_record_from_snapshot


class MockBackend:
    """Hash-based generative backend with no external model dependencies."""

    def __init__(self, model_id: str = "mock") -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """Return the backend model identifier."""
        return self._model_id

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> GenerationRecord:
        """Return deterministic generation output for ``snapshot``."""
        text = _deterministic_text(
            snapshot,
            self._model_id,
            max_new_tokens,
            temperature,
            top_k,
        )
        return generation_record_from_snapshot(text, snapshot)


def _deterministic_text(
    snapshot: ConversationSnapshot,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(snapshot.snapshot_id.encode("utf-8"))
    for turn in snapshot.turns:
        for message in turn.messages:
            hasher.update(message.text.encode("utf-8"))
    hasher.update(model_id.encode("utf-8"))
    hasher.update(
        struct.pack("<ifi", max_new_tokens, temperature, top_k),
    )
    digest = hasher.hexdigest()
    token_count = max(1, min(max_new_tokens, 16))
    return " ".join(digest[index : index + 4] for index in range(0, token_count * 4, 4))
