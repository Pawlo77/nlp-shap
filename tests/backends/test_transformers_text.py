"""Tests for the Hugging Face transformers text backend."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from nlp_shap.backends.transformers import text as transformers_text
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.errors import BackendUnavailableError
from nlp_shap.pipeline.config import BackendConfig


class _TokenizerStub:
    pad_token_id = 0
    eos_token_id = 0
    eos_token = "</s>"
    pad_token = "</s>"

    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if prompt.endswith("beta"):
            return [1, 2, 3, 5]
        return [1, 2, 3, 4]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _ModelStub:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.forward_calls = 0

    def to(self, device: object) -> "_ModelStub":
        del device
        return self

    def eval(self) -> "_ModelStub":
        return self

    def parameters(self) -> Any:
        torch = pytest.importorskip("torch")
        yield torch.zeros(1)

    def get_input_embeddings(self) -> Any:
        torch = pytest.importorskip("torch")

        def embed(ids: Any) -> Any:
            return ids.to(dtype=torch.float32).unsqueeze(-1)

        return embed

    def __call__(self, input_ids: Any, **kwargs: Any) -> SimpleNamespace:
        torch = pytest.importorskip("torch")
        self.forward_calls += 1
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return SimpleNamespace(
            past_key_values={"layer": input_ids.detach().clone()},
            hidden_states=(hidden,),
        )

    def generate(self, **kwargs: Any) -> Any:
        torch = pytest.importorskip("torch")
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([[7, 8]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def _backend_with_stub() -> transformers_text.TransformersTextBackend:
    torch = pytest.importorskip("torch")
    backend = transformers_text.TransformersTextBackend(
        BackendConfig(kind="transformers", model_id="stub-model"),
    )
    backend._tokenizer = _TokenizerStub()
    backend._model = _ModelStub()
    backend._device = torch.device("cpu")
    backend.set_kv_cache_enabled(True)
    return backend


def test_transformers_backend_generate_is_deterministic_at_zero_temperature() -> None:
    """Identical snapshots produce identical generations when temperature is zero."""
    backend = _backend_with_stub()
    snapshot = _snapshot("shared alpha")

    async def run() -> tuple[str, str]:
        first = await backend.generate(snapshot, 4, 0.0, 1)
        second = await backend.generate(snapshot, 4, 0.0, 1)
        return first.text, second.text

    first_text, second_text = asyncio.run(run())
    assert first_text == second_text


def test_transformers_backend_varies_with_masked_text() -> None:
    """Different masked snapshots produce different token prompts."""
    backend = _backend_with_stub()

    async def run() -> None:
        await backend.generate(_snapshot("shared alpha"), 4, 0.0, 1)
        await backend.generate(_snapshot("shared beta"), 4, 0.0, 1)

    asyncio.run(run())
    first_ids = backend._model.generate_calls[0]["input_ids"].tolist()
    second_ids = backend._model.generate_calls[1]["input_ids"].tolist()
    assert first_ids != second_ids


def test_transformers_backend_rejects_audio_snapshots() -> None:
    """Audio-marked snapshots raise before model generation."""
    backend = _backend_with_stub()
    snapshot = _snapshot("audio:payload")

    with pytest.raises(ValueError, match="audio snapshots"):
        asyncio.run(backend.generate(snapshot, 4, 0.0, 1))


def test_transformers_backend_populates_embedding_fields() -> None:
    """Generation records include static and contextual embedding vectors."""
    backend = _backend_with_stub()

    async def run() -> GenerationRecord:
        return await backend.generate(_snapshot("shared alpha"), 4, 0.0, 1)

    record = asyncio.run(run())
    assert record.embedding
    assert record.contextual_embedding


def test_transformers_backend_reports_kv_cache_hits() -> None:
    """Shared token prefixes increment kv_cache_hits on subsequent generations."""
    backend = _backend_with_stub()

    async def run() -> int:
        await backend.generate(_snapshot("shared alpha"), 4, 0.0, 1)
        await backend.generate(_snapshot("shared beta"), 4, 0.0, 1)
        return backend.kv_cache_hits

    hits = asyncio.run(run())
    assert hits > 0


def test_transformers_backend_raises_when_optional_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing transformers dependencies surface BackendUnavailableError."""

    def fail_import() -> Any:
        msg = "blocked"
        raise ImportError(msg)

    monkeypatch.setattr(transformers_text, "_import_transformers", fail_import)
    monkeypatch.setattr(transformers_text, "_import_torch", fail_import)
    backend = transformers_text.TransformersTextBackend(
        BackendConfig(kind="transformers", model_id="stub-model"),
    )

    with pytest.raises(BackendUnavailableError, match="transformers backend requires"):
        asyncio.run(backend.generate(_snapshot("hello"), 4, 0.0, 1))
