"""Contract tests for the OpenAI-compatible API backend."""

import asyncio
import json

import httpx
import pytest

from nlp_shap.backends.api import ApiBackend
from nlp_shap.backends.api import payload as api_payload
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.errors import BackendUnavailableError
from nlp_shap.pipeline.config import BackendConfig


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def _success_response(text: str) -> httpx.Response:
    body = {
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }
    return httpx.Response(200, json=body)


def test_api_backend_generate_returns_assistant_text() -> None:
    """ApiBackend parses chat-completions JSON into a generation record."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "remote-model"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["temperature"] == 0.0
        return _success_response("world")

    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
        transport=httpx.MockTransport(handler),
    )

    async def run() -> str:
        try:
            record = await backend.generate(_snapshot("hello"), 8, 0.0, 1)
            return record.text
        finally:
            await backend.aclose()

    assert asyncio.run(run()) == "world"
    assert calls == 1


def test_api_backend_dedups_identical_requests_at_zero_temperature() -> None:
    """Identical deterministic requests reuse the in-memory response cache."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        del request
        return _success_response("cached")

    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
        transport=httpx.MockTransport(handler),
    )
    snapshot = _snapshot("repeat me")

    async def run() -> tuple[str, str]:
        try:
            first = await backend.generate(snapshot, 4, 0.0, 1)
            second = await backend.generate(snapshot, 4, 0.0, 1)
            return first.text, second.text
        finally:
            await backend.aclose()

    first_text, second_text = asyncio.run(run())
    assert first_text == second_text == "cached"
    assert calls == 1


def test_api_backend_skips_dedup_when_temperature_is_positive() -> None:
    """Sampling requests always hit the remote endpoint."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["temperature"] == 0.7
        return _success_response(f"sample-{calls}")

    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
        transport=httpx.MockTransport(handler),
    )
    snapshot = _snapshot("sample me")

    async def run() -> tuple[str, str]:
        try:
            first = await backend.generate(snapshot, 4, 0.7, 5)
            second = await backend.generate(snapshot, 4, 0.7, 5)
            return first.text, second.text
        finally:
            await backend.aclose()

    first_text, second_text = asyncio.run(run())
    assert first_text == "sample-1"
    assert second_text == "sample-2"
    assert calls == 2


def test_api_backend_raises_on_invalid_response_payload() -> None:
    """Malformed chat-completions payloads raise BackendUnavailableError."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": []})

    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
        transport=httpx.MockTransport(handler),
    )

    async def run() -> None:
        try:
            await backend.generate(_snapshot("hello"), 4, 0.0, 1)
        finally:
            await backend.aclose()

    with pytest.raises(BackendUnavailableError, match="unexpected chat/completions"):
        asyncio.run(run())


def test_api_backend_raises_on_http_error() -> None:
    """HTTP transport failures surface as BackendUnavailableError."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, json={"error": "unavailable"})

    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
        transport=httpx.MockTransport(handler),
    )

    async def run() -> None:
        try:
            await backend.generate(_snapshot("hello"), 4, 0.0, 1)
        finally:
            await backend.aclose()

    with pytest.raises(
        BackendUnavailableError, match="HTTP API generation request failed"
    ):
        asyncio.run(run())


def test_api_backend_rejects_audio_snapshots() -> None:
    """Audio-marked snapshots are rejected before any HTTP call."""
    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
        transport=httpx.MockTransport(lambda request: _success_response("noop")),
    )
    snapshot = _snapshot("audio:clip.wav")

    async def run() -> None:
        try:
            await backend.generate(snapshot, 4, 0.0, 1)
        finally:
            await backend.aclose()

    with pytest.raises(ValueError, match="audio snapshots"):
        asyncio.run(run())


def test_payload_cache_key_is_stable_for_equivalent_payloads() -> None:
    """Canonical JSON hashing collapses key-order differences."""
    left = {
        "model": "m",
        "messages": [{"role": "user", "content": "a"}],
        "temperature": 0.0,
    }
    right = {
        "temperature": 0.0,
        "messages": [{"content": "a", "role": "user"}],
        "model": "m",
    }
    assert api_payload.payload_cache_key(left) == api_payload.payload_cache_key(right)


def test_api_backend_raises_when_httpx_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing httpx dependency surfaces BackendUnavailableError."""

    def fail_import() -> None:
        msg = "httpx package is required for the API backend extra"
        raise BackendUnavailableError(msg)

    monkeypatch.setattr("nlp_shap.backends.api.backend._import_httpx", fail_import)
    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
    )

    async def run() -> None:
        try:
            await backend.generate(_snapshot("hello"), 4, 0.0, 1)
        finally:
            await backend.aclose()

    with pytest.raises(BackendUnavailableError, match="httpx package is required"):
        asyncio.run(run())
