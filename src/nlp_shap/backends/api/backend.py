"""OpenAI-compatible HTTP generative backend."""

import asyncio
import os
from typing import Any

from ...backends.mock.generation import generation_record_from_snapshot
from ...backends.transformers.chat import snapshot_has_audio, snapshot_to_chat_messages
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...errors import BackendUnavailableError
from ...pipeline.config import BackendConfig
from .payload import build_chat_payload, extract_message_content, payload_cache_key


class ApiBackend:
    """Generate text through an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        config: BackendConfig,
        *,
        transport: Any | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._client: Any | None = None
        self._response_cache: dict[str, str] = {}
        self._init_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        """Return the configured remote model identifier."""
        return self._config.model_id

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> GenerationRecord:
        """Generate assistant text for ``snapshot`` via the HTTP API."""
        if snapshot_has_audio(snapshot):
            msg = "audio snapshots are not supported by the API backend"
            raise ValueError(msg)
        messages = snapshot_to_chat_messages(snapshot)
        payload = build_chat_payload(
            self._config.model_id,
            messages,
            max_new_tokens,
            temperature,
            top_k,
        )
        cache_key: str | None = None
        if temperature == 0.0:
            cache_key = payload_cache_key(payload)
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                return generation_record_from_snapshot(cached, snapshot)
        text = await self._post_chat_completions(payload)
        if cache_key is not None:
            self._response_cache[cache_key] = text
        return generation_record_from_snapshot(text, snapshot)

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def _post_chat_completions(self, payload: dict[str, Any]) -> str:
        client = await self._ensure_client()
        url = f"{self._base_url()}/chat/completions"
        headers = self._headers()
        httpx = _import_httpx()
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = "HTTP API generation request failed"
            raise BackendUnavailableError(msg) from exc
        data = response.json()
        if not isinstance(data, dict):
            msg = "HTTP API response was not a JSON object"
            raise BackendUnavailableError(msg)
        return extract_message_content(data)

    def _base_url(self) -> str:
        raw = (
            self._config.api_host
            or os.environ.get("OPENAI_BASE_URL")
            or "http://127.0.0.1:1234/v1"
        )
        return raw.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NLP_SHAP_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _ensure_client(self) -> Any:
        async with self._init_lock:
            if self._client is not None:
                return self._client
            httpx = _import_httpx()
            client_kwargs: dict[str, Any] = {"timeout": 600.0}
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**client_kwargs)
            return self._client


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        msg = "httpx package is required for the API backend extra"
        raise BackendUnavailableError(msg) from exc
    return httpx
