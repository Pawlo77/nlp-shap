"""LM Studio generative backend."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ...backends.mock.generation import generation_record_from_snapshot
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...errors import BackendUnavailableError
from ...pipeline.config import BackendConfig
from .chat import snapshot_to_chat
from .models import resolve_model_key


class LmStudioBackend:
    """Generate text through a local LM Studio API server."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config
        self._client: Any | None = None
        self._model: Any | None = None
        self._init_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        """Return the configured LM Studio model identifier."""
        return self._config.model_id

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> GenerationRecord:
        """Generate assistant text for ``snapshot`` via LM Studio."""
        lms = _import_lmstudio()
        chat = snapshot_to_chat(snapshot, lms.Chat)
        prediction_config: dict[str, float | int] = {
            "temperature": temperature,
            "maxTokens": max_new_tokens,
        }
        if top_k > 0:
            prediction_config["topKSampling"] = top_k
        async with self._generation_slot():
            model = await self._ensure_model()
            try:
                result = await model.respond(chat, config=prediction_config)
            except Exception:
                # Model may have crashed/unloaded; drop handle and retry once.
                self._model = None
                try:
                    model = await self._ensure_model()
                    result = await model.respond(chat, config=prediction_config)
                except Exception as retry_exc:
                    msg = "LM Studio generation request failed"
                    raise BackendUnavailableError(msg) from retry_exc
            text = _extract_text(result)
            return generation_record_from_snapshot(text, snapshot)

    async def aclose(self) -> None:
        """Close the underlying LM Studio client connection."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
        self._client = None
        self._model = None

    @asynccontextmanager
    async def _generation_slot(self) -> AsyncIterator[None]:
        if self._config.serialize_generate:
            async with self._generate_lock:
                yield
        else:
            yield

    async def _ensure_model(self) -> Any:
        async with self._init_lock:
            if self._model is not None:
                return self._model
            client = await self._connect_client()
            try:
                model_key = await resolve_model_key(client, self._config)
                # ttl=None keeps the instance loaded (UI Idle TTL still applies).
                self._model = await client.llm.model(model_key, ttl=None)
            except BackendUnavailableError:
                await self.aclose()
                raise
            except Exception as exc:
                await self.aclose()
                msg = f"failed to load LM Studio model {self._config.model_id!r}"
                raise BackendUnavailableError(msg) from exc
            return self._model

    async def _connect_client(self) -> Any:
        if self._client is not None:
            return self._client
        lms = _import_lmstudio()
        host = self._config.api_host
        if not host:
            host = await lms.AsyncClient.find_default_local_api_host()
        if host is None or not await lms.AsyncClient.is_valid_api_host(host):
            msg = "LM Studio API server is not reachable on the local machine"
            raise BackendUnavailableError(msg)
        client = lms.AsyncClient(host)
        try:
            await client.__aenter__()
        except Exception as exc:
            msg = f"failed to connect to LM Studio API host {host!r}"
            raise BackendUnavailableError(msg) from exc
        self._client = client
        return client


def _extract_text(result: object) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content.strip()
    msg = "LM Studio prediction result did not include text content"
    raise BackendUnavailableError(msg)


def _import_lmstudio() -> Any:
    try:
        import lmstudio as lms
    except ImportError as exc:
        msg = "lmstudio package is required for the LM Studio backend extra"
        raise BackendUnavailableError(msg) from exc
    _silence_lmstudio_sdk_logs()
    return lms


def _silence_lmstudio_sdk_logs() -> None:
    """Drop LM Studio SDK websocket chatter (class-named loggers at INFO)."""
    # SDK uses type(self).__name__ as logger names (not under ``lmstudio.*``).
    for name in (
        "_AsyncLMStudioWebsocket",
        "AsyncWebsocketHandler",
        "AsyncWebsocketThread",
        "_SyncLMStudioWebsocket",
        "SyncWebsocketHandler",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
