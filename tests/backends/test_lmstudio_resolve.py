"""Async tests for LM Studio model resolution."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nlp_shap.backends.lmstudio.models import resolve_model_key
from nlp_shap.pipeline.config import BackendConfig


def test_resolve_model_key_uses_downloaded_match() -> None:
    """Configured ids resolve to an already-downloaded model key."""
    client = SimpleNamespace(
        llm=SimpleNamespace(
            list_downloaded=AsyncMock(
                return_value=(SimpleNamespace(model_key="qwen2.5-3b-instruct"),),
            ),
            get_model_info=AsyncMock(),
        ),
    )
    config = BackendConfig(kind="lmstudio", model_id="qwen2.5-3b-instruct")

    async def run() -> str:
        return await resolve_model_key(client, config)

    key = asyncio.run(run())
    assert key == "qwen2.5-3b-instruct"
    client.llm.get_model_info.assert_not_called()


def test_resolve_model_key_downloads_when_missing() -> None:
    """Missing models are downloaded from the repository when auto_download is on."""
    download_option = SimpleNamespace(
        info=SimpleNamespace(quantization="Q4_K_M", name="q4", recommended=True),
        download=AsyncMock(return_value="qwen2.5-3b-instruct"),
    )
    search_hit = SimpleNamespace(
        search_result=SimpleNamespace(name="Qwen2.5 3B Instruct", exact=True),
        get_download_options=AsyncMock(return_value=(download_option,)),
    )
    client = SimpleNamespace(
        llm=SimpleNamespace(
            list_downloaded=AsyncMock(return_value=()),
            get_model_info=AsyncMock(side_effect=RuntimeError("missing")),
        ),
        repository=SimpleNamespace(
            search_models=AsyncMock(return_value=(search_hit,)),
        ),
    )
    config = BackendConfig(kind="lmstudio", model_id="qwen2.5-3b-instruct")

    async def run() -> str:
        return await resolve_model_key(client, config)

    key = asyncio.run(run())
    assert key == "qwen2.5-3b-instruct"
    client.repository.search_models.assert_awaited_once()
    download_option.download.assert_awaited_once()
