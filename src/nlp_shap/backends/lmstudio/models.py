"""Resolve and download LM Studio models through the ``lmstudio`` SDK."""

import logging
import re
from collections.abc import Sequence
from typing import Any

from ...errors import BackendUnavailableError
from ...pipeline.config import BackendConfig

logger = logging.getLogger(__name__)

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def normalize_model_token(value: str) -> str:
    """Normalize a model identifier for fuzzy comparison."""
    return _TOKEN_SPLIT_RE.sub("", value.casefold())


def match_downloaded_model_key(
    downloaded: Sequence[Any],
    model_id: str,
) -> str | None:
    """Return a downloaded ``model_key`` matching ``model_id``, if any."""
    target = normalize_model_token(model_id)
    if not target:
        return None
    for entry in downloaded:
        key = str(getattr(entry, "model_key", ""))
        candidate = normalize_model_token(key)
        if not candidate:
            continue
        if candidate == target or target in candidate or candidate in target:
            return key
    return None


def pick_search_result(results: Sequence[Any], model_id: str) -> Any:
    """Choose the best repository search hit for ``model_id``."""
    if not results:
        msg = f"LM Studio repository returned no results for {model_id!r}"
        raise BackendUnavailableError(msg)
    target = normalize_model_token(model_id)

    def score(entry: Any) -> tuple[int, int]:
        data = entry.search_result
        name = normalize_model_token(str(getattr(data, "name", "")))
        exact = 1 if bool(getattr(data, "exact", False)) else 0
        name_match = 2 if name == target else 1 if target and target in name else 0
        return (exact, name_match)

    return max(results, key=score)


def pick_download_option(options: Sequence[Any], quantization: str) -> Any:
    """Choose a repository download option for the requested quantization."""
    if not options:
        msg = "LM Studio model has no download options"
        raise BackendUnavailableError(msg)
    target = quantization.casefold()
    for option in options:
        value = getattr(option.info, "quantization", None)
        if isinstance(value, str) and value.casefold() == target:
            return option
    for option in options:
        if bool(getattr(option.info, "recommended", False)):
            return option
    for option in options:
        name = str(getattr(option.info, "name", "")).casefold()
        if target in name:
            return option
    return options[0]


async def resolve_model_key(client: Any, config: BackendConfig) -> str:
    """Return a loadable LM Studio model key, downloading when configured."""
    downloaded = await client.llm.list_downloaded()
    matched = match_downloaded_model_key(downloaded, config.model_id)
    if matched is not None:
        logger.info(
            "Using downloaded LM Studio model %r for configured id %r",
            matched,
            config.model_id,
        )
        return matched

    try:
        await client.llm.get_model_info(config.model_id)
    except Exception:
        if not config.auto_download:
            msg = (
                f"LM Studio model {config.model_id!r} is not downloaded "
                "and auto_download is disabled"
            )
            raise BackendUnavailableError(msg) from None
        return await _download_model(client, config)

    return config.model_id


async def _download_model(client: Any, config: BackendConfig) -> str:
    search_term = config.hub_search or config.model_id
    logger.info(
        "Downloading LM Studio model for %r (search=%r, quant=%s)",
        config.model_id,
        search_term,
        config.quantization,
    )
    results = await client.repository.search_models(
        search_term=search_term,
        limit=10,
        compatibility_types=["gguf"],
    )
    chosen = pick_search_result(results, config.model_id)
    options = await chosen.get_download_options()
    option = pick_download_option(options, config.quantization)
    quant = getattr(option.info, "quantization", None) or config.quantization
    model_name = getattr(chosen.search_result, "name", search_term)
    logger.info("Downloading %s (%s) from LM Studio repository", model_name, quant)

    def on_progress(update: Any) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        downloaded_bytes = int(getattr(update, "downloaded_bytes", 0))
        total_bytes = int(getattr(update, "total_bytes", 0))
        logger.info(
            "LM Studio download %s: %d/%d bytes",
            model_name,
            downloaded_bytes,
            total_bytes,
        )

    try:
        model_key = await option.download(on_progress=on_progress)
    except Exception as exc:
        msg = f"failed to download LM Studio model for {config.model_id!r}"
        raise BackendUnavailableError(msg) from exc
    logger.info("Downloaded LM Studio model %r", model_key)
    return str(model_key)
