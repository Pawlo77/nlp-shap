"""Chat-completions request and response helpers."""

import hashlib
import json
from typing import Any

from ...errors import BackendUnavailableError


def build_chat_payload(
    model_id: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    """Build an OpenAI-style chat-completions JSON body."""
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_new_tokens,
    }
    if temperature > 0.0:
        payload["temperature"] = float(temperature)
        if top_k > 0:
            payload["top_p"] = 0.95
    else:
        payload["temperature"] = 0.0
    return payload


def payload_cache_key(payload: dict[str, Any]) -> str:
    """Return a stable SHA256 key for deterministic request deduplication."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_message_content(data: dict[str, Any]) -> str:
    """Parse assistant text from a chat-completions JSON response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = "unexpected chat/completions response payload"
        raise BackendUnavailableError(msg) from exc
    if not isinstance(content, str):
        msg = "chat/completions content was not a string"
        raise BackendUnavailableError(msg)
    return content.strip()
