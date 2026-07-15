"""Performance benchmarks for the API backend."""

import time

import pytest

from nlp_shap.backends.api.payload import build_chat_payload, payload_cache_key


@pytest.mark.bench
def test_payload_cache_key_hashing() -> None:
    """Hashing chat payloads stays cheap enough for per-request dedup keys."""
    payload = build_chat_payload(
        "remote-model",
        [{"role": "user", "content": "token attribution smoke"}],
        32,
        0.0,
        1,
    )
    start = time.perf_counter()
    keys = [payload_cache_key(payload) for _ in range(10_000)]
    elapsed = time.perf_counter() - start
    assert len(keys[0]) == 64
    assert elapsed < 1.0
