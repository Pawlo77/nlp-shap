"""Micro-benchmarks for coalition deduplication."""

import time

import pytest

from nlp_shap.pipeline.config import GenerationConfig
from nlp_shap.runtime.dedup import CoalitionDedupRegistry, build_coalition_key

from ._regression import BaselineStore

_GENERATION = GenerationConfig(max_new_tokens=32, temperature=0.0, top_k=1)
_PLAYER_IDS = tuple(f"p{index}" for index in range(16))


@pytest.mark.bench
def test_dedup_key_throughput_10k(baseline_store: BaselineStore) -> None:
    """Building and observing 10k coalition keys stays within budget."""
    registry = CoalitionDedupRegistry()
    start = time.perf_counter()
    duplicates = 0
    for index in range(10_000):
        mask = tuple(bool(((index % 100) >> bit) & 1) for bit in range(16))
        key = build_coalition_key(
            snapshot_id="bench-snapshot",
            player_ids=_PLAYER_IDS,
            mask_present=mask,
            absence_policy="pad",
            model_id="mock",
            generation=_GENERATION,
        )
        if not registry.observe(key):
            duplicates += 1
    elapsed = time.perf_counter() - start
    assert len(registry) == 100
    assert duplicates == 10_000 - 100
    baseline_store.check("dedup_keys_10k", elapsed, ceiling_s=0.5)
