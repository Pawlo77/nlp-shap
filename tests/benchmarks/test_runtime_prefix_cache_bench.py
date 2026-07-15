"""Micro-benchmarks for prefix cache hit rate."""

import time

import pytest

from nlp_shap.runtime.kv_cache import PrefixCacheManager

from ._regression import BaselineStore


@pytest.mark.bench
def test_prefix_cache_hits_10k(baseline_store: BaselineStore) -> None:
    """Prefix cache lookups on shared prefixes stay within budget."""
    cache = PrefixCacheManager()
    prefixes = [tuple(range(length)) for length in range(4, 20)]
    for prefix in prefixes:
        cache.store(prefix, object())

    start = time.perf_counter()
    hits = 0
    for index in range(10_000):
        token_ids = prefixes[index % len(prefixes)] + (index % 7,)
        matched, length = cache.lookup(token_ids)
        if length > 0:
            hits += 1
            assert matched is not None
    elapsed = time.perf_counter() - start
    assert hits == 10_000
    assert cache.hits == 10_000
    baseline_store.check("prefix_cache_hits_10k", elapsed, ceiling_s=0.15)
