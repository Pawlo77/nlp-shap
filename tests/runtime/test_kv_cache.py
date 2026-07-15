"""Tests for prefix-cache helpers."""

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.runtime.kv_cache import (
    PrefixCacheManager,
    build_snapshot_prefix_hash,
    group_jobs_for_prefix_cache,
)
from nlp_shap.runtime.scheduler import CoalitionJob


def test_build_snapshot_prefix_hash_ignores_final_token() -> None:
    """Coalitions that differ only in the last token share a prefix hash."""
    first = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="shared alpha"),)),
    ))
    second = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="shared beta"),)),
    ))
    assert build_snapshot_prefix_hash(first) == build_snapshot_prefix_hash(second)


def test_prefix_cache_manager_reports_hits() -> None:
    """Lookup returns the longest cached prefix and increments hit counters."""
    cache = PrefixCacheManager()
    cache.store((1, 2, 3), {"kv": "prefix-123"})

    matched, matched_len = cache.lookup((1, 2, 3, 4))

    assert matched == {"kv": "prefix-123"}
    assert matched_len == 3
    assert cache.hits == 1
    assert cache.misses == 0


def test_group_jobs_for_prefix_cache_sorts_by_prefix_hash(
    sample_snapshot,
) -> None:
    """Prefix grouping keeps coalitions with the same hash adjacent."""
    jobs = [
        CoalitionJob(
            coalition_key="b",
            snapshot_id=sample_snapshot.snapshot_id,
            snapshot=sample_snapshot,
            absence_policy="pad",
            mask_words=b"\x01",
            mask_n_bits=1,
            prefix_hash="beta",
            model_id="mock",
            utility=0.0,
        ),
        CoalitionJob(
            coalition_key="a",
            snapshot_id=sample_snapshot.snapshot_id,
            snapshot=sample_snapshot,
            absence_policy="pad",
            mask_words=b"\x02",
            mask_n_bits=1,
            prefix_hash="alpha",
            model_id="mock",
            utility=0.0,
        ),
    ]

    grouped = group_jobs_for_prefix_cache(jobs, key=lambda job: job.prefix_hash)

    assert [job.prefix_hash for job in grouped] == ["alpha", "beta"]
