"""Tests for async coalition inference scheduling."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nlp_shap.domain.conversation import ConversationSnapshot
from nlp_shap.masking.codec import MaskCodec
from nlp_shap.pipeline.config import GenerationConfig
from nlp_shap.runtime.archive import RunArchive
from nlp_shap.runtime.dedup import CoalitionDedupRegistry, build_coalition_key
from nlp_shap.runtime.scheduler import CoalitionJob, InferenceScheduler
from nlp_shap.runtime.store import HotResultStore


async def _slow_generate(snapshot: ConversationSnapshot) -> str:
    await asyncio.sleep(0.01)
    return f"generated:{snapshot.snapshot_id}"


def _make_job(
    snapshot: ConversationSnapshot,
    coalition_key: str,
    mask_present: tuple[bool, ...] = (True, False, True),
) -> CoalitionJob:
    generation = GenerationConfig(temperature=0.0)
    key = coalition_key or build_coalition_key(
        snapshot_id=snapshot.snapshot_id,
        player_ids=("p0", "p1", "p2"),
        mask_present=mask_present,
        absence_policy="delete",
        model_id="mock",
        generation=generation,
    )
    packed = MaskCodec.pack(mask_present)
    return CoalitionJob(
        coalition_key=key,
        snapshot_id=snapshot.snapshot_id,
        snapshot=snapshot,
        absence_policy="delete",
        mask_words=packed.words,
        mask_n_bits=packed.n_bits,
        model_id="mock",
        utility=1.0,
    )


def test_scheduler_bounds_inflight(sample_snapshot: ConversationSnapshot) -> None:
    """Concurrent backend calls never exceed max_inflight."""
    max_inflight = 2
    concurrent = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def tracked_generate(snapshot: ConversationSnapshot) -> str:
        nonlocal concurrent, max_seen
        async with lock:
            concurrent += 1
            max_seen = max(max_seen, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        return f"text:{snapshot.snapshot_id}"

    async def run() -> int:
        scheduler = InferenceScheduler(
            max_inflight=max_inflight,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=None,
        )
        jobs = [
            _make_job(sample_snapshot, coalition_key=f"unique-{index}")
            for index in range(100)
        ]
        metrics = await scheduler.run(jobs, tracked_generate)
        return max_seen, metrics.executed

    max_seen, executed = asyncio.run(run())
    assert max_seen <= max_inflight
    assert executed == 100


def test_scheduler_reuses_store_for_repeated_masks(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """One hundred jobs with ten unique coalition keys execute ten backend calls."""
    executed = 0

    async def counting_generate(snapshot: ConversationSnapshot) -> str:
        nonlocal executed
        executed += 1
        return "ok"

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=4,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=CoalitionDedupRegistry(),
        )
        jobs = [
            _make_job(sample_snapshot, coalition_key=f"key-{index % 10}")
            for index in range(100)
        ]
        metrics = await scheduler.run(jobs, counting_generate)
        assert metrics.requested == 100
        assert metrics.executed == 10
        assert metrics.cache_hits == 90
        assert metrics.deduplicated == 0

    asyncio.run(run())
    assert executed == 10


def test_scheduler_counts_deduplicated_inflight_duplicates(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Concurrent duplicate keys deduplicate before the hot store is populated."""
    started = asyncio.Event()
    release = asyncio.Event()
    executed = 0

    async def gated_generate(snapshot: ConversationSnapshot) -> str:
        nonlocal executed
        executed += 1
        started.set()
        await release.wait()
        return "ok"

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=4,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=CoalitionDedupRegistry(),
        )
        key = "shared-key"
        jobs = [_make_job(sample_snapshot, coalition_key=key) for _ in range(3)]
        task = asyncio.create_task(scheduler.run(jobs, gated_generate))
        await started.wait()
        release.set()
        metrics = await task
        assert metrics.executed == 1
        assert metrics.deduplicated == 2

    asyncio.run(run())


def test_scheduler_uses_hot_store_before_backend(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Cached coalition results avoid backend execution."""
    store = HotResultStore()
    key = build_coalition_key(
        snapshot_id=sample_snapshot.snapshot_id,
        player_ids=("p0",),
        mask_present=(True,),
        absence_policy="delete",
        model_id="mock",
        generation=GenerationConfig(temperature=0.0),
    )
    store.put(key, "cached")

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=2,
            generation=GenerationConfig(temperature=0.0),
            store=store,
            dedup=CoalitionDedupRegistry(),
        )
        jobs = [_make_job(sample_snapshot, coalition_key=key)]
        metrics = await scheduler.run(jobs, _slow_generate)
        assert metrics.cache_hits == 1
        assert metrics.executed == 0

    asyncio.run(run())


def test_scheduler_appends_archive_records(
    tmp_path: Path,
    sample_snapshot: ConversationSnapshot,
    run_manifest,
) -> None:
    """Executed jobs append coalition rows to the run archive."""
    root = tmp_path / "run-scheduler"

    async def run() -> None:
        with RunArchive.open(root, run_manifest) as archive:
            scheduler = InferenceScheduler(
                max_inflight=2,
                generation=GenerationConfig(temperature=0.0),
                store=HotResultStore(),
                dedup=None,
                archive=archive,
            )
            jobs = [_make_job(sample_snapshot, coalition_key="archive-key")]
            await scheduler.run(jobs, _slow_generate)

    asyncio.run(run())

    with RunArchive.open(root, run_manifest) as archive:
        records = list(archive.history_lazy())
    assert len(records) == 1
    assert records[0].generation_text.startswith("generated:")
