"""Async coalition inference scheduling with deduplication."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ..domain.conversation import ConversationSnapshot
from ..masking.codec import PackedMask
from ..pipeline.config import GenerationConfig
from .archive import CoalitionRecordDraft, RunArchive
from .dedup import CoalitionDedupRegistry
from .store import HotResultStore

GenerateFn = Callable[[ConversationSnapshot], Awaitable[str]]
"""Async callable that returns generated text for one snapshot."""


@dataclass(frozen=True, slots=True)
class SchedulerMetrics:
    """Counters collected while executing coalition jobs."""

    requested: int
    """Number of coalition jobs submitted to the scheduler."""

    executed: int
    """Number of jobs that invoked the backend generate callable."""

    deduplicated: int
    """Number of jobs skipped because the coalition key was already executed."""

    cache_hits: int
    """Number of jobs served from the hot result store without backend calls."""


@dataclass(frozen=True, slots=True)
class CoalitionJob:
    """One coalition evaluation request routed through the scheduler."""

    coalition_key: str
    """Stable deduplication key for the coalition."""

    snapshot_id: str
    """Identifier of the evaluated conversation snapshot."""

    snapshot: ConversationSnapshot
    """Conversation snapshot passed to the backend for generation."""

    absence_policy: str
    """Registered absence-policy identifier used for rendering."""

    mask_words: bytes
    """Packed coalition mask bytes."""

    mask_n_bits: int
    """Original coalition mask bit length."""

    model_id: str
    """Backend model identifier used for generation."""

    utility: float
    """Utility score assigned after generation."""


class InferenceScheduler:
    """Execute coalition jobs with bounded concurrency and optional deduplication."""

    def __init__(
        self,
        *,
        max_inflight: int,
        generation: GenerationConfig,
        store: HotResultStore,
        dedup: CoalitionDedupRegistry | None = None,
        archive: RunArchive | None = None,
    ) -> None:
        if max_inflight < 1:
            msg = "max_inflight must be at least 1"
            raise ValueError(msg)
        self._max_inflight = max_inflight
        self._generation = generation
        self._store = store
        self._dedup = dedup
        self._archive = archive
        self._semaphore = asyncio.Semaphore(max_inflight)

    async def run(
        self,
        jobs: Sequence[CoalitionJob],
        generate: GenerateFn,
    ) -> SchedulerMetrics:
        """Execute coalition jobs and return scheduler counters."""
        metrics = SchedulerMetrics(
            requested=len(jobs),
            executed=0,
            deduplicated=0,
            cache_hits=0,
        )
        mutable = {"executed": 0, "deduplicated": 0, "cache_hits": 0}

        async def _run_job(job: CoalitionJob) -> None:
            cached = self._store.get(job.coalition_key)
            if cached is not None:
                mutable["cache_hits"] += 1
                self._maybe_archive(job, cached, cache_hit=True, elapsed_ms=0.0)
                return

            if self._dedup is not None and not self._dedup.observe(job.coalition_key):
                mutable["deduplicated"] += 1
                return

            async with self._semaphore:
                started = time.perf_counter()
                generation_text = await generate(job.snapshot)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            mutable["executed"] += 1
            self._store.put(job.coalition_key, generation_text)
            self._maybe_archive(
                job,
                generation_text,
                cache_hit=False,
                elapsed_ms=elapsed_ms,
            )

        await asyncio.gather(*(_run_job(job) for job in jobs))
        return SchedulerMetrics(
            requested=metrics.requested,
            executed=mutable["executed"],
            deduplicated=mutable["deduplicated"],
            cache_hits=mutable["cache_hits"],
        )

    def _maybe_archive(
        self,
        job: CoalitionJob,
        generation_text: str,
        *,
        cache_hit: bool,
        elapsed_ms: float,
    ) -> None:
        if self._archive is None:
            return
        self._archive.append(
            CoalitionRecordDraft(
                snapshot_id=job.snapshot_id,
                coalition_key=job.coalition_key,
                mask=PackedMask(words=job.mask_words, n_bits=job.mask_n_bits),
                absence_policy=job.absence_policy,
                model_id=job.model_id,
                generation_text=generation_text,
                utility=job.utility,
                elapsed_ms=elapsed_ms,
                cache_hit=cache_hit,
            )
        )
