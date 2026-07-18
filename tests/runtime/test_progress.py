"""Tests for optional coalition progress callbacks."""

import asyncio
from dataclasses import dataclass, field

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.pipeline.config import GenerationConfig
from nlp_shap.runtime.progress import NullCoalitionProgress
from nlp_shap.runtime.scheduler import CoalitionJob, InferenceScheduler
from nlp_shap.runtime.store import HotResultStore


@dataclass
class RecordingProgress:
    """Capture progress callbacks for assertions."""

    planned: list[int] = field(default_factory=list)
    finished: list[tuple[int, int]] = field(default_factory=list)

    def on_coalitions_planned(self, total: int) -> None:
        self.planned.append(total)

    def on_coalition_finished(self, done: int, total: int) -> None:
        self.finished.append((done, total))


def _job(key: str) -> CoalitionJob:
    snapshot = ConversationSnapshot(
        turns=(Turn(messages=(Message(role=Role.USER, text=key),)),),
        snapshot_id="s",
    )
    return CoalitionJob(
        coalition_key=key,
        snapshot_id="s",
        snapshot=snapshot,
        absence_policy="pad",
        mask_words=b"\x00",
        mask_n_bits=1,
        model_id="mock",
        utility=0.0,
    )


async def _generate(snapshot: ConversationSnapshot) -> str:
    _ = snapshot
    return "ok"


def test_null_progress_is_safe_noop() -> None:
    progress = NullCoalitionProgress()
    progress.on_coalitions_planned(3)
    progress.on_coalition_finished(1, 3)


def test_scheduler_reports_planned_and_finished_progress() -> None:
    progress = RecordingProgress()
    jobs = (_job("a"), _job("b"), _job("c"))
    scheduler = InferenceScheduler(
        max_inflight=2,
        generation=GenerationConfig(),
        store=HotResultStore(maxsize=8),
        progress=progress,
    )
    metrics = asyncio.run(scheduler.run(jobs, _generate))
    assert metrics.executed == 3
    assert progress.planned == [3]
    assert progress.finished[-1] == (3, 3)
    assert len(progress.finished) == 3
