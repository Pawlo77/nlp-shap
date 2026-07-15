"""Explain pipeline result types."""

from dataclasses import dataclass

from ..domain.estimands import Estimand
from ..runtime.metrics import PerfSummary
from ..runtime.scheduler import SchedulerMetrics
from .manifest import RunManifest


@dataclass(frozen=True, slots=True)
class ExplainResult:
    """Attribution output for a single explain run."""

    estimand: Estimand
    """Estimand used to aggregate coalition samples."""

    values: tuple[float, ...]
    """Per-player attribution values."""


@dataclass(frozen=True, slots=True)
class ExplainRunOutput:
    """Explain pipeline output including scheduler metrics."""

    result: ExplainResult
    """Aggregated and normalized attribution values."""

    metrics: SchedulerMetrics
    """Coalition scheduler counters for the run."""

    run_id: str
    """Stable identifier for the explain run."""

    manifest: RunManifest
    """Manifest metadata persisted with optional archives."""

    perf: PerfSummary | None = None
    """Optional wall-clock breakdown for the run."""
