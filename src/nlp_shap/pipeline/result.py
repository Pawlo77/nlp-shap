"""Explain pipeline result types."""

from dataclasses import dataclass

from ..domain.estimands import Estimand


@dataclass(frozen=True, slots=True)
class ExplainResult:
    """Attribution output for a single explain run."""

    estimand: Estimand
    """Estimand used to aggregate coalition samples."""

    values: tuple[float, ...]
    """Per-player attribution values."""
