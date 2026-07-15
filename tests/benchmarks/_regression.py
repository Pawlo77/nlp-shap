"""Baseline comparison helpers for runtime micro-benchmarks."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

BASELINES_PATH = Path(__file__).resolve().parent / "baselines.json"
REGRESSION_TOLERANCE = 0.10


@dataclass(frozen=True, slots=True)
class BaselineStore:
    """Load recorded medians and compare new timings."""

    baselines: dict[str, float]
    regression_enabled: bool

    @classmethod
    def from_config(cls, config: pytest.Config) -> "BaselineStore":
        regression_enabled = bool(config.getoption("bench_regression"))
        baselines: dict[str, float] = {}
        if BASELINES_PATH.is_file():
            payload = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
            raw = payload.get("medians_s", {})
            if isinstance(raw, dict):
                baselines = {str(key): float(value) for key, value in raw.items()}
        return cls(baselines=baselines, regression_enabled=regression_enabled)

    def check(self, key: str, elapsed_s: float, *, ceiling_s: float) -> None:
        """Assert elapsed time stays under a hard ceiling and optional baseline."""
        if elapsed_s > ceiling_s:
            msg = f"{key} took {elapsed_s:.4f}s; ceiling is {ceiling_s:.4f}s"
            raise AssertionError(msg)
        if not self.regression_enabled or key not in self.baselines:
            return
        baseline = self.baselines[key]
        limit = baseline * (1.0 + REGRESSION_TOLERANCE)
        if elapsed_s > limit:
            msg = (
                f"{key} regressed: {elapsed_s:.4f}s > {limit:.4f}s "
                f"(baseline {baseline:.4f}s + {REGRESSION_TOLERANCE:.0%})"
            )
            raise AssertionError(msg)
