"""Pytest configuration and fixtures for nlp_shap tests."""

import json
import logging
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


class Benchmark:
    """Performance tracker for benchmark tests."""

    def __init__(self) -> None:
        """Initialize benchmark tracker."""
        self.timings: dict[str, list[float]] = {}
        self.enabled = False

    def start_timing(self, _test_name: str) -> float:
        """Record start time for a test."""
        return time.perf_counter()

    def end_timing(self, test_name: str, start_time: float) -> None:
        """Record end time and store duration."""
        if not self.enabled:
            return
        elapsed = time.perf_counter() - start_time
        if test_name not in self.timings:
            self.timings[test_name] = []
        self.timings[test_name].append(elapsed)

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all timed tests."""
        if not self.timings:
            return {}

        summary: dict[str, dict[str, float]] = {}
        for test_name, times in self.timings.items():
            if times:
                summary[test_name] = {
                    "min_s": min(times),
                    "median_s": sorted(times)[len(times) // 2],
                    "max_s": max(times),
                    "mean_s": sum(times) / len(times),
                    "runs": len(times),
                }
        return summary

    def save_report(self, output_path: Path | None = None) -> None:
        """Save benchmark report to JSON file."""
        if not self.enabled or not self.timings:
            return

        if output_path is None:
            output_path = Path(".pytest_benchmark/benchmark.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "benchmarks": self.get_summary(),
            "total_tests_tracked": len(self.timings),
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        logger.info("Benchmark report saved to %s", output_path)


_benchmark = Benchmark()


@pytest.fixture
def benchmark() -> Benchmark:
    """Fixture to access benchmark tracker in tests."""
    return _benchmark


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest plugins and enable benchmarking if requested."""
    if config.getoption("benchmark"):
        _benchmark.enabled = True
        logger.info("Benchmarking enabled")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options."""
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable benchmarking of all tests (tracks execution time)",
    )
    parser.addoption(
        "--benchmark-report",
        default=".pytest_benchmark/benchmark.json",
        help="Path to save benchmark report (JSON format)",
    )


def _is_bench_test(item: pytest.Item) -> bool:
    return item.get_closest_marker("bench") is not None


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Setup timing for each benchmark test if benchmarking enabled."""
    if _benchmark.enabled and _is_bench_test(item):
        item._start_time = _benchmark.start_timing(item.nodeid)


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    """Record timing after each benchmark test."""
    if (
        _benchmark.enabled
        and _is_bench_test(item)
        and call.when == "call"
        and hasattr(item, "_start_time")
    ):
        _benchmark.end_timing(item.nodeid, item._start_time)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Save benchmark report at end of test session."""
    if _benchmark.enabled:
        report_path = Path(session.config.getoption("benchmark_report"))
        _benchmark.save_report(report_path)
        summary = _benchmark.get_summary()
        if summary:
            logger.info(
                "Benchmarked %d tests. Median times range: %.2f-%.2f ms",
                len(summary),
                min(entry["median_s"] * 1000 for entry in summary.values()),
                max(entry["median_s"] * 1000 for entry in summary.values()),
            )
