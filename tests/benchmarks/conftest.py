"""Benchmark fixtures for runtime micro-benchmarks."""

import pytest

from ._regression import BaselineStore


@pytest.fixture
def baseline_store(request: pytest.FixtureRequest) -> BaselineStore:
    """Optional baseline regression gate for benchmark tests."""
    return BaselineStore.from_config(request.config)
