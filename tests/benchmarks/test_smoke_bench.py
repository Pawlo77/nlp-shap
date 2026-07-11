"""Benchmark smoke tests."""

import pytest


@pytest.mark.bench
def test_import_bench() -> None:
    """Importing the package should stay fast."""
    import nlp_shap

    assert nlp_shap.__version__
