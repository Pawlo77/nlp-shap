"""Smoke tests for the nlp_shap package."""

from importlib.metadata import version

import nlp_shap


def test_version_is_defined() -> None:
    """Package exposes the installed distribution version."""
    assert nlp_shap.__version__ == version("nlp-shap")
