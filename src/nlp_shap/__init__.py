"""Multimodal explainability tool for NLP based on Shapley value."""

from importlib.metadata import PackageNotFoundError, version

from nlp_shap._logging import bootstrap_logging

__all__ = ["__version__"]

try:
    __version__ = version("nlp-shap")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
"""Current package version."""

bootstrap_logging()
