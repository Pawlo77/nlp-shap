"""Multimodal explainability tool for NLP based on Shapley value."""

from importlib.metadata import PackageNotFoundError, version

from ._logging import bootstrap_logging
from .domain.estimands import Estimand
from .estimation.estimands import BanzhafAggregator, ShapleyAggregator
from .pipeline.manifest import RunManifest, parse_manifest
from .pipeline.result import ExplainResult

__all__ = [
    "BanzhafAggregator",
    "Estimand",
    "ExplainResult",
    "RunManifest",
    "ShapleyAggregator",
    "__version__",
    "parse_manifest",
]

try:
    __version__ = version("nlp-shap")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
"""Current package version."""

bootstrap_logging()
