"""Multimodal explainability tool for NLP based on Shapley value."""

from importlib.metadata import PackageNotFoundError, version

from ._logging import bootstrap_logging
from .domain import (
    CoalitionMask,
    ConversationSnapshot,
    CooperativeGame,
    Estimand,
    GenerationRecord,
    Message,
    PlayerSet,
    Role,
    Turn,
)
from .estimation.complementary import ComplementaryEstimator
from .estimation.estimands import BanzhafAggregator, ShapleyAggregator
from .estimation.exact import ExactEstimator
from .estimation.monte_carlo import MonteCarloEstimator
from .estimation.neyman import NeymanEstimator
from .estimation.normalizers import IdentityNormalizer, MinMaxNormalizer
from .pipeline import (
    ExplainConfig,
    ExplainResult,
    ExplainRunner,
    ExplainRunOutput,
    RunManifest,
    explain_config_from_yaml,
    explain_config_to_yaml,
    parse_manifest,
)
from .plugins import PluginGroup, PluginRegistry
from .value import CosineEmbeddingValue, LogprobValue, TfIdfCosineValue

__all__ = [
    "BanzhafAggregator",
    "CoalitionMask",
    "ComplementaryEstimator",
    "ConversationSnapshot",
    "CooperativeGame",
    "CosineEmbeddingValue",
    "Estimand",
    "ExactEstimator",
    "ExplainConfig",
    "ExplainResult",
    "ExplainRunOutput",
    "ExplainRunner",
    "GenerationRecord",
    "IdentityNormalizer",
    "LogprobValue",
    "Message",
    "MinMaxNormalizer",
    "MonteCarloEstimator",
    "NeymanEstimator",
    "PlayerSet",
    "PluginGroup",
    "PluginRegistry",
    "Role",
    "RunManifest",
    "ShapleyAggregator",
    "TfIdfCosineValue",
    "Turn",
    "__version__",
    "explain_config_from_yaml",
    "explain_config_to_yaml",
    "parse_manifest",
]

try:
    __version__ = version("nlp-shap")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
"""Current package version."""

bootstrap_logging()
