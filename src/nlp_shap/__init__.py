"""Multimodal explainability tool for NLP based on Shapley value."""

from importlib.metadata import PackageNotFoundError, version

from ._logging import bootstrap_logging
from .domain import (
    CoalitionMask,
    ConversationSnapshot,
    CooperativeGame,
    Estimand,
    Message,
    PlayerSet,
    Role,
    Turn,
)
from .estimation.estimands import BanzhafAggregator, ShapleyAggregator
from .estimation.exact import ExactEstimator
from .pipeline import (
    ExplainConfig,
    ExplainResult,
    RunManifest,
    explain_config_from_yaml,
    explain_config_to_yaml,
    parse_manifest,
)
from .plugins import PluginGroup, PluginRegistry

__all__ = [
    "BanzhafAggregator",
    "CoalitionMask",
    "ConversationSnapshot",
    "CooperativeGame",
    "Estimand",
    "ExactEstimator",
    "ExplainConfig",
    "ExplainResult",
    "Message",
    "PlayerSet",
    "PluginGroup",
    "PluginRegistry",
    "Role",
    "RunManifest",
    "ShapleyAggregator",
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
