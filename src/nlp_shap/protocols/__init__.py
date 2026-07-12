"""Protocol contracts for pipeline plugins."""

from .backend import GenerationResult, GenerativeBackend
from .embedding import EmbeddingProvider
from .estimand import EstimandAggregator
from .estimator import EstimatorStrategy
from .masking import AbsencePolicy
from .partition import PlayerPartitioner
from .value import ValueFunction

__all__ = [
    "AbsencePolicy",
    "EmbeddingProvider",
    "EstimandAggregator",
    "EstimatorStrategy",
    "GenerationResult",
    "GenerativeBackend",
    "PlayerPartitioner",
    "ValueFunction",
]
