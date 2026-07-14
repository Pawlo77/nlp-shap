"""Built-in plugin registration hooks."""

from ..estimation.complementary import ComplementaryEstimator
from ..estimation.exact import ExactEstimator
from ..estimation.monte_carlo import MonteCarloEstimator
from ..estimation.neyman import NeymanEstimator
from ..estimation.normalizers import (
    AbsSumNormalizer,
    IdentityNormalizer,
    MinMaxNormalizer,
    PowerShiftNormalizer,
)
from ..masking.partitions import TokenPartitioner
from ..masking.policies import DeletePolicy, NeutralPolicy, PadPolicy
from ..value.embedding import CosineEmbeddingValue, EuclideanEmbeddingValue
from ..value.logprob import LogprobValue
from ..value.tfidf import TfIdfCosineValue
from .groups import PluginGroup
from .registry import PluginRegistry


def register_builtin_plugins(registry: PluginRegistry) -> None:
    """Register built-in partition, policy, estimator, value, and normalizer plugins."""
    registry.register(PluginGroup.ESTIMATORS, "exact", ExactEstimator)
    registry.register(PluginGroup.ESTIMATORS, "mc", MonteCarloEstimator)
    registry.register(PluginGroup.ESTIMATORS, "complementary", ComplementaryEstimator)
    registry.register(PluginGroup.ESTIMATORS, "neyman_cc", NeymanEstimator)
    registry.register(PluginGroup.VALUE_FNS, "tfidf_cosine", TfIdfCosineValue)
    registry.register(PluginGroup.VALUE_FNS, "embedding_cosine", CosineEmbeddingValue)
    registry.register(
        PluginGroup.VALUE_FNS,
        "embedding_euclidean",
        EuclideanEmbeddingValue,
    )
    registry.register(PluginGroup.VALUE_FNS, "logprob", LogprobValue)
    registry.register(PluginGroup.NORMALIZERS, "identity", IdentityNormalizer)
    registry.register(PluginGroup.NORMALIZERS, "abs_sum", AbsSumNormalizer)
    registry.register(PluginGroup.NORMALIZERS, "power_shift", PowerShiftNormalizer)
    registry.register(PluginGroup.NORMALIZERS, "min_max", MinMaxNormalizer)
    registry.register(PluginGroup.PARTITIONS, "tokens", TokenPartitioner)
    registry.register(PluginGroup.ABSENCE_POLICIES, "delete", DeletePolicy)
    registry.register(PluginGroup.ABSENCE_POLICIES, "pad", PadPolicy)
    registry.register(PluginGroup.ABSENCE_POLICIES, "neutral", NeutralPolicy)
