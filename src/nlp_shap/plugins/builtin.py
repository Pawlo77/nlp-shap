"""Built-in plugin registration hooks."""

from ..estimation.complementary import ComplementaryEstimator
from ..estimation.exact import ExactEstimator
from ..estimation.monte_carlo import MonteCarloEstimator
from ..estimation.neyman import NeymanEstimator
from ..masking.partitions import TokenPartitioner
from ..masking.policies import DeletePolicy, NeutralPolicy, PadPolicy
from .groups import PluginGroup
from .registry import PluginRegistry


def register_builtin_plugins(registry: PluginRegistry) -> None:
    """Register built-in partition, absence-policy, and estimator plugins."""
    registry.register(PluginGroup.ESTIMATORS, "exact", ExactEstimator)
    registry.register(PluginGroup.ESTIMATORS, "mc", MonteCarloEstimator)
    registry.register(PluginGroup.ESTIMATORS, "complementary", ComplementaryEstimator)
    registry.register(PluginGroup.ESTIMATORS, "neyman_cc", NeymanEstimator)
    registry.register(PluginGroup.PARTITIONS, "tokens", TokenPartitioner)
    registry.register(PluginGroup.ABSENCE_POLICIES, "delete", DeletePolicy)
    registry.register(PluginGroup.ABSENCE_POLICIES, "pad", PadPolicy)
    registry.register(PluginGroup.ABSENCE_POLICIES, "neutral", NeutralPolicy)
