"""Plugin group identifiers for entry-point loading."""

from enum import StrEnum


class PluginGroup(StrEnum):
    """Entry-point groups registered in packaging metadata."""

    ESTIMATORS = "nlp_shap.estimators"
    """Coalition-sampling estimator plugins."""

    ESTIMANDS = "nlp_shap.estimands"
    """Coalition payoff aggregation plugins."""

    VALUE_FNS = "nlp_shap.value_fns"
    """Utility scoring plugins for coalition outputs."""

    BACKENDS = "nlp_shap.backends"
    """Model inference backend plugins."""

    PARTITIONS = "nlp_shap.partitions"
    """Player-partitioning plugins for conversation snapshots."""
