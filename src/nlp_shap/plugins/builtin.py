"""Built-in plugin registration hooks."""

from __future__ import annotations

from ..masking.partitions import TokenPartitioner
from ..masking.policies import DeletePolicy, NeutralPolicy, PadPolicy
from .groups import PluginGroup
from .registry import PluginRegistry


def register_builtin_plugins(registry: PluginRegistry) -> None:
    """Register built-in partition and absence-policy plugins."""
    registry.register(PluginGroup.PARTITIONS, "tokens", TokenPartitioner)
    registry.register(PluginGroup.ABSENCE_POLICIES, "delete", DeletePolicy)
    registry.register(PluginGroup.ABSENCE_POLICIES, "pad", PadPolicy)
    registry.register(PluginGroup.ABSENCE_POLICIES, "neutral", NeutralPolicy)
