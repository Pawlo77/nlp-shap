"""Plugin registry context helpers."""

from dataclasses import dataclass

from .registry import PluginRegistry


@dataclass(frozen=True, slots=True)
class PluginContext:
    """Registry handle passed through pipeline setup."""

    registry: PluginRegistry
    """Plugin registry used to resolve pipeline components."""
