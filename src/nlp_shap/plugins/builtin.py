"""Built-in plugin registration hooks."""

from __future__ import annotations

from .registry import PluginRegistry


def register_builtin_plugins(registry: PluginRegistry) -> None:
    """Register built-in plugins.

    Phase 1 ships an empty hook; concrete registrations arrive in Phase 2.
    """
