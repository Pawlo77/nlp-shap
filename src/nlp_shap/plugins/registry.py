"""Plugin registry with entry-point discovery."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import cast

from .groups import PluginGroup

PluginFactory = Callable[[], object]
"""Callable that materializes a plugin instance."""


class PluginRegistry:
    """Register and resolve pipeline plugins by group and name."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, PluginFactory]] = {}

    def register(
        self,
        group: PluginGroup | str,
        name: str,
        factory: PluginFactory,
    ) -> None:
        """Register a plugin factory under ``group`` and ``name``."""
        group_key = str(group)
        factories = self._factories.setdefault(group_key, {})
        factories[name] = factory

    def resolve(self, group: PluginGroup | str, name: str) -> object:
        """Instantiate a plugin registered under ``group`` and ``name``."""
        group_key = str(group)
        try:
            factory = self._factories[group_key][name]
        except KeyError as error:
            msg = f"unknown plugin {name!r} in group {group_key!r}"
            raise LookupError(msg) from error
        return factory()

    def names(self, group: PluginGroup | str) -> tuple[str, ...]:
        """Return sorted plugin names registered for ``group``."""
        group_key = str(group)
        return tuple(sorted(self._factories.get(group_key, {})))

    def load_entry_points(self, group: PluginGroup | str) -> None:
        """Discover and register plugins from a packaging entry-point group."""
        group_key = str(group)
        discovered = entry_points(group=group_key)
        for entry_point in discovered:
            target = entry_point.load()
            if isinstance(target, type):
                factory: PluginFactory = target
            else:
                factory = cast(PluginFactory, target)
            self.register(group_key, entry_point.name, factory)
