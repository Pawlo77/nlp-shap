"""Backend plugin construction helpers."""

from typing import cast

from ..pipeline.config import BackendConfig
from ..protocols.backend import GenerativeBackend
from .groups import PluginGroup
from .registry import PluginRegistry


def instantiate_backend(
    config: BackendConfig,
    registry: PluginRegistry,
) -> GenerativeBackend:
    """Construct a backend instance for ``config.kind`` and ``config.model_id``."""
    backend_type = registry.resolve_type(PluginGroup.BACKENDS, config.kind)
    return cast(GenerativeBackend, backend_type(config.model_id))
