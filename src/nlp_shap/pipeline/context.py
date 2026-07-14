"""Shared state for one explain pipeline run."""

from dataclasses import dataclass

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..plugins.registry import PluginRegistry
from .config import ExplainConfig


@dataclass(frozen=True, slots=True)
class ExplainContext:
    """Inputs resolved for one explain orchestrator run."""

    config: ExplainConfig
    """Validated explain configuration."""

    snapshot: ConversationSnapshot
    """Conversation under explanation."""

    player_set: PlayerSet
    """Explainability players derived from the snapshot."""

    registry: PluginRegistry
    """Plugin registry used to resolve pipeline components."""

    run_id: str
    """Stable identifier for archive paths and manifests."""
