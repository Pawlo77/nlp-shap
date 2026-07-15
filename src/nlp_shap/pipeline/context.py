"""Shared state for one explain pipeline run."""

from dataclasses import dataclass
from pathlib import Path

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..plugins.registry import PluginRegistry
from ..runtime.telemetry import NullObservabilitySink, ObservabilitySink
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

    archive_root: Path | None = None
    """Optional filesystem root for run archive persistence."""

    telemetry: ObservabilitySink | None = None
    """Optional observability sink for stage spans."""

    def resolved_telemetry(self) -> ObservabilitySink:
        """Return the configured telemetry sink or a no-op implementation."""
        return self.telemetry or NullObservabilitySink()
