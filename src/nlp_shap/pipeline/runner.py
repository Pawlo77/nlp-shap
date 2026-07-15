"""Public explain pipeline entry point."""

import asyncio
from pathlib import Path

from ..domain.conversation import ConversationSnapshot
from ..masking.partitions import TokenPartitioner
from ..plugins import PluginGroup, PluginRegistry, register_builtin_plugins
from ..plugins.backends import instantiate_backend
from ..runtime.telemetry import NullObservabilitySink, ObservabilitySink
from .config import ExplainConfig
from .context import ExplainContext
from .orchestrator import ExplainOrchestrator
from .reanalyze import reanalyze_sync
from .result import ExplainRunOutput


class ExplainRunner:
    """Run the explain pipeline for a conversation snapshot."""

    def __init__(
        self,
        config: ExplainConfig,
        registry: PluginRegistry | None = None,
        telemetry: ObservabilitySink | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or _default_registry()
        self._telemetry = telemetry or NullObservabilitySink()

    async def explain(self, snapshot: ConversationSnapshot) -> ExplainRunOutput:
        """Execute the async explain pipeline."""
        context = self._build_context(snapshot)
        backend = instantiate_backend(self._config.backend, self._registry)
        try:
            orchestrator = ExplainOrchestrator(context, backend)
            return await orchestrator.run()
        finally:
            aclose = getattr(backend, "aclose", None)
            if aclose is not None:
                await aclose()

    def explain_sync(self, snapshot: ConversationSnapshot) -> ExplainRunOutput:
        """Execute the explain pipeline on the current event loop policy."""
        return asyncio.run(self.explain(snapshot))

    def reanalyze(self, archive_root: Path | str) -> ExplainRunOutput:
        """Rescore an archived run with the runner's current explain settings."""
        return reanalyze_sync(
            Path(archive_root),
            self._config,
            self._registry,
            telemetry=self._telemetry,
        )

    def _build_context(self, snapshot: ConversationSnapshot) -> ExplainContext:
        partitioner = self._registry.resolve(
            PluginGroup.PARTITIONS,
            self._config.explanation.players,
        )
        if not isinstance(partitioner, TokenPartitioner):
            msg = "only token partitioners are supported by ExplainRunner"
            raise TypeError(msg)
        player_set = partitioner.partition(snapshot)
        run_id = f"{snapshot.snapshot_id}-{self._config.explanation.seed}"
        return ExplainContext(
            config=self._config,
            snapshot=snapshot,
            player_set=player_set,
            registry=self._registry,
            run_id=run_id,
            archive_root=self._resolve_archive_root(run_id),
            telemetry=self._telemetry,
        )

    def _resolve_archive_root(self, run_id: str) -> Path | None:
        template = self._config.explanation.archive.path.strip()
        if not template:
            return None
        return Path(template.format(run_id=run_id))


def _default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    for group in (
        PluginGroup.ESTIMATORS,
        PluginGroup.ESTIMANDS,
        PluginGroup.VALUE_FNS,
        PluginGroup.NORMALIZERS,
        PluginGroup.BACKENDS,
        PluginGroup.PARTITIONS,
        PluginGroup.ABSENCE_POLICIES,
    ):
        registry.load_entry_points(group)
    return registry
