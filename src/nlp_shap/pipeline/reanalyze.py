"""Reanalyze archived coalition generations without backend calls."""

import json
import time
from pathlib import Path
from typing import cast

from ..domain.generation import GenerationRecord
from ..masking.codec import MaskCodec
from ..plugins.groups import PluginGroup
from ..plugins.registry import PluginRegistry
from ..protocols.estimand import EstimandAggregator
from ..protocols.normalizer import Normalizer
from ..protocols.value import ValueFunction
from ..runtime.archive import RunArchive
from ..runtime.metrics import PerfSummary
from ..runtime.scheduler import SchedulerMetrics
from ..runtime.telemetry import NullObservabilitySink, ObservabilitySink
from ..value.generation_records import generation_record_from_text
from ..value.tfidf import TfIdfCosineValue
from .config import ExplainConfig
from .manifest import RunManifest, parse_manifest
from .result import ExplainResult, ExplainRunOutput


def reanalyze_sync(
    archive_root: Path,
    config: ExplainConfig,
    registry: PluginRegistry,
    *,
    manifest: RunManifest | None = None,
    telemetry: ObservabilitySink | None = None,
) -> ExplainRunOutput:
    """Rescore and re-aggregate an archived explain run without backend calls."""
    sink = telemetry or NullObservabilitySink()
    started = time.perf_counter()

    with sink.span("reanalyze.load"):
        archive = RunArchive.load(archive_root)
        try:
            resolved_manifest = manifest or parse_manifest(
                json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
            )
            records = list(archive.history_lazy())
            base_text = archive.read_base_generation()
        finally:
            archive.close()

    if not records:
        msg = f"archive contains no coalition records: {archive_root}"
        raise ValueError(msg)
    if base_text is None:
        msg = f"archive missing base generation: {archive_root / 'base_generation.txt'}"
        raise ValueError(msg)

    explanation = config.explanation
    aggregator = cast(
        EstimandAggregator,
        registry.resolve(PluginGroup.ESTIMANDS, explanation.estimand.value),
    )
    value_fn = cast(
        ValueFunction,
        registry.resolve(PluginGroup.VALUE_FNS, explanation.value_fn),
    )
    normalizer = cast(
        Normalizer,
        registry.resolve(PluginGroup.NORMALIZERS, explanation.normalizer),
    )

    scoring_started = time.perf_counter()
    with sink.span("reanalyze.score"):
        base_record = generation_record_from_text(base_text)
        coalition_records = [
            generation_record_from_text(record.generation_text) for record in records
        ]
        corpus = (base_record, *coalition_records)
        _fit_value_function(value_fn, corpus)
        payoffs = [
            value_fn.score(base_record, coalition_record)
            for coalition_record in coalition_records
        ]
    scoring_ms = (time.perf_counter() - scoring_started) * 1000.0

    aggregation_started = time.perf_counter()
    with sink.span("reanalyze.aggregate"):
        mask_presents = [list(MaskCodec.unpack(record.mask)) for record in records]
        raw_values = aggregator.aggregate(mask_presents, payoffs)
        normalized = normalizer.normalize(raw_values)
    aggregation_ms = (time.perf_counter() - aggregation_started) * 1000.0

    total_ms = (time.perf_counter() - started) * 1000.0
    metrics = SchedulerMetrics(
        requested=len(records),
        executed=0,
        deduplicated=0,
        cache_hits=len(records),
        kv_cache_hits=0,
    )
    result = ExplainResult(
        estimand=explanation.estimand,
        values=tuple(normalized),
    )
    return ExplainRunOutput(
        result=result,
        metrics=metrics,
        run_id=resolved_manifest.run_id,
        manifest=resolved_manifest,
        perf=PerfSummary(
            total_ms=total_ms,
            generation_ms=0.0,
            scoring_ms=scoring_ms,
            aggregation_ms=aggregation_ms,
        ),
    )


def _fit_value_function(
    value_fn: ValueFunction,
    corpus: tuple[GenerationRecord, ...],
) -> None:
    if isinstance(value_fn, TfIdfCosineValue):
        value_fn.fit(corpus)
