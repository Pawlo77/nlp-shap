"""Explain pipeline orchestration."""

import time
from typing import Any, cast

from ..backends.mock.generation import generation_record_from_snapshot
from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.generation import GenerationRecord
from ..domain.players import PlayerSet
from ..masking.builder import MaskBuilder
from ..masking.codec import MaskCodec
from ..plugins.groups import PluginGroup
from ..protocols.backend import GenerativeBackend
from ..protocols.estimand import EstimandAggregator
from ..protocols.estimator import EstimatorStrategy
from ..protocols.masking import AbsencePolicy
from ..protocols.normalizer import Normalizer
from ..protocols.value import ValueFunction
from ..runtime.archive import BASE_GENERATION_FILE, RunArchive
from ..runtime.dedup import CoalitionDedupRegistry, build_coalition_key, dedup_enabled
from ..runtime.metrics import PerfSummary
from ..runtime.scheduler import CoalitionJob, InferenceScheduler
from ..runtime.store import HotResultStore
from ..value.tfidf import TfIdfCosineValue
from .context import ExplainContext
from .manifest import RunManifest
from .result import ExplainResult, ExplainRunOutput


class ExplainOrchestrator:
    """Wire partition, sampling, generation, scoring, and aggregation."""

    def __init__(
        self,
        context: ExplainContext,
        backend: GenerativeBackend,
    ) -> None:
        self._context = context
        self._backend = backend

    async def run(self) -> ExplainRunOutput:
        """Execute the configured explain pipeline and return attributions."""
        telemetry = self._context.resolved_telemetry()
        run_started = time.perf_counter()
        generation_ms = 0.0
        scoring_ms = 0.0
        aggregation_ms = 0.0

        with telemetry.span("orchestrator.run"):
            config = self._context.config
            explanation = config.explanation
            registry = self._context.registry
            snapshot = self._context.snapshot
            player_set = self._context.player_set

            estimator = cast(
                EstimatorStrategy,
                registry.resolve(PluginGroup.ESTIMATORS, explanation.estimator),
            )
            estimator.bind_snapshot(snapshot)

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

            absence_policy = cast(
                AbsencePolicy,
                registry.resolve(
                    PluginGroup.ABSENCE_POLICIES,
                    explanation.absence_policy,
                ),
            )
            mask_builder = MaskBuilder(absence_policy)

            archive: RunArchive | None = None
            manifest = RunManifest(
                estimand=explanation.estimand,
                run_id=self._context.run_id,
            )
            if self._context.archive_root is not None:
                archive = RunArchive.open(
                    self._context.archive_root,
                    manifest,
                    flush_every=explanation.archive.flush_every,
                )

            try:
                base_started = time.perf_counter()
                with telemetry.span("orchestrator.generate_base"):
                    base_record = await self._generate_base(mask_builder, player_set)
                generation_ms += (time.perf_counter() - base_started) * 1000.0

                if archive is not None:
                    archive.write_base_generation(base_record.text)
                    archive_root = self._context.archive_root
                    if archive_root is None:
                        msg = "archive root missing while archive is open"
                        raise RuntimeError(msg)
                    base_path = archive_root / BASE_GENERATION_FILE
                    if not base_path.is_file():
                        msg = "base generation was not persisted before coalition loop"
                        raise RuntimeError(msg)

                with telemetry.span("estimator.sample_masks"):
                    masks = list(
                        estimator.sample_masks(
                            player_set,
                            explanation.budget.fraction,
                            explanation.include_minimal_masks,
                            explanation.seed,
                        )
                    )

                store = HotResultStore()
                dedup = (
                    CoalitionDedupRegistry()
                    if dedup_enabled(explanation.dedup, config.generation)
                    else None
                )
                scheduler = InferenceScheduler(
                    max_inflight=explanation.max_inflight,
                    generation=config.generation,
                    store=store,
                    dedup=dedup,
                    archive=archive,
                )

                jobs = self._build_jobs(mask_builder, player_set, masks)
                generation = config.generation

                async def generate(snapshot_for_job: ConversationSnapshot) -> str:
                    record = await self._backend.generate(
                        snapshot_for_job,
                        generation.max_new_tokens,
                        generation.temperature,
                        generation.top_k,
                    )
                    return record.text

                schedule_started = time.perf_counter()
                with telemetry.span("orchestrator.schedule"):
                    metrics = await scheduler.run(jobs, generate)
                generation_ms += (time.perf_counter() - schedule_started) * 1000.0

                coalition_records = self._collect_generation_records(jobs, store)
                corpus = (base_record, *coalition_records)

                scoring_started = time.perf_counter()
                with telemetry.span("orchestrator.score"):
                    self._fit_value_function(value_fn, corpus)
                    payoffs = [
                        value_fn.score(base_record, coalition_record)
                        for coalition_record in coalition_records
                    ]
                scoring_ms = (time.perf_counter() - scoring_started) * 1000.0

                aggregation_started = time.perf_counter()
                with telemetry.span("orchestrator.aggregate"):
                    raw_values = self._estimate_attributions(
                        estimator,
                        masks,
                        payoffs,
                        aggregator,
                    )
                    normalized = normalizer.normalize(raw_values)
                aggregation_ms = (time.perf_counter() - aggregation_started) * 1000.0
            finally:
                if archive is not None:
                    archive.close()

        total_ms = (time.perf_counter() - run_started) * 1000.0
        result = ExplainResult(
            estimand=explanation.estimand,
            values=tuple(normalized),
        )
        return ExplainRunOutput(
            result=result,
            metrics=metrics,
            run_id=self._context.run_id,
            manifest=manifest,
            perf=PerfSummary(
                total_ms=total_ms,
                generation_ms=generation_ms,
                scoring_ms=scoring_ms,
                aggregation_ms=aggregation_ms,
            ),
        )

    async def _generate_base(
        self,
        mask_builder: MaskBuilder,
        player_set: PlayerSet,
    ) -> GenerationRecord:
        config = self._context.config
        if not config.generation.precompute_base:
            msg = "precompute_base must be enabled for value-function scoring"
            raise RuntimeError(msg)
        grand_mask = CoalitionMask.from_sequence((True,) * player_set.num_players)
        masked = mask_builder.view(
            self._context.snapshot,
            player_set,
            grand_mask,
        )
        rendered = mask_builder.render(masked)
        generation = config.generation
        generated = await self._backend.generate(
            rendered,
            generation.max_new_tokens,
            generation.temperature,
            generation.top_k,
        )
        return generation_record_from_snapshot(generated.text, rendered)

    def _build_jobs(
        self,
        mask_builder: MaskBuilder,
        player_set: PlayerSet,
        masks: list[CoalitionMask],
    ) -> list[CoalitionJob]:
        config = self._context.config
        jobs: list[CoalitionJob] = []
        for mask in masks:
            masked = mask_builder.view(self._context.snapshot, player_set, mask)
            rendered = mask_builder.render(masked)
            packed = MaskCodec.pack(mask.present)
            coalition_key = build_coalition_key(
                snapshot_id=self._context.snapshot.snapshot_id,
                player_ids=player_set.player_ids,
                mask_present=mask.present,
                absence_policy=mask_builder.policy_name,
                model_id=self._backend.model_id,
                generation=config.generation,
            )
            jobs.append(
                CoalitionJob(
                    coalition_key=coalition_key,
                    snapshot_id=self._context.snapshot.snapshot_id,
                    snapshot=rendered,
                    absence_policy=mask_builder.policy_name,
                    mask_words=packed.words,
                    mask_n_bits=packed.n_bits,
                    model_id=self._backend.model_id,
                    utility=0.0,
                )
            )
        return jobs

    def _collect_generation_records(
        self,
        jobs: list[CoalitionJob],
        store: HotResultStore,
    ) -> list[GenerationRecord]:
        records: list[GenerationRecord] = []
        for job in jobs:
            text = store.get(job.coalition_key)
            if text is None:
                msg = f"missing generation for coalition {job.coalition_key!r}"
                raise RuntimeError(msg)
            records.append(generation_record_from_snapshot(text, job.snapshot))
        return records

    @staticmethod
    def _fit_value_function(
        value_fn: ValueFunction,
        corpus: tuple[GenerationRecord, ...],
    ) -> None:
        if isinstance(value_fn, TfIdfCosineValue):
            value_fn.fit(corpus)

    @staticmethod
    def _estimate_attributions(
        estimator: EstimatorStrategy,
        masks: list[CoalitionMask],
        payoffs: list[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        estimator_impl = cast(Any, estimator)
        match estimator.name:
            case "exact" | "mc":
                return cast(
                    list[float],
                    estimator_impl.estimate_attributions(masks, payoffs, aggregator),
                )
            case "complementary" | "neyman_cc":
                return cast(
                    list[float],
                    estimator_impl.estimate_attributions(masks, payoffs),
                )
            case _ as unsupported:
                msg = f"unsupported estimator for orchestrator: {unsupported!r}"
                raise ValueError(msg)
