"""Text MVP sign-off checklist from nlp-shap Package Rewrite (Phase 13)."""

import ast
import asyncio
import json
import tracemalloc
from collections.abc import Iterable
from itertools import product
from pathlib import Path

import httpx
import pytest

from nlp_shap.backends.api import ApiBackend
from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation.complementary import ComplementaryEstimator
from nlp_shap.estimation.estimands.banzhaf import BanzhafAggregator
from nlp_shap.estimation.estimands.shapley import ShapleyAggregator
from nlp_shap.estimation.exact import ExactEstimator
from nlp_shap.estimation.monte_carlo import MonteCarloEstimator
from nlp_shap.estimation.neyman import NeymanEstimator
from nlp_shap.masking.codec import MaskCodec
from nlp_shap.pipeline.config import (
    BackendConfig,
    GenerationConfig,
    explain_config_from_yaml,
    explain_config_to_yaml,
)
from nlp_shap.pipeline.manifest import RunManifest
from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins
from nlp_shap.runtime.archive import CoalitionRecordDraft, RunArchive
from nlp_shap.runtime.dedup import CoalitionDedupRegistry
from nlp_shap.runtime.scheduler import CoalitionJob, InferenceScheduler
from nlp_shap.runtime.store import HotResultStore
from nlp_shap.value.embedding import CosineEmbeddingValue, EuclideanEmbeddingValue
from nlp_shap.value.logprob import LogprobValue
from nlp_shap.value.tfidf import TfIdfCosineValue

CONFIG_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "pipeline"
    / "fixtures"
    / "explain_config.yaml"
)


def _make_job(snapshot: ConversationSnapshot, coalition_key: str) -> CoalitionJob:
    packed = MaskCodec.pack((True, False, True))
    return CoalitionJob(
        coalition_key=coalition_key,
        snapshot_id=snapshot.snapshot_id,
        snapshot=snapshot,
        absence_policy="delete",
        mask_words=packed.words,
        mask_n_bits=packed.n_bits,
        model_id="mock",
        utility=1.0,
    )


def test_text_mvp_domain_modules_do_not_import_backends() -> None:
    """Domain imports no torch or backend code."""
    domain_root = Path(__file__).resolve().parents[2] / "src" / "nlp_shap" / "domain"
    forbidden_roots = {"torch", "transformers", "nlp_shap.backends", "nlp_shap.plugins"}
    violations: list[str] = []
    for path in sorted(domain_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if any(
                        module == root or module.startswith(f"{root}.")
                        for root in forbidden_roots
                    ):
                        violations.append(f"{path.name}: import {module}")
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module = node.module
                if any(
                    module == root or module.startswith(f"{root}.")
                    for root in forbidden_roots
                ):
                    violations.append(f"{path.name}: from {module}")
    assert violations == []


def test_text_mvp_explain_config_yaml_round_trip() -> None:
    """YAML round-trip through ExplainConfig."""
    original_text = CONFIG_FIXTURE.read_text(encoding="utf-8")
    config = explain_config_from_yaml(original_text)
    restored = explain_config_from_yaml(explain_config_to_yaml(config))
    assert restored == config


def test_text_mvp_scheduler_streams_100_coalitions_with_bounded_memory(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Mock scheduler processes 100 streamed coalitions without unbounded growth."""

    def job_stream() -> Iterable[CoalitionJob]:
        for index in range(100):
            yield _make_job(sample_snapshot, coalition_key=f"stream-{index}")

    async def _slow_generate(snapshot: ConversationSnapshot) -> str:
        await asyncio.sleep(0)
        return snapshot.snapshot_id

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=2,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=None,
            pending_limit=4,
        )
        tracemalloc.start()
        metrics = await scheduler.run_iter(job_stream(), _slow_generate)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert metrics.requested == 100
        assert metrics.executed == 100
        assert peak < 32 * 1024 * 1024

    asyncio.run(run())


def test_text_mvp_dedup_executes_ten_backend_calls_for_hundred_jobs(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Dedup: 100 jobs / 10 masks → 10 backend calls at temp=0."""
    executed = 0

    async def counting_generate(snapshot: ConversationSnapshot) -> str:
        nonlocal executed
        executed += 1
        return "ok"

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=4,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=CoalitionDedupRegistry(),
        )
        jobs = [
            _make_job(sample_snapshot, coalition_key=f"key-{index % 10}")
            for index in range(100)
        ]
        metrics = await scheduler.run(jobs, counting_generate)
        assert metrics.requested == 100
        assert metrics.executed == 10

    asyncio.run(run())
    assert executed == 10


def test_text_mvp_history_lazy_returns_timing_and_text(
    tmp_path: Path,
    run_manifest: RunManifest,
    sample_snapshot: ConversationSnapshot,
) -> None:
    """history_lazy() yields full records with timing and generation text."""
    root = tmp_path / "signoff-archive"
    packed = MaskCodec.pack((True, False, True))

    with RunArchive.open(root, run_manifest, flush_every=25) as archive:
        for index in range(100):
            archive.append(
                CoalitionRecordDraft(
                    snapshot_id=sample_snapshot.snapshot_id,
                    coalition_key=f"key-{index}",
                    mask=packed,
                    absence_policy="delete",
                    model_id="mock",
                    generation_text=f"generation-{index}",
                    utility=float(index),
                    elapsed_ms=float(index) + 0.5,
                    cache_hit=False,
                )
            )

    with RunArchive.open(root, run_manifest) as archive:
        records = list(archive.history_lazy())

    assert len(records) == 100
    assert records[0].generation_text == "generation-0"
    assert records[0].elapsed_ms == pytest.approx(0.5)
    assert records[-1].generation_text == "generation-99"
    assert records[-1].elapsed_ms == pytest.approx(99.5)


def test_text_mvp_shapley_and_banzhaf_diverge_on_identical_samples() -> None:
    """Shapley vs Banzhaf differ on identical coalition samples."""
    masks = tuple(
        CoalitionMask.from_sequence(mask)
        for mask in product([False, True], repeat=3)
        if not all(mask)
    )
    payoffs = [1.0 if sum(mask.present) >= 2 else 0.0 for mask in masks]
    estimator = ExactEstimator()
    shapley = estimator.estimate_attributions(
        masks,
        payoffs,
        ShapleyAggregator(),
    )
    banzhaf = estimator.estimate_attributions(
        masks,
        payoffs,
        BanzhafAggregator(),
    )
    assert shapley != banzhaf


def test_text_mvp_exact_and_mc_match_on_additive_toy_game() -> None:
    """Exact and Monte Carlo estimators agree on a full-budget additive game."""
    num_players = 3
    coefficients = (1.0, 2.0, 3.0)
    player_set = PlayerSet(
        player_ids=tuple(f"p{index}" for index in range(num_players))
    )
    all_masks = tuple(
        CoalitionMask.from_sequence(mask)
        for mask in product([False, True], repeat=num_players)
        if not all(mask)
    )
    all_payoffs = [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in all_masks
    ]
    exact = ExactEstimator().estimate_attributions(
        all_masks,
        all_payoffs,
        ShapleyAggregator(),
    )
    mc_masks = list(
        MonteCarloEstimator().sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=True,
            seed=3,
        )
    )
    mc_payoffs = [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in mc_masks
    ]
    mc = MonteCarloEstimator().estimate_attributions(
        mc_masks,
        mc_payoffs,
        ShapleyAggregator(),
    )
    assert mc == pytest.approx(exact, rel=1e-6, abs=1e-6)


def test_text_mvp_complementary_and_neyman_run_on_toy_game() -> None:
    """Complementary and Neyman estimators produce attributions on toy games."""
    player_set = PlayerSet(player_ids=("p0", "p1", "p2"))
    complementary = ComplementaryEstimator()
    cc_masks = list(
        complementary.sample_masks(
            player_set,
            budget_fraction=0.6,
            include_minimal_masks=True,
            seed=5,
        )
    )
    cc_payoffs = [float(sum(mask.present)) for mask in cc_masks]
    cc_values = complementary.estimate_attributions(cc_masks, cc_payoffs)
    assert len(cc_values) == 3

    neyman = NeymanEstimator(use_standard_method=False)
    phase_one = list(
        neyman.sample_masks(
            player_set,
            budget_fraction=0.6,
            include_minimal_masks=False,
            seed=11,
        )
    )
    payoffs_one = [float(sum(mask.present)) for mask in phase_one]
    neyman.begin_allocation(phase_one, payoffs_one)
    phase_two = list(neyman.sample_allocation_masks())
    payoffs_two = [float(sum(mask.present)) for mask in phase_two]
    neyman_values = neyman.estimate_attributions(
        phase_one + phase_two,
        payoffs_one + payoffs_two,
    )
    assert len(neyman_values) == 3


def test_text_mvp_value_functions_cover_legacy_text_modes() -> None:
    """All 0.x text value/similarity modes are registered and constructible."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    expected = (
        "tfidf_cosine",
        "embedding_cosine",
        "embedding_euclidean",
        "logprob",
    )
    names = registry.names(PluginGroup.VALUE_FNS)
    for name in expected:
        assert name in names
    assert isinstance(
        registry.resolve(PluginGroup.VALUE_FNS, "tfidf_cosine"),
        TfIdfCosineValue,
    )
    assert isinstance(
        registry.resolve(PluginGroup.VALUE_FNS, "embedding_cosine"),
        CosineEmbeddingValue,
    )
    assert isinstance(
        registry.resolve(PluginGroup.VALUE_FNS, "embedding_euclidean"),
        EuclideanEmbeddingValue,
    )
    assert isinstance(registry.resolve(PluginGroup.VALUE_FNS, "logprob"), LogprobValue)


def test_text_mvp_api_backend_smoke() -> None:
    """API backend smoke test parses chat-completions JSON."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "remote-model"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    backend = ApiBackend(
        BackendConfig(kind="api", model_id="remote-model", api_host="http://mock"),
        transport=httpx.MockTransport(handler),
    )
    turn = Turn(messages=(Message(role=Role.USER, text="hello"),))
    snapshot = ConversationSnapshot.from_turns((turn,))

    async def run() -> str:
        try:
            record = await backend.generate(snapshot, 8, 0.0, 1)
            return record.text
        finally:
            await backend.aclose()

    assert asyncio.run(run()) == "ok"
    assert calls == 1


def test_text_mvp_hierarchical_explainer_absent_from_package() -> None:
    """Hierarchical explainer is absent from package source and public docs."""
    package_root = Path(__file__).resolve().parents[2] / "src" / "nlp_shap"
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    for root in (package_root, docs_root):
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".rst"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert "HierarchicalExplainer" not in text, f"found in {path}"
            assert "shap/hierarchical" not in text, f"found in {path}"
