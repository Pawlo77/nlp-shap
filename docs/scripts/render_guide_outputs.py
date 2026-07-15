"""Execute guide snippets and write stdout / figures for Sphinx guides."""

import asyncio
import io
import sys
import tempfile
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "_static" / "guide_outputs"


def _capture(run: Callable[[], None]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run()
    return buffer.getvalue().strip()


def _write_text(snippet_id: str, text: str) -> None:
    path = OUTPUT_DIR / f"{snippet_id}.txt"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_figure(snippet_id: str, figure: object) -> None:
    path = OUTPUT_DIR / f"{snippet_id}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(figure)


def getting_started_majority() -> None:
    from itertools import product

    from nlp_shap import BanzhafAggregator, ShapleyAggregator

    num_players = 3
    masks = [tuple(bits) for bits in product([False, True], repeat=num_players)]

    def majority_payoff(mask: tuple[bool, ...]) -> float:
        return 1.0 if sum(mask) >= 2 else 0.0

    payoffs = [majority_payoff(mask) for mask in masks]
    shapley = ShapleyAggregator().aggregate(masks, payoffs)
    banzhaf = BanzhafAggregator().aggregate(masks, payoffs)
    print("Shapley:", shapley)
    print("Banzhaf:", banzhaf)


def getting_started_explain() -> None:
    from nlp_shap import ExplainConfig, ExplainRunner
    from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
    from nlp_shap.domain.enums import Role

    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="refund my order"),)),
    ))
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "estimator": "exact",
            "value_fn": "tfidf_cosine",
            "absence_policy": "pad",
        },
    })
    output = ExplainRunner(config).explain_sync(snapshot)
    print(output.result.values)


def getting_started_manifest() -> None:
    from nlp_shap import Estimand, ExplainResult, RunManifest, parse_manifest

    result = ExplainResult(
        estimand=Estimand.SHAPLEY,
        values=(0.333, 0.333, 0.333),
    )
    manifest = RunManifest(estimand=result.estimand, run_id="run-42")
    restored = parse_manifest(manifest.to_dict())
    print("estimand:", restored.estimand.value)
    print("run_id:", restored.run_id)


def estimands_coalition_weights() -> None:
    from nlp_shap import BanzhafAggregator, ShapleyAggregator

    shapley = ShapleyAggregator()
    banzhaf = BanzhafAggregator()
    num_players = 4
    for coalition_size in range(num_players):
        print(
            f"k={coalition_size}",
            round(shapley.coalition_weight(coalition_size, num_players), 6),
            round(banzhaf.coalition_weight(coalition_size, num_players), 6),
        )


def masking_partition() -> None:
    from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
    from nlp_shap.domain.enums import Role
    from nlp_shap.masking.partitions import TokenPartitioner

    turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
    snapshot = ConversationSnapshot(turns=(turn,), snapshot_id="demo-snap")
    players = TokenPartitioner().partition(snapshot)
    print(players.player_ids)


def masking_render_coalition() -> None:
    from nlp_shap.domain.coalition import CoalitionMask
    from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
    from nlp_shap.domain.enums import Role
    from nlp_shap.masking import DeletePolicy, MaskBuilder
    from nlp_shap.masking.partitions import TokenPartitioner

    turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
    snapshot = ConversationSnapshot(turns=(turn,), snapshot_id="demo-snap")
    players = TokenPartitioner().partition(snapshot)
    mask = CoalitionMask.from_sequence((True, False, True))
    builder = MaskBuilder(DeletePolicy())
    view = builder.view(snapshot, players, mask)
    rendered = builder.render(view)
    print(rendered.turns[0].messages[0].text)


def masking_absence_policies() -> None:
    from nlp_shap.domain.coalition import CoalitionMask
    from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
    from nlp_shap.domain.enums import Role
    from nlp_shap.masking import MaskBuilder, NeutralPolicy, PadPolicy
    from nlp_shap.masking.partitions import TokenPartitioner

    turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
    snapshot = ConversationSnapshot(turns=(turn,), snapshot_id="demo-snap")
    players = TokenPartitioner().partition(snapshot)
    mask = CoalitionMask.from_sequence((True, False, True))
    pad_view = MaskBuilder(PadPolicy()).view(snapshot, players, mask)
    neutral_view = MaskBuilder(NeutralPolicy()).view(snapshot, players, mask)
    print(MaskBuilder(PadPolicy()).render(pad_view).turns[0].messages[0].text)
    print(MaskBuilder(NeutralPolicy()).render(neutral_view).turns[0].messages[0].text)


def masking_plugins() -> None:
    from nlp_shap import ExplainConfig, PluginGroup, PluginRegistry
    from nlp_shap.plugins import register_builtin_plugins

    registry = PluginRegistry()
    register_builtin_plugins(registry)
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {"players": "tokens", "absence_policy": "pad"},
    })
    partitioner = registry.resolve(PluginGroup.PARTITIONS, config.explanation.players)
    policy = registry.resolve(
        PluginGroup.ABSENCE_POLICIES,
        config.explanation.absence_policy,
    )
    print(type(partitioner).__name__)
    print(type(policy).__name__)


def exact_quick_start() -> None:
    from itertools import product

    from nlp_shap.domain.coalition import CoalitionMask
    from nlp_shap.domain.players import PlayerSet
    from nlp_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
    from nlp_shap.estimation.exact import ExactEstimator

    player_set = PlayerSet(player_ids=("p0", "p1", "p2"))
    estimator = ExactEstimator()
    masks = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=False,
            seed=0,
        )
    )
    print("coalitions:", len(masks))
    all_masks = [
        tuple(bits) for bits in product([False, True], repeat=player_set.num_players)
    ]

    def majority_payoff(mask: tuple[bool, ...]) -> float:
        return 1.0 if sum(mask) >= 2 else 0.0

    payoffs = [majority_payoff(mask) for mask in all_masks]
    coalition_masks = tuple(CoalitionMask.from_sequence(mask) for mask in all_masks)
    shapley = estimator.estimate_attributions(
        coalition_masks,
        payoffs,
        ShapleyAggregator(),
    )
    banzhaf = estimator.estimate_attributions(
        coalition_masks,
        payoffs,
        BanzhafAggregator(),
    )
    print("Shapley:", shapley)
    print("Banzhaf:", banzhaf)


def exact_bitmask_iteration() -> None:
    from nlp_shap.estimation.exact import ExactEstimator

    previews: list[str] = []
    for index, mask_int in enumerate(ExactEstimator.iter_mask_ints(3)):
        present = ExactEstimator.mask_int_to_present(mask_int, 3)
        previews.append(str(present))
        if index >= 2:
            break
    print("\n".join(previews))


def exact_budget_requirement() -> None:
    from nlp_shap.domain.players import PlayerSet
    from nlp_shap.estimation.exact import ExactEstimator

    player_set = PlayerSet(player_ids=("p0",))
    try:
        ExactEstimator().sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=False,
            seed=0,
        )
    except ValueError as exc:
        print(exc)


def exact_plugin() -> None:
    from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins

    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.ESTIMATORS)
    estimator = registry.resolve(PluginGroup.ESTIMATORS, "exact")
    print(estimator.name)


def pipeline_quick_start() -> None:
    from nlp_shap import ExplainConfig, ExplainRunner
    from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
    from nlp_shap.domain.enums import Role

    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="hello world"),)),
    ))
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "estimator": "exact",
            "estimand": "shapley",
            "value_fn": "tfidf_cosine",
            "normalizer": "identity",
            "absence_policy": "pad",
        },
    })
    output = ExplainRunner(config).explain_sync(snapshot)
    print("values:", output.result.values)
    print("requested:", output.metrics.requested)
    print("executed:", output.metrics.executed)


def runtime_archive() -> None:
    from nlp_shap.domain.estimands import Estimand
    from nlp_shap.masking.codec import MaskCodec
    from nlp_shap.pipeline.manifest import RunManifest
    from nlp_shap.runtime.archive import CoalitionRecordDraft, RunArchive

    manifest = RunManifest(estimand=Estimand.SHAPLEY, run_id="demo-run")
    packed = MaskCodec.pack((True, False, True))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "demo-run"
        with RunArchive.open(root, manifest, flush_every=25) as archive:
            archive.append(
                CoalitionRecordDraft(
                    snapshot_id="snap-1",
                    coalition_key="coalition-1",
                    mask=packed,
                    absence_policy="delete",
                    model_id="mock",
                    generation_text="Who you?",
                    utility=0.8,
                    elapsed_ms=12.5,
                    cache_hit=False,
                )
            )
        with RunArchive.open(root, manifest) as archive:
            for record in archive.history_lazy():
                print(record.record_id, record.generation_text)


def runtime_dedup() -> None:
    from nlp_shap.pipeline.config import DedupConfig, GenerationConfig
    from nlp_shap.runtime.dedup import build_coalition_key, dedup_enabled

    generation = GenerationConfig(temperature=0.0)
    key = build_coalition_key(
        snapshot_id="snap-1",
        player_ids=("snap-1:0:0:0", "snap-1:0:0:1"),
        mask_present=(True, False),
        absence_policy="delete",
        model_id="mock",
        generation=generation,
    )
    print(dedup_enabled(DedupConfig(enabled="auto"), generation), key[:12])


def runtime_scheduler() -> None:
    from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
    from nlp_shap.domain.enums import Role
    from nlp_shap.masking.codec import MaskCodec
    from nlp_shap.pipeline.config import GenerationConfig
    from nlp_shap.runtime.dedup import CoalitionDedupRegistry
    from nlp_shap.runtime.scheduler import CoalitionJob, InferenceScheduler
    from nlp_shap.runtime.store import HotResultStore

    turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
    snapshot = ConversationSnapshot.from_turns((turn,))
    packed = MaskCodec.pack((True, False, True))

    async def generate(snapshot: ConversationSnapshot) -> str:
        return snapshot.turns[0].messages[0].text

    scheduler = InferenceScheduler(
        max_inflight=2,
        generation=GenerationConfig(temperature=0.0),
        store=HotResultStore(),
        dedup=CoalitionDedupRegistry(),
    )
    jobs = [
        CoalitionJob(
            coalition_key=f"key-{index % 3}",
            snapshot_id=snapshot.snapshot_id,
            snapshot=snapshot,
            absence_policy="delete",
            mask_words=packed.words,
            mask_n_bits=packed.n_bits,
            model_id="mock",
            utility=1.0,
        )
        for index in range(9)
    ]
    metrics = asyncio.run(scheduler.run(jobs, generate))
    print(metrics)


def value_tfidf() -> None:
    from nlp_shap.value.generation_records import GenerationRecord
    from nlp_shap.value.tfidf import TfIdfCosineValue

    base = GenerationRecord(
        text="hello world",
        text_token_rows=((1, 0), (0, 1)),
    )
    candidate = GenerationRecord(
        text="hello there",
        text_token_rows=((1, 0), (2, 3)),
    )
    value_fn = TfIdfCosineValue()
    value_fn.fit((base, candidate))
    print(round(value_fn.score(base, candidate), 4))


def value_embedding() -> None:
    from nlp_shap.domain.enums import EmbeddingMode
    from nlp_shap.value.embedding import CosineEmbeddingValue
    from nlp_shap.value.generation_records import GenerationRecord

    base = GenerationRecord(text="base", embedding=(1.0, 0.0))
    candidate = GenerationRecord(text="candidate", embedding=(0.9, 0.1))
    value_fn = CosineEmbeddingValue(embedding_mode=EmbeddingMode.STATIC)
    print(round(value_fn.score(base, candidate), 4))


def value_logprob() -> None:
    from nlp_shap.value.generation_records import GenerationRecord
    from nlp_shap.value.logprob import LogprobValue

    candidate = GenerationRecord(text="answer", logprobs=(-0.1, -0.2, -0.3))
    print(round(LogprobValue().score(candidate, candidate), 4))


def value_normalizer() -> None:
    from nlp_shap.estimation.normalizers import MinMaxNormalizer

    raw = [1.0, -2.0, 3.0]
    print(MinMaxNormalizer().normalize(raw))


def approximation_monte_carlo() -> None:
    from nlp_shap.domain.players import PlayerSet
    from nlp_shap.estimation.estimands import ShapleyAggregator
    from nlp_shap.estimation.monte_carlo import MonteCarloEstimator

    player_set = PlayerSet(player_ids=("p0", "p1", "p2", "p3"))
    estimator = MonteCarloEstimator()
    masks = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=True,
            seed=42,
        )
    )
    payoffs = [float(index % 3) for index in range(len(masks))]
    values = estimator.estimate_attributions(masks, payoffs, ShapleyAggregator())
    print("samples:", len(masks))
    print("values:", tuple(round(value, 4) for value in values))


def approximation_plugins() -> None:
    from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins

    registry = PluginRegistry()
    register_builtin_plugins(registry)
    mc = registry.resolve(PluginGroup.ESTIMATORS, "mc")
    complementary = registry.resolve(PluginGroup.ESTIMATORS, "complementary")
    neyman = registry.resolve(PluginGroup.ESTIMATORS, "neyman_cc")
    print(mc.name, complementary.name, neyman.name)


def config_roundtrip() -> None:
    from nlp_shap import explain_config_from_yaml, explain_config_to_yaml

    yaml_text = """
backend:
  kind: mock
  model_id: stub
explanation:
  estimator: exact
  estimand: shapley
""".strip()
    config = explain_config_from_yaml(yaml_text)
    roundtrip = explain_config_to_yaml(config).strip().splitlines()[:4]
    print("\n".join(roundtrip))


def visualization_render() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from nlp_shap import ExplainConfig, ExplainRunner, render_attribution
    from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
    from nlp_shap.domain.enums import Role
    from nlp_shap.masking.partitions import TokenPartitioner

    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="refund within thirty days"),)),
    ))
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "estimator": "exact",
            "estimand": "shapley",
            "value_fn": "tfidf_cosine",
            "absence_policy": "pad",
        },
    })
    output = ExplainRunner(config).explain_sync(snapshot)
    player_set = TokenPartitioner().partition(snapshot)
    bar_figure = render_attribution(
        output,
        snapshot,
        player_set,
        renderer="token_bar",
    )
    text_figure = render_attribution(
        output,
        snapshot,
        player_set,
        renderer="token_text",
    )
    _write_figure("visualization_bar", bar_figure)
    _write_figure("visualization_text", text_figure)
    print("saved attribution figures")


TEXT_SNIPPETS: dict[str, Callable[[], None]] = {
    "getting_started_majority": getting_started_majority,
    "getting_started_explain": getting_started_explain,
    "getting_started_manifest": getting_started_manifest,
    "estimands_majority": getting_started_majority,
    "estimands_coalition_weights": estimands_coalition_weights,
    "estimands_manifest": getting_started_manifest,
    "masking_partition": masking_partition,
    "masking_render_coalition": masking_render_coalition,
    "masking_absence_policies": masking_absence_policies,
    "masking_plugins": masking_plugins,
    "exact_quick_start": exact_quick_start,
    "exact_bitmask_iteration": exact_bitmask_iteration,
    "exact_budget_requirement": exact_budget_requirement,
    "exact_plugin": exact_plugin,
    "pipeline_quick_start": pipeline_quick_start,
    "runtime_archive": runtime_archive,
    "runtime_dedup": runtime_dedup,
    "runtime_scheduler": runtime_scheduler,
    "value_tfidf": value_tfidf,
    "value_embedding": value_embedding,
    "value_logprob": value_logprob,
    "value_normalizer": value_normalizer,
    "approximation_monte_carlo": approximation_monte_carlo,
    "approximation_plugins": approximation_plugins,
    "config_roundtrip": config_roundtrip,
    "visualization_render": visualization_render,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for snippet_id, runner in TEXT_SNIPPETS.items():
        text = _capture(runner)
        _write_text(snippet_id, text)
        print(f"wrote {snippet_id}")


if __name__ == "__main__":
    main()
