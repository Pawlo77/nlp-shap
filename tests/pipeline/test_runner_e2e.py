"""End-to-end explain pipeline tests with the mock backend."""

from pathlib import Path

import pytest

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.estimands import Estimand
from nlp_shap.estimation.exact import ExactEstimator
from nlp_shap.runtime import BASE_GENERATION_FILE, InMemoryObservabilitySink
from nlp_shap.runtime.archive import RunArchive


def _three_token_snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="a b c"),))
    return ConversationSnapshot.from_turns((turn,))


def _e2e_config(archive_path: str = "") -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "generation": {
            "max_new_tokens": 8,
            "temperature": 0.0,
            "precompute_base": True,
        },
        "explanation": {
            "estimator": "exact",
            "estimand": "shapley",
            "value_fn": "tfidf_cosine",
            "normalizer": "identity",
            "players": "tokens",
            "absence_policy": "pad",
            "budget": {"fraction": 1.0},
            "max_inflight": 2,
            "seed": 0,
            "archive": {"path": archive_path, "flush_every": 1},
        },
    })


def test_runner_e2e_mock_exact_shapley() -> None:
    """Mock E2E returns attributions and scheduler metrics."""
    output = ExplainRunner(_e2e_config()).explain_sync(_three_token_snapshot())

    assert output.result.estimand is Estimand.SHAPLEY
    assert len(output.result.values) == 3
    assert output.metrics.requested == ExactEstimator.num_coalitions(3)
    assert output.metrics.executed == output.metrics.requested
    assert output.run_id
    assert output.manifest.estimand is Estimand.SHAPLEY
    assert output.perf is not None
    assert output.perf.total_ms >= 0.0


def test_runner_e2e_mock_exact_with_delete_policy() -> None:
    """Exact SHAP completes under delete absence, including empty coalition."""
    payload = _e2e_config().model_dump(mode="python")
    payload["explanation"]["absence_policy"] = "delete"
    output = ExplainRunner(ExplainConfig.model_validate(payload)).explain_sync(
        _three_token_snapshot()
    )
    assert len(output.result.values) == 3
    assert output.metrics.requested == ExactEstimator.num_coalitions(3)
    assert output.metrics.executed == output.metrics.requested


def test_runner_e2e_exact_shapley_beyond_default_hot_store_size() -> None:
    """Exact SHAP must retain all coalition outputs when 2^n exceeds store LRU default.

    Regression: HotResultStore maxsize=128 dropped early coalitions for n>=8,
    causing ``missing generation for coalition`` during scoring.
    """
    turn = Turn(
        messages=(Message(role=Role.USER, text="a b c d e f g h"),),
    )
    snapshot = ConversationSnapshot.from_turns((turn,))
    output = ExplainRunner(_e2e_config()).explain_sync(snapshot)

    assert len(output.result.values) == 8
    assert output.metrics.requested == ExactEstimator.num_coalitions(8)
    assert output.metrics.executed == output.metrics.requested
    assert ExactEstimator.num_coalitions(8) > 128


def test_runner_archives_base_generation_before_coalitions(
    tmp_path: Path,
) -> None:
    """Grand-coalition text is persisted before coalition rows are appended."""
    archive_template = str(tmp_path / "{run_id}")
    telemetry = InMemoryObservabilitySink()
    output = ExplainRunner(
        _e2e_config(archive_path=archive_template),
        telemetry=telemetry,
    ).explain_sync(_three_token_snapshot())

    archive_root = tmp_path / output.run_id
    base_path = archive_root / BASE_GENERATION_FILE
    assert base_path.is_file()

    with RunArchive.load(archive_root) as archive:
        base_text = archive.read_base_generation()
        records = list(archive.history_lazy())

    assert base_text
    assert len(records) == output.metrics.requested
    span_names = [span.name for span in telemetry.spans]
    assert "orchestrator.generate_base" in span_names
    assert "orchestrator.schedule" in span_names
    assert "estimator.sample_masks" in span_names


def test_runner_reanalyze_rescores_without_backend_calls(tmp_path: Path) -> None:
    """Reanalyze swaps value functions with zero scheduler executions."""
    archive_template = str(tmp_path / "{run_id}")
    initial = ExplainRunner(_e2e_config(archive_path=archive_template)).explain_sync(
        _three_token_snapshot()
    )
    archive_root = tmp_path / initial.run_id

    rescored = ExplainRunner(
        _e2e_config(archive_path=archive_template).model_copy(
            update={
                "explanation": _e2e_config(
                    archive_path=archive_template
                ).explanation.model_copy(update={"value_fn": "logprob"})
            }
        )
    ).reanalyze(archive_root)

    assert rescored.metrics.executed == 0
    assert rescored.metrics.requested == initial.metrics.requested
    assert len(rescored.result.values) == len(initial.result.values)
    assert rescored.perf is not None
    assert rescored.perf.generation_ms == 0.0


def test_reanalyze_requires_base_generation(tmp_path: Path) -> None:
    """Reanalyze rejects archives that omit the base generation prelude."""
    from nlp_shap.domain.estimands import Estimand
    from nlp_shap.masking.codec import MaskCodec
    from nlp_shap.pipeline.manifest import RunManifest
    from nlp_shap.runtime.archive import CoalitionRecordDraft

    root = tmp_path / "incomplete"
    manifest = RunManifest(estimand=Estimand.SHAPLEY, run_id="incomplete")
    packed = MaskCodec.pack((True, False, True))
    with RunArchive.open(root, manifest) as archive:
        archive.append(
            CoalitionRecordDraft(
                snapshot_id="snap",
                coalition_key="key-0",
                mask=packed,
                absence_policy="pad",
                model_id="mock",
                generation_text="text",
                utility=1.0,
                elapsed_ms=1.0,
                cache_hit=False,
            )
        )

    with pytest.raises(ValueError, match="missing base generation"):
        ExplainRunner(_e2e_config()).reanalyze(root)
