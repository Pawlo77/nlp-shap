"""Tests for optional coalition progress on ExplainRunner."""

from dataclasses import dataclass, field

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role


@dataclass
class RecordingProgress:
    """Capture progress callbacks for assertions."""

    planned: list[int] = field(default_factory=list)
    finished: list[tuple[int, int]] = field(default_factory=list)

    def on_coalitions_planned(self, total: int) -> None:
        self.planned.append(total)

    def on_coalition_finished(self, done: int, total: int) -> None:
        self.finished.append((done, total))


def _snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="a b c"),))
    return ConversationSnapshot.from_turns((turn,))


def _config() -> ExplainConfig:
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
        },
    })


def test_explain_runner_reports_coalition_progress() -> None:
    progress = RecordingProgress()
    output = ExplainRunner(_config(), progress=progress).explain_sync(_snapshot())
    assert progress.planned == [output.metrics.requested]
    assert progress.finished[-1] == (
        output.metrics.requested,
        output.metrics.requested,
    )
    assert len(progress.finished) == output.metrics.requested
