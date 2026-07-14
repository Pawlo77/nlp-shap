"""End-to-end explain pipeline tests with the mock backend."""

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.estimands import Estimand
from nlp_shap.estimation.exact import ExactEstimator


def _three_token_snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="a b c"),))
    return ConversationSnapshot.from_turns((turn,))


def _e2e_config() -> ExplainConfig:
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


def test_runner_e2e_mock_exact_shapley() -> None:
    """Mock E2E returns attributions and scheduler metrics."""
    output = ExplainRunner(_e2e_config()).explain_sync(_three_token_snapshot())

    assert output.result.estimand is Estimand.SHAPLEY
    assert len(output.result.values) == 3
    assert output.metrics.requested == ExactEstimator.num_coalitions(3)
    assert output.metrics.executed == output.metrics.requested
    assert output.run_id
    assert output.manifest.estimand is Estimand.SHAPLEY
