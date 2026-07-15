"""Optional GPU integration tests for the transformers backend."""

from pathlib import Path

import pytest

from nlp_shap import ExplainRunner, explain_config_from_yaml
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def transformers_backend_available() -> None:
    """Skip GPU tests when torch/transformers are unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


def _three_token_snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="one two three"),))
    return ConversationSnapshot.from_turns((turn,))


def test_transformers_runner_exact_reports_kv_cache_hits(
    transformers_backend_available: None,
) -> None:
    """A tiny exact run against HF reports non-zero prefix-cache hits."""
    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "dev_transformers.yaml"
    )
    output = ExplainRunner(
        explain_config_from_yaml(config_path.read_text(encoding="utf-8"))
    ).explain_sync(_three_token_snapshot())

    assert len(output.result.values) == 3
    assert output.metrics.executed > 0
    assert output.metrics.kv_cache_hits > 0
