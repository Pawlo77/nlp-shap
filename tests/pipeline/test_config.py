"""Tests for ExplainConfig YAML round-trip."""

from pathlib import Path

from nlp_shap.domain.enums import EmbeddingMode
from nlp_shap.domain.estimands import Estimand
from nlp_shap.pipeline.config import (
    explain_config_from_yaml,
    explain_config_to_yaml,
)

FIXTURE = Path(__file__).parent / "fixtures" / "explain_config.yaml"


def test_explain_config_yaml_round_trip_matches_fixture() -> None:
    """YAML load and dump preserve the Appendix A skeleton."""
    original_text = FIXTURE.read_text(encoding="utf-8")
    config = explain_config_from_yaml(original_text)
    restored = explain_config_from_yaml(explain_config_to_yaml(config))
    assert restored == config


def test_explain_config_parses_appendix_a_fields() -> None:
    """Nested sections deserialize to typed models."""
    config = explain_config_from_yaml(FIXTURE.read_text(encoding="utf-8"))
    assert config.backend.kind == "lmstudio"
    assert config.backend.model_id == "qwen2-500m-instruct"
    assert config.generation.max_new_tokens == 128
    assert config.explanation.estimand is Estimand.SHAPLEY
    assert config.explanation.budget.fraction == 0.3
    assert config.explanation.include_minimal_masks is False
    assert config.explanation.embedding_mode is EmbeddingMode.STATIC
    assert config.explanation.seed == 42


def test_explain_config_rejects_unknown_fields() -> None:
    """Extra YAML keys are rejected to keep the schema strict."""
    text = FIXTURE.read_text(encoding="utf-8") + "\nunknown_root: true\n"
    try:
        explain_config_from_yaml(text)
    except Exception as error:
        assert "unknown_root" in str(error)
    else:
        raise AssertionError("expected validation error")
