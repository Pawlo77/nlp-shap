"""Tests for attribution renderer plugins."""

import pytest

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.estimands import Estimand
from nlp_shap.masking.partitions import TokenPartitioner
from nlp_shap.plugins import PluginGroup, PluginRegistry
from nlp_shap.viz import render_attribution, render_attribution_html
from nlp_shap.viz.colors import diverging_rgba
from nlp_shap.viz.token_bar import TokenBarRenderer
from nlp_shap.viz.token_text import TokenTextRenderer

pytest.importorskip("matplotlib")


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def _three_token_output() -> tuple[object, ConversationSnapshot, object]:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "generation": {
            "max_new_tokens": 4,
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
            "seed": 0,
        },
    })
    snapshot = _snapshot("a b c")
    output = ExplainRunner(config).explain_sync(snapshot)
    player_set = TokenPartitioner().partition(snapshot)
    return output, snapshot, player_set


def test_renderer_entry_points_resolve() -> None:
    """Packaging entry points expose token renderers."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.RENDERERS)
    assert registry.names(PluginGroup.RENDERERS) == ("token_bar", "token_text")


def test_token_text_renderer_assigns_distinct_signed_colors() -> None:
    """Positive and negative tokens map to different color channels."""
    positive = diverging_rgba(1.0, 1.0)
    negative = diverging_rgba(-1.0, 1.0)
    assert positive != negative


def test_render_attribution_text_figure_is_non_empty() -> None:
    """Toy exact run renders a token-text figure."""
    output, snapshot, player_set = _three_token_output()
    figure = render_attribution(output, snapshot, player_set, renderer="token_text")
    assert len(figure.axes) == 1
    assert figure.axes[0].texts


def test_token_bar_renderer_sorts_by_absolute_value() -> None:
    """Horizontal bars follow descending absolute attribution order."""
    renderer = TokenBarRenderer()
    figure = renderer.render(
        ("a", "b", "c"),
        (0.1, -0.9, 0.4),
        estimand=Estimand.SHAPLEY,
    )
    bar_container = figure.axes[0].containers[0]
    heights = [bar.get_width() for bar in bar_container]
    assert heights == sorted(heights, key=abs, reverse=True)


def test_render_attribution_html_includes_tokens() -> None:
    """HTML renderer emits colored spans for each token."""
    output, snapshot, player_set = _three_token_output()
    html = render_attribution_html(output, snapshot, player_set)
    assert "a" in html
    assert "background-color:rgba" in html


def test_token_text_renderer_rejects_length_mismatch() -> None:
    """Mismatched labels and values raise ValueError."""
    renderer = TokenTextRenderer()
    with pytest.raises(ValueError, match="same length"):
        renderer.render(("a",), (0.1, 0.2), estimand=Estimand.SHAPLEY)
