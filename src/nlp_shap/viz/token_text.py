"""SHAP-style inline colored token attribution renderer."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ..domain.estimands import Estimand
from .colors import token_background_css, token_background_rgba
from .style import new_attribution_figure, polish_text_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class TokenTextRenderer:
    """Render attributions as inline colored tokens."""

    @property
    def name(self) -> str:
        """Return the registered renderer identifier."""
        return "token_text"

    def render(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        estimand: Estimand,
        title: str | None = None,
    ) -> "Figure":
        """Build a matplotlib figure with SHAP-style colored token labels."""
        if len(labels) != len(values):
            msg = "labels and values must have the same length"
            raise ValueError(msg)
        if not labels:
            msg = "cannot render an empty attribution"
            raise ValueError(msg)

        vmax = max(abs(value) for value in values)
        header = title or f"{estimand.value.title()} token attributions"
        width = max(7.5, min(14.0, 1.1 + sum(len(label) for label in labels) * 0.11))
        fig, axis = new_attribution_figure(width=width, height=2.4)
        polish_text_axes(axis)
        axis.set_title(
            header,
            loc="left",
            fontsize=13,
            fontweight="semibold",
            color="#111827",
            pad=12,
        )
        cursor_x = 0.02
        baseline_y = 0.42
        for label, value in zip(labels, values, strict=True):
            red, green, blue, alpha = token_background_rgba(value, vmax)
            token_width = max(0.06, len(label) * 0.013)
            axis.text(
                cursor_x,
                baseline_y,
                label,
                transform=axis.transAxes,
                fontsize=12,
                va="center",
                ha="left",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.35,rounding_size=0.08",
                    "facecolor": (red, green, blue, alpha),
                    "edgecolor": "none",
                },
            )
            cursor_x += token_width
        _draw_color_legend(axis)
        fig.tight_layout()
        return cast("Figure", fig)

    def to_html(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        estimand: Estimand,
        title: str | None = None,
    ) -> str:
        """Return an HTML fragment with colored token spans for Jupyter."""
        if len(labels) != len(values):
            msg = "labels and values must have the same length"
            raise ValueError(msg)
        vmax = max(abs(value) for value in values)
        header = title or f"{estimand.value.title()} token attributions"
        spans: list[str] = []
        for label, value in zip(labels, values, strict=True):
            background = token_background_css(value, vmax)
            value_label = f"{value:+.3f}"
            spans.append(
                '<span class="nlp-shap-token" '
                f'title="{estimand.value}: {value_label}" '
                f'style="background-color:{background};">'
                f"{_escape_html(label)}</span>"
            )
        body = "".join(spans)
        return (
            '<div class="nlp-shap-attribution" '
            'style="font-family:system-ui,-apple-system,Segoe UI,Roboto,'
            'Helvetica Neue,Arial,sans-serif;line-height:1.8;color:#111827;">'
            f'<div style="font-size:14px;font-weight:600;margin-bottom:8px;">'
            f"{_escape_html(header)}</div>"
            '<div style="font-size:12px;color:#6B7280;margin-bottom:10px;">'
            '<span style="display:inline-block;width:10px;height:10px;'
            'border-radius:2px;background:#FF0D57;margin-right:4px;"></span>'
            "increases&nbsp;&nbsp;"
            '<span style="display:inline-block;width:10px;height:10px;'
            'border-radius:2px;background:#1E88E5;margin-right:4px;"></span>'
            "decreases</div>"
            f'<div style="font-size:15px;word-break:break-word;">{body}</div>'
            "</div>"
            "<style>"
            ".nlp-shap-token{display:inline-block;padding:3px 7px;margin:2px 3px 2px 0;"
            "border-radius:4px;transition:box-shadow .15s ease;}"
            ".nlp-shap-token:hover{box-shadow:0 0 0 1px rgba(17,24,39,.12);}"
            "</style>"
        )


def _draw_color_legend(axis: Any) -> None:
    axis.text(
        0.02,
        0.08,
        "● increases",
        transform=axis.transAxes,
        fontsize=9,
        color="#FF0D57",
        ha="left",
        va="center",
    )
    axis.text(
        0.16,
        0.08,
        "● decreases",
        transform=axis.transAxes,
        fontsize=9,
        color="#1E88E5",
        ha="left",
        va="center",
    )


def _escape_html(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
