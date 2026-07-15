"""SHAP-style inline colored token attribution renderer."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ..domain.estimands import Estimand
from .colors import diverging_rgba

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
        """Build a matplotlib figure with colored token labels."""
        if len(labels) != len(values):
            msg = "labels and values must have the same length"
            raise ValueError(msg)
        if not labels:
            msg = "cannot render an empty attribution"
            raise ValueError(msg)
        plt = _import_pyplot()
        vmax = max(abs(value) for value in values)
        fig, axis = cast(Any, plt).subplots(figsize=(max(6.0, len(labels) * 0.55), 1.8))
        axis.set_axis_off()
        header = title or f"{estimand.value.title()} token attributions"
        axis.set_title(header)
        if len(labels) == 1:
            positions = [0.5]
        else:
            positions = [index / (len(labels) - 1) for index in range(len(labels))]
        for position, label, value in zip(positions, labels, values, strict=True):
            rgba = diverging_rgba(value, vmax)
            axis.text(
                position,
                0.5,
                label,
                transform=axis.transAxes,
                fontsize=12,
                va="center",
                ha="center",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": rgba,
                    "edgecolor": "none",
                },
            )
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
            red, green, blue, alpha = diverging_rgba(value, vmax)
            color = _rgba_css(red, green, blue, alpha)
            spans.append(
                f'<span style="background-color:{color};'
                f'padding:2px 4px;margin:1px;border-radius:4px;">'
                f"{_escape_html(label)}</span>"
            )
        body = " ".join(spans)
        return f"<div><strong>{_escape_html(header)}</strong><br/>{body}</div>"


def _rgba_css(red: float, green: float, blue: float, alpha: float) -> str:
    return f"rgba({int(red * 255)},{int(green * 255)},{int(blue * 255)},{alpha:.2f})"


def _escape_html(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _import_pyplot() -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = "matplotlib is required for the visualization extra"
        raise ImportError(msg) from exc
    return plt
