"""Horizontal bar chart renderer for token attributions."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ..domain.estimands import Estimand
from .colors import bar_color
from .style import new_attribution_figure, polish_bar_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class TokenBarRenderer:
    """Render attributions as a SHAP-style horizontal bar chart."""

    @property
    def name(self) -> str:
        """Return the registered renderer identifier."""
        return "token_bar"

    def render(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        estimand: Estimand,
        title: str | None = None,
    ) -> "Figure":
        """Build a horizontal bar chart sorted by absolute attribution."""
        if len(labels) != len(values):
            msg = "labels and values must have the same length"
            raise ValueError(msg)
        if not labels:
            msg = "cannot render an empty attribution"
            raise ValueError(msg)

        ordering = sorted(
            range(len(labels)),
            key=lambda index: abs(values[index]),
            reverse=True,
        )
        ordered_labels = [labels[index] for index in ordering]
        ordered_values = [values[index] for index in ordering]
        height = max(3.0, len(labels) * 0.55)
        fig, axis = new_attribution_figure(width=8.5, height=height)

        y_positions = list(range(len(ordered_labels)))
        bars = axis.barh(
            y_positions,
            ordered_values,
            height=0.68,
            color=[bar_color(value) for value in ordered_values],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.set_yticks(y_positions, ordered_labels)
        axis.set_xlabel("Attribution", color="#6B7280", fontsize=11)
        axis.set_title(
            title or f"{estimand.value.title()} token attributions",
            loc="left",
            fontsize=13,
            fontweight="semibold",
            color="#111827",
            pad=14,
        )
        polish_bar_axes(axis)
        _annotate_bars(axis, bars, ordered_values)
        _add_sign_legend(axis)
        fig.tight_layout()
        return cast("Figure", fig)


def _annotate_bars(axis: Any, bars: Any, values: Sequence[float]) -> None:
    x_span = max(abs(value) for value in values) or 1.0
    offset = x_span * 0.02
    for bar, value in zip(bars, values, strict=True):
        width = float(bar.get_width())
        label = f"{value:+.3f}"
        x_pos = width + offset if width >= 0 else width - offset
        ha = "left" if width >= 0 else "right"
        axis.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            fontsize=9,
            color="#4B5563",
        )


def _add_sign_legend(axis: Any) -> None:
    from matplotlib.patches import Patch

    legend = axis.legend(
        handles=[
            Patch(facecolor="#FF0D57", edgecolor="none", label="Positive"),
            Patch(facecolor="#1E88E5", edgecolor="none", label="Negative"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color("#6B7280")
