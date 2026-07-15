"""Horizontal bar chart renderer for token attributions."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ..domain.estimands import Estimand
from .colors import diverging_rgba

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class TokenBarRenderer:
    """Render attributions as a horizontal bar chart."""

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
        plt = _import_pyplot()
        ordering = sorted(
            range(len(labels)),
            key=lambda index: abs(values[index]),
            reverse=True,
        )
        ordered_labels = [labels[index] for index in ordering]
        ordered_values = [values[index] for index in ordering]
        vmax = max(abs(value) for value in ordered_values)
        colors = [diverging_rgba(value, vmax) for value in ordered_values]
        height = max(2.5, len(labels) * 0.45)
        fig, axis = cast(Any, plt).subplots(figsize=(8.0, height))
        axis.barh(ordered_labels, ordered_values, color=colors)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Attribution")
        axis.set_title(title or f"{estimand.value.title()} token attributions")
        fig.tight_layout()
        return cast("Figure", fig)


def _import_pyplot() -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = "matplotlib is required for the visualization extra"
        raise ImportError(msg) from exc
    return plt
