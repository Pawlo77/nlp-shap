"""Diverging color scale for signed attribution values."""

from typing import Any, cast


def diverging_rgba(value: float, vmax: float) -> tuple[float, float, float, float]:
    """Map a signed attribution to an RGBA tuple (red positive, blue negative)."""
    matplotlib = cast(Any, _import_matplotlib())
    if vmax <= 0.0:
        return (0.92, 0.92, 0.92, 1.0)
    normalized = max(-1.0, min(1.0, value / vmax))
    cmap = matplotlib.colormaps["RdBu_r"]
    rgba = cmap((normalized + 1.0) / 2.0)
    channels = tuple(float(channel) for channel in rgba)
    return cast(tuple[float, float, float, float], channels)


def _import_matplotlib() -> object:
    try:
        import matplotlib
    except ImportError as exc:
        msg = "matplotlib is required for the visualization extra"
        raise ImportError(msg) from exc
    return matplotlib
