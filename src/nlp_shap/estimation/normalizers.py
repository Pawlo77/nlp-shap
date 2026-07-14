"""Post-aggregation attribution normalizers."""

from collections.abc import Sequence

import numpy as np


class IdentityNormalizer:
    """Return attributions unchanged."""

    @property
    def name(self) -> str:
        """Return the registered normalizer identifier."""
        return "identity"

    def normalize(self, values: Sequence[float]) -> list[float]:
        """Return a copy of ``values``."""
        return list(values)


class AbsSumNormalizer:
    """Scale attributions by the sum of absolute values."""

    @property
    def name(self) -> str:
        """Return the registered normalizer identifier."""
        return "abs_sum"

    def normalize(self, values: Sequence[float]) -> list[float]:
        """Return ``values`` divided by the sum of absolute entries."""
        array = np.asarray(values, dtype=np.float64)
        abs_sum = float(np.abs(array).sum())
        if abs_sum == 0.0:
            return list(values)
        normalized = array / abs_sum
        return [float(value) for value in normalized]


class PowerShiftNormalizer:
    """Shift to non-negative values, apply a power, then scale to sum one."""

    def __init__(self, power: float = 1.0) -> None:
        if power <= 0.0:
            msg = "power must be a positive float"
            raise ValueError(msg)
        self.power = power

    @property
    def name(self) -> str:
        """Return the registered normalizer identifier."""
        return "power_shift"

    def normalize(self, values: Sequence[float]) -> list[float]:
        """Return power-shift normalized attributions."""
        array = np.asarray(values, dtype=np.float64)
        shifted = array - array.min()
        powered = np.power(shifted, self.power)
        total = float(powered.sum())
        if total == 0.0:
            return list(values)
        normalized = powered / total
        return [float(value) for value in normalized]


class MinMaxNormalizer:
    """Min-max scale to ``[0, 1]`` then normalize to sum one."""

    @property
    def name(self) -> str:
        """Return the registered normalizer identifier."""
        return "min_max"

    def normalize(self, values: Sequence[float]) -> list[float]:
        """Return min-max scaled attributions that sum to one."""
        array = np.asarray(values, dtype=np.float64)
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val - min_val == 0.0:
            uniform = np.ones_like(array) / len(array)
            return [float(value) for value in uniform]
        scaled = (array - min_val) / (max_val - min_val)
        total = float(scaled.sum())
        normalized = scaled / total
        return [float(value) for value in normalized]
