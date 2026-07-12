"""Tests for coalition weight formulas used by estimand aggregators."""

import math

import pytest

from nlp_shap.estimation.estimands.banzhaf import BanzhafAggregator
from nlp_shap.estimation.estimands.shapley import ShapleyAggregator


@pytest.mark.parametrize(
    ("coalition_size", "num_players", "expected"),
    [
        (0, 3, math.factorial(0) * math.factorial(2) / math.factorial(3)),
        (1, 3, math.factorial(1) * math.factorial(1) / math.factorial(3)),
        (2, 3, math.factorial(2) * math.factorial(0) / math.factorial(3)),
    ],
)
def test_shapley_coalition_weight(
    coalition_size: int,
    num_players: int,
    expected: float,
) -> None:
    """Shapley weight matches k!(n-k-1)!/n!."""
    aggregator = ShapleyAggregator()
    assert aggregator.coalition_weight(coalition_size, num_players) == pytest.approx(
        expected
    )


def test_banzhaf_coalition_weight_is_uniform() -> None:
    """Banzhaf assigns the same weight to every coalition size."""
    aggregator = BanzhafAggregator()
    num_players = 4
    expected = 1.0 / (2 ** (num_players - 1))
    for coalition_size in range(num_players):
        assert aggregator.coalition_weight(coalition_size, num_players) == pytest.approx(
            expected
        )
