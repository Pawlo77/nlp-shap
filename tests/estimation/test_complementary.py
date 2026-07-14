"""Tests for complementary pair sampling."""

import numpy as np
import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation._shared import (
    aggregate_complementary_shapley,
    build_c_matrix_from_pairs,
)
from nlp_shap.estimation.complementary import ComplementaryEstimator


def test_complementary_pairs_are_negations() -> None:
    """Each emitted pair consists of a mask and its bitwise complement."""
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    masks = list(
        ComplementaryEstimator().sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=True,
            seed=1,
        )
    )
    assert len(masks) % 2 == 0
    for index in range(0, len(masks), 2):
        left = masks[index].present
        right = masks[index + 1].present
        assert tuple(not value for value in left) == right


def test_complementary_budget_is_even_and_capped() -> None:
    """Complementary sampling respects the even budget cap."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    masks = list(
        ComplementaryEstimator().sample_masks(
            player_set,
            budget_fraction=0.4,
            include_minimal_masks=True,
            seed=2,
        )
    )
    assert len(masks) % 2 == 0
    assert len(masks) <= 14


def test_complementary_estimate_matches_manual_computation() -> None:
    """Complementary aggregation matches manual C/M ratios on toy data."""
    estimator = ComplementaryEstimator()
    estimator.reset_sampling_state(2)
    assert estimator._m_counts is not None
    estimator._m_counts[:] = np.array(
        [
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.int64,
    )
    masks = (
        CoalitionMask.from_sequence((True, True)),
        CoalitionMask.from_sequence((False, False)),
        CoalitionMask.from_sequence((True, False)),
        CoalitionMask.from_sequence((False, True)),
    )
    payoffs = [1.0, 0.0, 2.0, 1.0]
    values = estimator.estimate_attributions(masks, payoffs)
    c_matrix = build_c_matrix_from_pairs(estimator._m_counts, masks, payoffs)
    expected = aggregate_complementary_shapley(estimator._m_counts, c_matrix)
    assert values == pytest.approx(expected)


def test_complementary_rejects_odd_mask_count() -> None:
    """Odd mask lists cannot be aggregated as complementary pairs."""
    estimator = ComplementaryEstimator()
    estimator.reset_sampling_state(2)
    masks = (CoalitionMask.from_sequence((True, False)),)
    with pytest.raises(ValueError, match="complementary pairs"):
        _ = estimator.estimate_attributions(masks, [1.0])


def test_complementary_estimator_exposes_name() -> None:
    """Estimator reports the registered plugin identifier."""
    assert ComplementaryEstimator().name == "complementary"
