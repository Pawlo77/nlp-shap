"""Tests for shared approximate-estimator utilities."""

import numpy as np
import pytest

from nlp_shap.estimation._shared import (
    aggregate_complementary_shapley,
    build_c_matrix_from_pairs,
    compute_complementary_num_splits,
    compute_mc_num_samples,
    increment_m_counts,
    present_to_mask_int,
    random_present,
)


def test_compute_mc_num_samples_respects_budget_fraction() -> None:
    """Monte Carlo sample counts scale with budget fraction."""
    assert compute_mc_num_samples(4, 0.5, include_minimal_masks=False) == 7
    assert compute_mc_num_samples(4, 1.0, include_minimal_masks=False) == 15


def test_compute_complementary_num_splits_is_even() -> None:
    """Complementary budgets are even and respect the minimum pair count."""
    assert compute_complementary_num_splits(4, 0.5, include_minimal_masks=True) == 8
    assert compute_complementary_num_splits(4, 0.3, include_minimal_masks=True) % 2 == 0


def test_increment_m_counts_tracks_coalition_sizes() -> None:
    """M-matrix counts increment on player and coalition-size axes."""
    counts = np.zeros((2, 3), dtype=np.int64)
    increment_m_counts(counts, (True, False))
    assert counts[0, 1] == 1
    assert counts[1, 0] == 0


def test_build_c_matrix_from_pairs_requires_complements() -> None:
    """Non-complementary pairs are rejected when building matrix C."""
    from nlp_shap.domain.coalition import CoalitionMask

    counts = np.zeros((2, 3), dtype=np.int64)
    masks = (
        CoalitionMask.from_sequence((True, False)),
        CoalitionMask.from_sequence((True, False)),
    )
    with pytest.raises(ValueError, match="complementary pairs"):
        _ = build_c_matrix_from_pairs(counts, masks, [1.0, 2.0])


def test_aggregate_complementary_shapley_matches_manual_ratio() -> None:
    """Complementary aggregation matches manual C/M ratios."""
    m_counts = np.array(
        [
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    c_matrix = np.array(
        [
            [0.0, 4.0, 3.0],
            [0.0, 2.0, 6.0],
        ],
        dtype=np.float64,
    )
    values = aggregate_complementary_shapley(m_counts, c_matrix)
    m_work = m_counts[:, 1:]
    c_work = c_matrix[:, 1:]
    expected = []
    for row in range(2):
        ratio = np.zeros_like(m_work[row])
        positive = m_work[row] > 0
        ratio[positive] = c_work[row, positive] / m_work[row, positive]
        expected.append(float(ratio.sum() / 2))
    assert values == pytest.approx(expected)


def test_random_present_honors_include_index() -> None:
    """Random coalition generation can force one player to be present."""
    rng = np.random.default_rng(0)
    present = random_present(rng, 4, true_count=2, include_index=1)
    assert present[1] is True
    assert sum(present) == 2


def test_present_to_mask_int_round_trip() -> None:
    """Coalition bitmasks round-trip through present vectors."""
    present = (True, False, True)
    assert present_to_mask_int(present) == 5
