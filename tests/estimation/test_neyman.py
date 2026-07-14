"""Tests for Neyman complementary allocation."""

import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation.neyman import NeymanEstimator


def _additive_payoffs(
    masks: list[CoalitionMask],
    coefficients: tuple[float, ...],
) -> list[float]:
    return [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in masks
    ]


def test_neyman_initial_sampling_is_reproducible() -> None:
    """Phase-one Neyman masks are reproducible at a fixed seed."""
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    kwargs = {
        "player_set": player_set,
        "budget_fraction": 0.8,
        "include_minimal_masks": False,
        "seed": 21,
    }
    first = [mask.present for mask in NeymanEstimator().sample_masks(**kwargs)]
    second = [mask.present for mask in NeymanEstimator().sample_masks(**kwargs)]
    assert first == second
    assert len(first) % 2 == 0


def test_neyman_two_phase_sampling_respects_budget() -> None:
    """Initial and allocation phases together stay within the total budget."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    estimator = NeymanEstimator(initial_fraction=0.25, use_standard_method=True)
    phase_one = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=0.6,
            include_minimal_masks=False,
            seed=5,
        )
    )
    payoffs = _additive_payoffs(phase_one, (1.0, 2.0, 3.0, 4.0))
    estimator.begin_allocation(phase_one, payoffs)
    phase_two = list(estimator.sample_allocation_masks())
    assert len(phase_one) + len(phase_two) <= 14


def test_neyman_rejects_minimal_masks_flag() -> None:
    """Neyman sampling rejects include_minimal_masks=True."""
    player_set = PlayerSet(player_ids=("a", "b"))
    with pytest.raises(ValueError, match="include_minimal_masks"):
        _ = list(
            NeymanEstimator().sample_masks(
                player_set,
                budget_fraction=0.5,
                include_minimal_masks=True,
                seed=0,
            )
        )


def test_neyman_allocation_requires_begin_allocation() -> None:
    """Allocation masks cannot be sampled before begin_allocation."""
    estimator = NeymanEstimator()
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    _ = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=False,
            seed=1,
        )
    )
    with pytest.raises(RuntimeError, match="allocation has not been initialized"):
        _ = list(estimator.sample_allocation_masks())


def test_neyman_estimator_exposes_name() -> None:
    """Estimator reports the registered plugin identifier."""
    assert NeymanEstimator().name == "neyman_cc"
