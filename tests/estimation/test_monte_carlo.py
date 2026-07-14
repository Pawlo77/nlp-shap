"""Tests for Monte Carlo coalition sampling."""

from itertools import product

import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation.estimands.banzhaf import BanzhafAggregator
from nlp_shap.estimation.estimands.shapley import ShapleyAggregator
from nlp_shap.estimation.exact import ExactEstimator
from nlp_shap.estimation.monte_carlo import MonteCarloEstimator


def _majority_payoffs(masks: tuple[tuple[bool, ...], ...]) -> list[float]:
    return [1.0 if sum(mask) >= 2 else 0.0 for mask in masks]


def test_mc_budget_cap_is_honoured() -> None:
    """Monte Carlo sampling never exceeds the configured budget."""
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    masks = list(
        MonteCarloEstimator().sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=False,
            seed=7,
        )
    )
    assert len(masks) <= 3


def test_mc_shapley_differs_from_banzhaf_same_seed() -> None:
    """Monte Carlo attributions depend on the selected estimand plugin."""
    player_set = PlayerSet(player_ids=("p0", "p1", "p2"))
    estimator = MonteCarloEstimator()
    masks = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=True,
            seed=11,
        )
    )
    payoffs = _majority_payoffs(tuple(mask.present for mask in masks))
    shapley = estimator.estimate_attributions(
        masks,
        payoffs,
        ShapleyAggregator(),
    )
    banzhaf = estimator.estimate_attributions(
        masks,
        payoffs,
        BanzhafAggregator(),
    )
    assert shapley != banzhaf


def test_mc_reproducible_with_seed() -> None:
    """Identical seeds produce identical mask samples."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    kwargs = {
        "player_set": player_set,
        "budget_fraction": 0.4,
        "include_minimal_masks": False,
        "seed": 99,
    }
    first = [mask.present for mask in MonteCarloEstimator().sample_masks(**kwargs)]
    second = [mask.present for mask in MonteCarloEstimator().sample_masks(**kwargs)]
    assert first == second


def test_mc_converges_toward_exact_on_additive_game() -> None:
    """Large MC budgets approach exact Shapley values on additive games."""
    num_players = 3
    coefficients = (1.0, 2.0, 3.0)
    player_set = PlayerSet(
        player_ids=tuple(f"p{index}" for index in range(num_players))
    )
    all_masks = tuple(
        CoalitionMask.from_sequence(mask)
        for mask in product([False, True], repeat=num_players)
        if not all(mask)
    )
    all_payoffs = [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in all_masks
    ]
    exact = ExactEstimator().estimate_attributions(
        all_masks,
        all_payoffs,
        ShapleyAggregator(),
    )
    mc_masks = list(
        MonteCarloEstimator().sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=True,
            seed=3,
        )
    )
    mc_payoffs = [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in mc_masks
    ]
    estimated = MonteCarloEstimator().estimate_attributions(
        mc_masks,
        mc_payoffs,
        ShapleyAggregator(),
    )
    assert estimated == pytest.approx(exact, rel=1e-6, abs=1e-6)


def test_mc_estimator_exposes_name() -> None:
    """Estimator reports the registered plugin identifier."""
    assert MonteCarloEstimator().name == "mc"
