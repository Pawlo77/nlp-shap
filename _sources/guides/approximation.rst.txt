Using approximate estimators
============================

Approximate estimators sample coalitions under a budget fraction and aggregate
payoffs into attributions. Monte Carlo delegates to estimand plugins; complementary
and Neyman estimators use complementary-contribution (CC) aggregation.

Monte Carlo
-----------

.. code-block:: python

   from nlp_shap.domain.players import PlayerSet
   from nlp_shap.estimation.estimands import ShapleyAggregator
   from nlp_shap.estimation.monte_carlo import MonteCarloEstimator

   player_set = PlayerSet(player_ids=("p0", "p1", "p2", "p3"))
   estimator = MonteCarloEstimator()

   masks = list(estimator.sample_masks(
       player_set,
       budget_fraction=0.5,
       include_minimal_masks=True,
       seed=42,
   ))
   payoffs = [...]  # one float per mask from your value function

   values = estimator.estimate_attributions(
       masks,
       payoffs,
       ShapleyAggregator(),
   )

Complementary pairs
-------------------

.. code-block:: python

   from nlp_shap.estimation.complementary import ComplementaryEstimator

   estimator = ComplementaryEstimator()
   masks = list(estimator.sample_masks(
       player_set,
       budget_fraction=0.4,
       include_minimal_masks=True,
       seed=7,
   ))
   # masks arrive in complementary pairs: S, N\\S, S, N\\S, ...
   payoffs = [...]
   values = estimator.estimate_attributions(masks, payoffs)

Neyman two-phase sampling
-------------------------

Neyman allocation needs phase-one payoffs before drawing phase-two masks:

.. code-block:: python

   from nlp_shap.estimation.neyman import NeymanEstimator

   estimator = NeymanEstimator(
       initial_fraction=0.25,
       use_standard_method=False,
   )
   phase_one = list(estimator.sample_masks(
       player_set,
       budget_fraction=0.6,
       include_minimal_masks=False,
       seed=11,
   ))
   payoffs_one = [...]
   estimator.begin_allocation(phase_one, payoffs_one)

   phase_two = list(estimator.sample_allocation_masks())
   payoffs_two = [...]

   all_masks = phase_one + phase_two
   all_payoffs = payoffs_one + payoffs_two
   values = estimator.estimate_attributions(all_masks, all_payoffs)

Configuration
-------------

Set the estimator and budget in :class:`~nlp_shap.pipeline.config.ExplainConfig`:

.. code-block:: yaml

   explanation:
     estimator: neyman_cc
     budget:
       fraction: 0.6
     include_minimal_masks: false
     neyman:
       initial_fraction: 0.25
       use_standard_method: false

Plugin resolution
-----------------

.. code-block:: python

   from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins

   registry = PluginRegistry()
   register_builtin_plugins(registry)

   mc = registry.resolve(PluginGroup.ESTIMATORS, "mc")
   complementary = registry.resolve(PluginGroup.ESTIMATORS, "complementary")
   neyman = registry.resolve(PluginGroup.ESTIMATORS, "neyman_cc")

Entry points are registered under ``nlp_shap.estimators`` in ``pyproject.toml``.

Choosing an estimator
---------------------

.. list-table::
   :header-rows: 1

   * - Estimator
     - Best when
     - Estimand support
   * - Monte Carlo
     - You need Banzhaf or Shapley from the same samples
     - Shapley, Banzhaf via plugins
   * - Complementary
     - Budget is moderate; CC variance reduction helps
     - CC Shapley only
   * - Neyman
     - Longer inputs; allocation targets high-variance coalition sizes
     - CC Shapley only (two-phase)

See :doc:`../examples` for a notebook comparing accuracy and cost across methods.
