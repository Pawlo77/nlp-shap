Using the exact estimator
=========================

:class:`~nlp_shap.estimation.exact.ExactEstimator` enumerates every coalition
mask (except the grand coalition) and delegates attribution to a registered
estimand plugin.

Quick start
-----------

.. code-block:: python

   from itertools import product

   from nlp_shap.domain.coalition import CoalitionMask
   from nlp_shap.domain.players import PlayerSet
   from nlp_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
   from nlp_shap.estimation.exact import ExactEstimator

   player_set = PlayerSet(player_ids=("p0", "p1", "p2"))
   estimator = ExactEstimator()

   masks = estimator.sample_masks(
       player_set,
       budget_fraction=1.0,
       include_minimal_masks=False,
       seed=0,
   )
   print("coalitions:", len(masks))  # 7 for three players

   all_masks = [
       tuple(bits) for bits in product([False, True], repeat=player_set.num_players)
   ]

   def majority_payoff(mask: tuple[bool, ...]) -> float:
       return 1.0 if sum(mask) >= 2 else 0.0

   payoffs = [majority_payoff(mask) for mask in all_masks]
   coalition_masks = tuple(CoalitionMask.from_sequence(mask) for mask in all_masks)

   shapley = estimator.estimate_attributions(
       coalition_masks,
       payoffs,
       ShapleyAggregator(),
   )
   banzhaf = estimator.estimate_attributions(
       coalition_masks,
       payoffs,
       BanzhafAggregator(),
   )

   print("Shapley:", shapley)
   print("Banzhaf:", banzhaf)

Budget requirement
------------------

Exact enumeration requires a full budget fraction:

.. code-block:: python

   from nlp_shap.domain.players import PlayerSet
   from nlp_shap.estimation.exact import ExactEstimator

   player_set = PlayerSet(player_ids=("p0",))
   try:
       ExactEstimator().sample_masks(
           player_set,
           budget_fraction=0.5,
           include_minimal_masks=False,
           seed=0,
       )
   except ValueError as exc:
       print(exc)  # exact estimator requires budget_fraction == 1.0

Plugin resolution
-----------------

Resolve the estimator from packaging entry points:

.. code-block:: python

   from nlp_shap.plugins import PluginGroup, PluginRegistry, register_builtin_plugins

   registry = PluginRegistry()
   register_builtin_plugins(registry)
   registry.load_entry_points(PluginGroup.ESTIMATORS)

   estimator = registry.resolve(PluginGroup.ESTIMATORS, "exact")
   print(estimator.name)  # exact

Notebook
--------

See the runnable walkthrough in :doc:`../examples` —
``examples/exact_estimation.ipynb``.

Further reading
---------------

- :doc:`estimands` — Shapley and Banzhaf aggregators
- :doc:`config` — ``explanation.estimator`` key
- Theory: :doc:`../theory/exact`
- API: :doc:`../api/estimation`
