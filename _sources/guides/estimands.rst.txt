Using estimand aggregators
==========================

Phase ``0.1.1`` exposes standalone **estimand aggregators**. You provide coalition
masks and payoffs :math:`v(S)`; the library returns per-player attributions and
records which estimand was used.

Quick start
-----------

.. code-block:: python

   from itertools import product

   from nlp_shap import BanzhafAggregator, Estimand, ShapleyAggregator

   num_players = 3
   masks = [tuple(bits) for bits in product([False, True], repeat=num_players)]

   def majority_payoff(mask: tuple[bool, ...]) -> float:
       return 1.0 if sum(mask) >= 2 else 0.0

   payoffs = [majority_payoff(mask) for mask in masks]

   shapley = ShapleyAggregator().aggregate(masks, payoffs)
   banzhaf = BanzhafAggregator().aggregate(masks, payoffs)

   print("Shapley:", shapley)   # ~[0.333, 0.333, 0.333]
   print("Banzhaf:", banzhaf)   # [0.5, 0.5, 0.5]

Coalition weights
-----------------

Inspect weights without full aggregation:

.. code-block:: python

   from nlp_shap import ShapleyAggregator, BanzhafAggregator

   shapley = ShapleyAggregator()
   banzhaf = BanzhafAggregator()

   n = 4
   for k in range(n):
       print(f"k={k}", shapley.coalition_weight(k, n), banzhaf.coalition_weight(k, n))

Explain results and manifests
-----------------------------

Label outputs explicitly for archives and papers:

.. code-block:: python

   from nlp_shap import Estimand, ExplainResult, RunManifest, parse_manifest

   result = ExplainResult(
       estimand=Estimand.SHAPLEY,
       values=(0.333, 0.333, 0.333),
   )

   manifest = RunManifest(estimand=result.estimand, run_id="run-42")
   payload = manifest.to_dict()
   restored = parse_manifest(payload)

   assert restored.estimand is Estimand.SHAPLEY

Wire values for JSON manifests use :data:`~nlp_shap.domain.estimands.EstimandWire`
(``"shapley"`` | ``"banzhaf"``).

Notebook
--------

See the runnable walkthrough in :doc:`../examples` —
``examples/estimands_toy_game.ipynb``.

Further reading
---------------

- Theory: :doc:`../theory/estimands`
- Cooperative games: :doc:`../theory/cooperative_games`
- API: :doc:`../api`
