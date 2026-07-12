Getting started
===============

This page gets you from zero to a working Shapley vs Banzhaf comparison in a
few minutes.

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install nlp-shap

From a local clone (requires `uv <https://docs.astral.sh/uv/>`_):

.. code-block:: bash

   git clone https://github.com/Pawlo77/nlp-shap
   cd nlp-shap
   make install

Your first aggregation
----------------------

The estimand aggregators take boolean **coalition masks** and aligned payoffs
:math:`v(S)`. On a three-player majority game, Shapley and Banzhaf disagree:

.. code-block:: python

   from itertools import product

   from nlp_shap import BanzhafAggregator, ShapleyAggregator

   num_players = 3
   masks = [tuple(bits) for bits in product([False, True], repeat=num_players)]

   def majority_payoff(mask: tuple[bool, ...]) -> float:
       return 1.0 if sum(mask) >= 2 else 0.0

   payoffs = [majority_payoff(mask) for mask in masks]

   shapley = ShapleyAggregator().aggregate(masks, payoffs)
   banzhaf = BanzhafAggregator().aggregate(masks, payoffs)

   print("Shapley:", shapley)   # ~[0.333, 0.333, 0.333]
   print("Banzhaf:", banzhaf)   # [0.5, 0.5, 0.5]

Label results for archives
--------------------------

When you persist explain outputs, record which estimand was used:

.. code-block:: python

   from nlp_shap import Estimand, ExplainResult, RunManifest, parse_manifest

   result = ExplainResult(
       estimand=Estimand.SHAPLEY,
       values=(0.333, 0.333, 0.333),
   )
   manifest = RunManifest(estimand=result.estimand, run_id="run-42")
   restored = parse_manifest(manifest.to_dict())
   assert restored.estimand is Estimand.SHAPLEY

Next steps
----------

- :doc:`estimands` — coalition weights, manifests, and wire types
- :doc:`../theory/shapley_values` — Shapley axioms and uniqueness
- :doc:`../theory/estimands` — why Shapley and Banzhaf differ
- :doc:`applications` — business and compliance use cases
- :doc:`../examples` — `estimands_toy_game.ipynb`, `masking_views.ipynb`, `runtime_core.ipynb`, and `exact_estimation.ipynb`
- :doc:`masking` — coalition masking and absence policies
- :doc:`runtime` — archive persistence, dedup, and async scheduling
- :doc:`exact` — exact coalition enumeration and estimand wiring
- :doc:`../api` — module reference
