Exact estimation
================

The :class:`~nlp_shap.estimation.exact.ExactEstimator` lazily enumerates every
coalition except the grand coalition and delegates attribution to an estimand
plugin. Use it as ground truth on tiny player sets before trusting approximate
samplers.

Enumeration scope
-----------------

For :math:`n` players the estimator evaluates :math:`2^n - 1` coalitions — all
masks except the all-present coalition, which backends often precompute via
``precompute_base``.

.. math::

   |\mathcal{S}_{\mathrm{exact}}| = 2^n - 1

Complexity grows exponentially in :math:`n`; reserve exact estimation for toy
models, sanity checks, and estimator validation.

Estimand delegation
-------------------

Exact enumeration produces coalition payoffs :math:`v(S)`. The estimand plugin
(Shapley or Banzhaf) applies coalition weights and marginal contributions:

.. math::

   \text{ExactEstimator} \rightarrow \{v(S)\}_{S}
   \xrightarrow{\text{EstimandAggregator}}
   (\phi_1, \ldots, \phi_n)

Monte Carlo and Neyman estimators reuse the same estimand boundary — exact
estimation establishes the reference path for validating approximate samplers.

Legacy parity
-------------

The implementation matches legacy ``PreciseShapExplainer`` Shapley values on
additive characteristic functions when all :math:`2^n` coalitions (including the
grand coalition) are supplied for aggregation.

Further reading
---------------

- Estimand definitions: :doc:`estimands`
- Cooperative-game foundations: :doc:`cooperative_games`
- Usage: :doc:`../guides/exact`
- API: :class:`~nlp_shap.estimation.exact.ExactEstimator`
