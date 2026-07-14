Approximate estimation
======================

When :math:`n` players make exact enumeration infeasible, ``nlp-shap`` approximates
Shapley-style attributions by sampling coalitions. Three built-in estimators trade
budget for accuracy:

- :class:`~nlp_shap.estimation.monte_carlo.MonteCarloEstimator` — uniform random
  coalitions; attributions come from estimand plugins (Shapley or Banzhaf).
- :class:`~nlp_shap.estimation.complementary.ComplementaryEstimator` — symmetric
  pairs :math:`(S, N\setminus S)` with complementary-contribution (CC) aggregation.
- :class:`~nlp_shap.estimation.neyman.NeymanEstimator` — two-phase CC sampling with
  Neyman allocation over coalition sizes.

Budget model
------------

All estimators read ``ExplainConfig.explanation.budget.fraction`` in
:math:`(0, 1]`. The fraction scales a method-specific maximum:

.. list-table::
   :header-rows: 1

   * - Estimator
     - Maximum coalitions
   * - Monte Carlo
     - :math:`2^n - 1` (grand coalition excluded)
   * - Complementary / Neyman
     - :math:`2^n - 2` (empty and grand masks excluded; pairs are even)

Monte Carlo and estimand delegation
-----------------------------------

Monte Carlo draws random coalitions and **does not** embed Shapley weights.
Coalition payoffs :math:`v(S)` are aggregated by the selected estimand plugin:

.. math::

   \text{MC} \rightarrow \{v(S_k)\}_{k=1}^{m}
   \xrightarrow{\text{EstimandAggregator}}
   (\phi_1, \ldots, \phi_n)

The same sampled masks yield different attributions under Shapley vs Banzhaf —
a useful sanity check that estimand wiring is correct.

Complementary contributions
---------------------------

Complementary sampling evaluates symmetric pairs and accumulates CC statistics in
:math:`M_{i,j}` (how often player :math:`i` appears in size-:math:`j` coalitions)
and :math:`C_{i,j}` (summed complementary contributions). For pair
:math:`(S, N\setminus S)` with payoffs :math:`v(S)` and :math:`v(N\setminus S)`:

.. math::

   u = v(S) - v(N\setminus S)

Each :math:`u` updates :math:`C` for players present in :math:`S` and
:math:`N\setminus S`. Final attributions combine element-wise ratios
:math:`C_{i,j}/M_{i,j}` across coalition sizes.

Neyman allocation
-----------------

Neyman-CC extends complementary sampling with two phases:

1. **Initial sampling** — structured draws that fill every :math:`M_{i,j}` cell
   at least :math:`m_{\text{init}}` times (unless the total budget is exhausted).
2. **Neyman allocation** — remaining budget is split across coalition sizes
   :math:`j \ge \lceil n/2 \rceil` proportional to estimated standard deviations
   :math:`\hat\sigma_{i,j}`.

Variance estimates use the complementary sample variance:

.. math::

   \hat\sigma^2_{i,j} = \frac{1}{M_{i,j}-1}
   \left(\sum u^2 - \frac{(\sum u)^2}{M_{i,j}}\right)

Allocation follows the Neyman rule (symmetric halves summed):

.. math::

   \hat M_j \propto
   \sqrt{
     \sum_{i} \frac{\hat\sigma^2_{i,j}}{j+1}
     + \sum_{i} \frac{\hat\sigma^2_{i,n-j-1}}{n-j}
   }

**Approximation floor:** when the total budget is small, phase one may consume the
entire allowance before Neyman allocation begins. In that regime the estimator
behaves like complementary sampling — a useful lower bound on accuracy per call.

Because allocation depends on phase-one payoffs, Neyman requires a two-step API:
:meth:`~nlp_shap.estimation.neyman.NeymanEstimator.sample_masks` (phase one),
:meth:`~nlp_shap.estimation.neyman.NeymanEstimator.begin_allocation`, then
:meth:`~nlp_shap.estimation.neyman.NeymanEstimator.sample_allocation_masks`.

Further reading
---------------

- Exact reference path: :doc:`exact`
- Estimand definitions: :doc:`estimands`
- Usage: :doc:`../guides/approximation`
- API: :doc:`../api/estimation`
