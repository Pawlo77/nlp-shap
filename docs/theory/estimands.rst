Estimands: Shapley vs Banzhaf
==============================

An **estimand** is the quantity your attribution procedure targets. Monte Carlo
coalition averaging does **not** automatically estimate Shapley values; unweighted
means over sampled coalitions estimate the **Banzhaf index** unless Shapley
weights are applied explicitly.

``nlp-shap`` keeps estimands separate from estimators so results and archive
manifests are honestly labelled.

Shapley value
-------------

The Shapley value :math:`\phi_i(v)` for player :math:`i` is:

.. math::

   \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!\,(|N|-|S|-1)!}{|N|!}
   \bigl(v(S \cup \{i\}) - v(S)\bigr)

The weight for a coalition of size :math:`|S| = k` (excluding :math:`i`) is:

.. math::

   w_{\mathrm{Shapley}}(k, n) = \frac{k!\,(n-k-1)!}{n!}

Implemented in :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator`.

**Properties (axiomatic):** efficiency, symmetry, dummy player, additivity —
see Shapley (1953).

Banzhaf index
-------------

The Banzhaf index :math:`\beta_i(v)` uses **uniform** weights over
:math:`S \subseteq N \setminus \{i\}`:

.. math::

   \beta_i(v) = \frac{1}{2^{|N|-1}}
   \sum_{S \subseteq N \setminus \{i\}}
   \bigl(v(S \cup \{i\}) - v(S)\bigr)

Equivalently, :math:`w_{\mathrm{Banzhaf}}(k, n) = 2^{-(n-1)}` for every
coalition size :math:`k`.

Implemented in :class:`~nlp_shap.estimation.estimands.banzhaf.BanzhafAggregator`.

When they differ
----------------

For **additive** games :math:`v(S) = \sum_{i \in S} a_i`, Shapley and Banzhaf
coincide (both recover coefficients :math:`a_i`).

For **non-additive** games they diverge. A standard toy example is the
**majority/threshold** game with :math:`|N|=3` and :math:`v(S)=1` iff
:math:`|S| \ge 2`, else :math:`0`:

- Shapley: each player receives :math:`2/6 \approx 0.333`
- Banzhaf: each player receives :math:`0.5`

The test suite locks this divergence on identical coalition samples.

Labelling requirement
---------------------

Every :class:`~nlp_shap.pipeline.result.ExplainResult` carries an
:class:`~nlp_shap.domain.estimands.Estimand` label. Run archives persist the
same label in :class:`~nlp_shap.pipeline.manifest.RunManifestPayload` so
downstream analysis never misreports Banzhaf output as Shapley.

0.x motivation
--------------

Legacy MLLM-Shap Monte Carlo paths averaged coalition contributions without
Shapley weights while labelling outputs "Shapley". Phase 0 fixes this at the
architecture level by splitting :class:`~nlp_shap.protocols.estimand.EstimandAggregator`
from future estimator plugins.

References
----------

- Shapley, L. S. (1953). *A Value for n-person Games*.
  `Project Euclid <https://projecteuclid.org/journals/contributions-to-the-theory-of-games/volume-2/issue-none/A-Value-for-n-person-Games/10.1215/sm/1175702444.full>`__
- Banzhaf, J. F. (1965). *Weighted voting doesn't work: A mathematical analysis*.
  *Rutgers Law Review*, 19(2), 317–343.
- Strumbelj, E., & Kononenko, I. (2014). *Explaining prediction models and
  individual predictions with feature contributions*.
  `DOI:10.1016/j.knosys.2013.10.012 <https://doi.org/10.1016/j.knosys.2013.10.012>`__
  (Banzhaf-style unweighted coalition averages in explainability)
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting
  Model Predictions* (SHAP). NeurIPS.
  `NeurIPS proceedings <https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions>`__
- Fryer, D., Strümke, I., & Nguyen, H. (2021). *Shapley Values for NLP
  Explainability*. `arXiv:2104.08744 <https://arxiv.org/abs/2104.08744>`__
