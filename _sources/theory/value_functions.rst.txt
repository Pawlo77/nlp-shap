Value functions and normalizers
===============================

Coalition **utility** is distinct from cooperative-game **value** (Shapley/Banzhaf
attributions). Value functions map model outputs to scalar payoffs; normalizers
reshape aggregated attributions for display only.

Utility families
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Plugin key
     - Paper utility
     - Mechanism
   * - ``tfidf_cosine``
     - U3
     - Hashed token rows → frozen-corpus TF-IDF → cosine similarity
   * - ``embedding_cosine``
     - U1 / U2
     - Cosine similarity of static or contextual embeddings
   * - ``embedding_euclidean``
     - U4
     - ``1 / (1 + L2 distance)`` between embeddings
   * - ``logprob``
     - sequence logprob
     - Sum of per-token log probabilities (stub when unavailable)

Frozen TF-IDF
~~~~~~~~~~~~~

Unlike MLLM-Shap 0.x, ``nlp-shap`` freezes IDF weights from the full run corpus
at explain start. Scoring reuses those weights for every coalition comparison.

Normalizers
-----------

Applied **after** :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator`
(or Banzhaf) aggregation:

* **Identity** — raw attributions
* **AbsSum** — divide by sum of absolute values
* **PowerShift** — shift to non-negative, power transform, scale to sum 1
* **MinMax** — scale to ``[0, 1]``, then scale to sum 1

Archive rows store pre-normalization utilities and attributions so
``reanalyze`` (Phase 8) can swap value functions or normalizers without new
model calls.
