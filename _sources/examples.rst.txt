Examples
========

The ``examples/`` directory contains **Jupyter notebooks** with runnable
``nlp-shap`` workflows. They complement guides, theory pages, and the API
reference.

Setup
-----

Install the package:

.. code-block:: bash

   pip install nlp-shap

From a local clone:

.. code-block:: bash

   make install

Use the project ``.venv`` as the notebook kernel when working from source.

Notebooks
---------

Each row links to the **rendered walkthrough** in these docs. Use **Source** to
open the ``.ipynb`` on GitHub or browse the full catalog in
`examples/README.md <https://github.com/Pawlo77/nlp-shap/blob/main/examples/README.md>`_.

.. list-table::
   :header-rows: 1
   :widths: 24 56 20

   * - Notebook
     - What it demonstrates
     - Source

   * - :doc:`notebooks/estimands_toy_game`
     - Shapley and Banzhaf aggregators on majority vs additive games;
       :class:`~nlp_shap.pipeline.result.ExplainResult` labelling and
       :class:`~nlp_shap.pipeline.manifest.RunManifest` wire format. CPU-only.
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/estimands_toy_game.ipynb>`_

   * - :doc:`notebooks/masking_views`
     - Token partition, delete/pad/neutral absence policies,
       :class:`~nlp_shap.masking.builder.MaskedSnapshot`,
       :class:`~nlp_shap.masking.codec.MaskCodec`, and
       :class:`~nlp_shap.masking.space.MaskSpace`. CPU-only.
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/masking_views.ipynb>`_

   * - :doc:`notebooks/runtime_core`
     - Run archive persistence, coalition dedup keys, hot LRU cache, and async
       :class:`~nlp_shap.runtime.scheduler.InferenceScheduler` metrics.
       CPU-only.
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/runtime_core.ipynb>`_

   * - :doc:`notebooks/exact_estimation`
     - :class:`~nlp_shap.estimation.exact.ExactEstimator` coalition enumeration
       and :meth:`~nlp_shap.estimation.exact.ExactEstimator.estimate_attributions`
       estimand delegation with budget guards. CPU-only.
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/exact_estimation.ipynb>`_

   * - :doc:`notebooks/estimator_comparison`
     - Eight-player benchmark (cap ≤ 20) plus five-player fraction sweep
       0.1–0.5, MC bias analysis, multi-seed L1 vs exact Shapley. CPU-only.
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/estimator_comparison.ipynb>`_

   * - :doc:`notebooks/attribution_viz`
     - Mock exact explain run with ``token_text`` and ``token_bar`` attribution
       renderers. Requires ``[viz]``.
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/attribution_viz.ipynb>`_

   * - :doc:`notebooks/text_explain_e2e`
     - Refund-policy prompt explained with LM Studio and transformers backends on
       the same model id, then visualized with
       :func:`~nlp_shap.render_attribution`. Optional LM Studio at runtime.
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/text_explain_e2e.ipynb>`_

.. toctree::
   :hidden:
   :maxdepth: 1

   notebooks/estimands_toy_game
   notebooks/masking_views
   notebooks/runtime_core
   notebooks/exact_estimation
   notebooks/estimator_comparison
   notebooks/attribution_viz
   notebooks/text_explain_e2e
