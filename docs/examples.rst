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

Toy-game walkthrough
--------------------

The notebook below is rendered inline. It compares Shapley and Banzhaf
aggregators on a three-player majority game and shows why estimand labelling
matters.

.. toctree::
   :maxdepth: 1

   notebooks/estimands_toy_game

Masking walkthrough
-------------------

Partition a conversation into token players, build shared masked views, and
compare delete, pad, and neutral absence policies.

.. toctree::
   :maxdepth: 1

   notebooks/masking_views

Gallery
-------

**estimands_toy_game.ipynb**
   End-to-end walkthrough of the 0.1.1 estimand API: Shapley and Banzhaf
   aggregators (labels, coalition weights, aggregation), majority vs additive
   games, :class:`~nlp_shap.pipeline.result.ExplainResult` labelling,
   :class:`~nlp_shap.pipeline.manifest.RunManifest` wire format, and plugin
   entry-point resolution. CPU-only; no optional extras.

**masking_views.ipynb**
   End-to-end walkthrough of the 0.1.3 masking API: token partition,
   :class:`~nlp_shap.masking.policies.DeletePolicy`,
   :class:`~nlp_shap.masking.policies.PadPolicy`, and
   :class:`~nlp_shap.masking.policies.NeutralPolicy` (defaults and custom
   parameters), :class:`~nlp_shap.masking.builder.MaskedSnapshot`,
   :class:`~nlp_shap.masking.codec.MaskCodec`,
   :class:`~nlp_shap.masking.space.MaskSpace`, plugin registry
   resolution, and validation guards. CPU-only; no optional extras.

Source on GitHub
----------------

- `estimands_toy_game.ipynb <https://github.com/Pawlo77/nlp-shap/blob/main/examples/estimands_toy_game.ipynb>`_
- `masking_views.ipynb <https://github.com/Pawlo77/nlp-shap/blob/main/examples/masking_views.ipynb>`_
- `examples/README.md <https://github.com/Pawlo77/nlp-shap/blob/main/examples/README.md>`_
