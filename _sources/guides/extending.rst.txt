Extending nlp-shap
==================

``nlp-shap`` is built around a small plugin registry. Third-party packages can
register estimators, value functions, backends, and renderers without forking
the core library.

Optional extras
---------------

Install only what your deployment needs:

.. code-block:: bash

   pip install nlp-shap                    # core only (no torch)
   pip install "nlp-shap[transformers]"    # Hugging Face text backend
   pip install "nlp-shap[lmstudio]"        # LM Studio SDK backend
   pip install "nlp-shap[api]"             # OpenAI-compatible HTTP backend
   pip install "nlp-shap[viz]"             # matplotlib token attribution charts

Entry-point groups
------------------

Register plugins under ``project.entry-points`` in your package
``pyproject.toml``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Purpose
   * - ``nlp_shap.estimators``
     - Coalition sampling strategies (``exact``, ``mc``, …)
   * - ``nlp_shap.estimands``
     - Payoff aggregation (``shapley``, ``banzhaf``)
   * - ``nlp_shap.value_fns``
     - Coalition scoring utilities (``tfidf_cosine``, ``logprob``, …)
   * - ``nlp_shap.normalizers``
     - Post-aggregation scaling (``identity``, ``min_max``, …)
   * - ``nlp_shap.partitions``
     - Player definitions (``tokens``, …)
   * - ``nlp_shap.absence_policies``
     - Masked snapshot rendering (``delete``, ``pad``, …)
   * - ``nlp_shap.backends``
     - Model inference connectors (``mock``, ``transformers``, …)
   * - ``nlp_shap.renderers``
     - Attribution visualization (``token_text``, ``token_bar``)

Example custom value function
-----------------------------

.. code-block:: toml

   [project.entry-points."nlp_shap.value_fns"]
   my_metric = "my_package.values:MyMetricValue"

Your class should satisfy :class:`~nlp_shap.protocols.value.ValueFunction`.
Resolve it by name in :class:`~nlp_shap.pipeline.config.ExplainConfig`:

.. code-block:: yaml

   explanation:
     value_fn: my_metric

Running with a custom registry
------------------------------

Pass an explicit :class:`~nlp_shap.PluginRegistry` to
:class:`~nlp_shap.ExplainRunner` when you need to load entry points from
multiple distributions or override builtins in tests:

.. code-block:: python

   from nlp_shap import ExplainRunner, PluginRegistry
   from nlp_shap.plugins import PluginGroup

   registry = PluginRegistry()
   registry.load_entry_points(PluginGroup.VALUE_FNS)
   runner = ExplainRunner(config, registry=registry)

Further reading
---------------

- :doc:`../api/plugins` — registry API and built-in groups
- :doc:`../api/protocols` — strategy protocols for plugins
- :doc:`config` — :class:`~nlp_shap.ExplainConfig` schema
