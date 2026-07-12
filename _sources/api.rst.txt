API reference
=============

``nlp-shap`` exposes a small, typed public surface. Import the symbols below
from :mod:`nlp_shap` or follow the module pages for implementation detail.

Public exports
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symbol
     - Description
   * - :class:`~nlp_shap.ExplainConfig`
     - Top-level explain pipeline configuration. See :doc:`api/pipeline`.
   * - :func:`~nlp_shap.explain_config_from_yaml`
     - Parse YAML into :class:`~nlp_shap.ExplainConfig`.
   * - :func:`~nlp_shap.explain_config_to_yaml`
     - Serialize :class:`~nlp_shap.ExplainConfig` to YAML.
   * - :class:`~nlp_shap.ConversationSnapshot`
     - Frozen conversation input for explainability. See :doc:`api/domain`.
   * - :class:`~nlp_shap.PlayerSet`
     - Ordered explainability players. See :doc:`api/domain`.
   * - :class:`~nlp_shap.CoalitionMask`
     - Boolean coalition presence mask. See :doc:`api/domain`.
   * - :class:`~nlp_shap.CooperativeGame`
     - Cooperative-game player set and references. See :doc:`api/domain`.
   * - :class:`~nlp_shap.PluginRegistry`
     - Plugin discovery and resolution. See :doc:`api/plugins`.
   * - :class:`~nlp_shap.Estimand`
     - Estimand label enum (``shapley`` | ``banzhaf``). See :doc:`api/domain`.
   * - :class:`~nlp_shap.ShapleyAggregator`
     - Shapley-weighted coalition aggregation. See :doc:`api/estimation`.
   * - :class:`~nlp_shap.BanzhafAggregator`
     - Uniform Banzhaf coalition aggregation. See :doc:`api/estimation`.
   * - :class:`~nlp_shap.ExplainResult`
     - Labelled explain output. See :doc:`api/pipeline`.
   * - :class:`~nlp_shap.RunManifest`
     - Run-archive metadata builder. See :doc:`api/pipeline`.
   * - :func:`~nlp_shap.parse_manifest`
     - Parse manifest payloads at archive boundaries. See :doc:`api/pipeline`.

Modules
-------

.. toctree::
   :maxdepth: 1

   api/domain
   api/protocols
   api/plugins
   api/pipeline
   api/estimation

Package version
---------------

.. automodule:: nlp_shap
   :members: __version__
   :undoc-members:
