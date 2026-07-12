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
   api/estimation
   api/pipeline
   api/protocols

Package version
---------------

.. automodule:: nlp_shap
   :members: __version__
   :undoc-members:
