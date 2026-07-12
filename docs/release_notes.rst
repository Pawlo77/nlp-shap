Release notes
=============

Changes in each published ``nlp-shap`` version. Newest first.

.. _release-unreleased:

Unreleased
----------

Documentation
~~~~~~~~~~~~~

- Sphinx theory pages for cooperative games and estimands (Shapley vs Banzhaf).
- Usage guide for estimand aggregators, results, and manifests.
- Expanded API reference for public modules.
- Example notebook ``examples/estimands_toy_game.ipynb``.
- Furo theme, getting-started page, embedded notebook rendering, and API module layout.

Tooling
~~~~~~~

- ``make notebooks`` target to execute example notebooks in place before commit.

.. _release-0-1-1:

0.1.1 (2026-07-12)
------------------

Added
~~~~~

- :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator` and
  :class:`~nlp_shap.estimation.estimands.banzhaf.BanzhafAggregator` for
  coalition-weighted attribution from masks and payoffs.
- :class:`~nlp_shap.domain.estimands.Estimand` enum and wire types for honest
  estimand labelling.
- :class:`~nlp_shap.pipeline.result.ExplainResult` with required
  ``estimand`` field.
- :class:`~nlp_shap.pipeline.manifest.RunManifest`,
  :class:`~nlp_shap.pipeline.manifest.RunManifestPayload`, and
  :func:`~nlp_shap.pipeline.manifest.parse_manifest` for run-archive metadata.
- :class:`~nlp_shap.protocols.estimand.EstimandAggregator` protocol.

.. _release-0-1-0:

0.1.0 (2026-07-12)
------------------

Added
~~~~~

- Initial PyPI package layout and ``nlp_shap`` import surface.
- Logging bootstrap from ``[tool.logging]`` in ``pyproject.toml`` via
  ``logging518``.
