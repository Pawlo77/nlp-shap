Release notes
=============

Changes in each published ``nlp-shap`` version. Newest first.

.. _release-unreleased:

Unreleased
----------

Documentation
~~~~~~~~~~~~~

- Expanded theory: Shapley axioms, uniqueness theorem, cooperative-game foundations.
- Business and compliance applications guide with verified references.
- Corrected bibliographic links (Shapley 1953 DOI, SHAP NeurIPS / arXiv).

.. _release-0-1-2:

0.1.2 (2026-07-12)
------------------

Added
~~~~~

- Domain types: :class:`~nlp_shap.ConversationSnapshot`,
  :class:`~nlp_shap.Turn`, :class:`~nlp_shap.Message`,
  :class:`~nlp_shap.PlayerSet`, :class:`~nlp_shap.CoalitionMask`,
  :class:`~nlp_shap.CooperativeGame`, and conversation enums.
- Protocols for backends, estimators, value functions, partitions, absence
  policies, and embedding providers.
- :class:`~nlp_shap.PluginRegistry` with packaging entry-point discovery.
- :class:`~nlp_shap.ExplainConfig` (Pydantic v2) with YAML load/save helpers.
- Configuration guide and expanded API reference pages.

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
