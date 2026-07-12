Release notes
=============

Changes in each published ``nlp-shap`` version. Newest first.

.. _release-unreleased:

Unreleased
----------

.. _release-0-1-3:

0.1.3 (2026-07-12)
------------------

Added
~~~~~

- Masking codec: :class:`~nlp_shap.masking.codec.PackedMask`,
  :class:`~nlp_shap.masking.codec.MaskCodec` with legacy-compatible packed hashing.
- :class:`~nlp_shap.masking.space.MaskSpace` for projecting feature splits to full masks.
- View-based :class:`~nlp_shap.masking.builder.MaskedSnapshot` and
  :class:`~nlp_shap.masking.builder.MaskBuilder` with structural sharing.
- Absence policies: :class:`~nlp_shap.masking.policies.DeletePolicy`,
  :class:`~nlp_shap.masking.policies.PadPolicy`,
  :class:`~nlp_shap.masking.policies.NeutralPolicy`.
- :class:`~nlp_shap.masking.partitions.TokenPartitioner` and built-in plugin
  registration for ``tokens``, ``delete``, ``pad``, and ``neutral``.
- ``numpy`` core dependency for mask packing.

Documentation
~~~~~~~~~~~~~

- Theory and usage guides for coalition masking and absence policies.
- API reference page for :mod:`nlp_shap.masking`.
- Example notebook ``examples/masking_views.ipynb``.

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
