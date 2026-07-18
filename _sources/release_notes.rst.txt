Release notes
=============

Changes in each published ``nlp-shap`` version. Newest first.

.. _release-unreleased:

Unreleased
----------

Added
~~~~~

- Optional sync :class:`~nlp_shap.runtime.progress.CoalitionProgress` callbacks on
  :class:`~nlp_shap.runtime.scheduler.InferenceScheduler` and
  :class:`~nlp_shap.pipeline.runner.ExplainRunner` for per-coalition progress
  reporting (no-op when omitted).

Changed
~~~~~~~

- :class:`~nlp_shap.masking.policies.DeletePolicy` renders the empty coalition as
  empty message text instead of raising, so exact Shapley can evaluate
  :math:`v(\emptyset)` under deletion absence.
- :class:`~nlp_shap.estimation.neyman.NeymanEstimator` initial sampling no longer
  waits on empty/grand M-matrix columns (which are never yielded), fixing an
  infinite loop on small player counts.

.. _release-0-1-15:

0.1.15 (2026-07-15)
-------------------

Added
~~~~~

- Runtime micro-benchmarks in ``tests/benchmarks/`` for dedup keys, mask
  pack/unpack, archive round-trip (10k coalitions), and prefix-cache hits.
- ``make bench`` and ``make bench-regression`` targets with recorded
  ``baselines.json`` and an optional >10% regression gate.

.. _release-0-1-14:

0.1.14 (2026-07-15)
-------------------

Added
~~~~~

- :doc:`guides/extending` user guide for optional extras and plugin entry points.
- ``[dev]`` optional extra for contributor tooling.
- Base-wheel smoke test ensuring ``torch`` is not installed without extras.

Changed
~~~~~~~

- Removed the transitional ``explain.use_v2`` YAML flag; the v2 explain pipeline
  is now the only path.
- :func:`~nlp_shap.render_attribution` and :func:`~nlp_shap.render_attribution_html`
  load lazily so ``import nlp_shap`` works without ``[viz]``.

Documentation
~~~~~~~~~~~~~

- README quickstart with :class:`~nlp_shap.ExplainRunner`.
- Getting-started explain walkthrough and API export table updates.

.. _release-0-1-13:

0.1.13 (2026-07-15)
-------------------

Added
~~~~~

- :mod:`nlp_shap.viz` with ``token_text`` and ``token_bar`` renderer plugins and
  :func:`~nlp_shap.render_attribution` public helper.
- :class:`~nlp_shap.protocols.renderer.AttributionRenderer` protocol and
  ``nlp_shap.renderers`` entry-point group.
- Optional ``[viz]`` extra (``matplotlib``, ``ipywidgets``).
- :doc:`guides/visualization` user guide and ``attribution_viz`` /
  ``text_explain_e2e`` example notebooks.

Fixed
~~~~~

- Lazy visualization imports so the published wheel passes the base ``import nlp_shap``
  smoke test without ``matplotlib``.

.. _release-0-1-12:

0.1.12 (2026-07-15)
-------------------

Added
~~~~~

- :class:`~nlp_shap.backends.api.ApiBackend` for OpenAI-compatible
  ``/chat/completions`` HTTP APIs via ``httpx``.
- In-memory response deduplication for identical request bodies at
  ``temperature: 0.0``.
- Optional ``[api]`` extra, ``config/dev_api.yaml``, and mock-transport contract
  tests in CI.
- User guide for generative backends: :doc:`guides/backends`.

.. _release-0-1-11:

0.1.11 (2026-07-15)
-------------------

Added
~~~~~

- :class:`~nlp_shap.backends.transformers.text.TransformersTextBackend` with chat
  templates, text-only validation, and contextual embedding hooks.
- :class:`~nlp_shap.runtime.kv_cache.PrefixCacheManager` with scheduler prefix
  grouping and ``kv_cache_hits`` metrics on :class:`~nlp_shap.runtime.scheduler.SchedulerMetrics`.
- Optional ``[transformers]`` extra and ``config/dev_transformers.yaml``.
- ``@pytest.mark.gpu`` integration tests for local Hugging Face runs.

.. _release-0-1-10:

0.1.10 (2026-07-15)
-------------------

Added
~~~~~

- :class:`~nlp_shap.backends.lmstudio.LmStudioBackend` using ``lmstudio.AsyncClient``.
- :exc:`~nlp_shap.errors.BackendUnavailableError` when the LM Studio API server is down.
- Optional ``[lmstudio]`` extra, ``config/dev_lmstudio.yaml``, and ``@pytest.mark.lms`` tests.

.. _release-0-1-9:

0.1.9 (2026-07-15)
------------------

Added
~~~~~

- :meth:`~nlp_shap.pipeline.runner.ExplainRunner.reanalyze` rescoring on archived
  coalitions with zero backend executions.
- Grand-coalition ``base_generation.txt`` persisted in run archives before the
  coalition scheduler loop when ``precompute_base`` is enabled.
- :mod:`nlp_shap.runtime.telemetry` with :class:`~nlp_shap.runtime.telemetry.InMemoryObservabilitySink`
  stage spans for orchestrator and estimator phases.
- :class:`~nlp_shap.runtime.metrics.PerfSummary` on
  :class:`~nlp_shap.pipeline.result.ExplainRunOutput`.
- :meth:`~nlp_shap.runtime.archive.RunArchive.load` for reopening existing archives.

.. _release-0-1-8:

0.1.8 (2026-07-14)
------------------

Added
~~~~~

- :class:`~nlp_shap.pipeline.runner.ExplainRunner` with ``explain`` and
  ``explain_sync`` entry points.
- :class:`~nlp_shap.pipeline.orchestrator.ExplainOrchestrator` wiring partition,
  estimation, generation, value functions, and normalizers.
- :class:`~nlp_shap.backends.mock.MockBackend` for deterministic CI-safe E2E runs.
- :class:`~nlp_shap.pipeline.result.ExplainRunOutput` with scheduler metrics and
  run manifest metadata.
- ``mock`` backend entry point and :meth:`~nlp_shap.plugins.registry.PluginRegistry.resolve_type`.

.. _release-0-1-7:

0.1.7 (2026-07-14)
------------------

Added
~~~~~

- :class:`~nlp_shap.value.tfidf.TfIdfCosineValue` with frozen-corpus TF-IDF cosine
  scoring (U3).
- :class:`~nlp_shap.value.embedding.CosineEmbeddingValue` and
  :class:`~nlp_shap.value.embedding.EuclideanEmbeddingValue` for embedding utilities
  (U1/U2/U4).
- :class:`~nlp_shap.value.logprob.LogprobValue` with deterministic stub logprobs.
- Presentation normalizers in ``nlp_shap.estimation.normalizers``.
- :class:`~nlp_shap.GenerationRecord` for token rows, embeddings, and logprobs.
- ``tfidf_cosine``, ``embedding_cosine``, ``embedding_euclidean``, and ``logprob``
  value-function entry points; ``identity``, ``abs_sum``, ``power_shift``, and
  ``min_max`` normalizer entry points.

.. _release-0-1-6:

0.1.6 (2026-07-14)
------------------

Added
~~~~~

- :class:`~nlp_shap.estimation.monte_carlo.MonteCarloEstimator` for budgeted
  random coalition sampling with estimand-plugin aggregation.
- :class:`~nlp_shap.estimation.complementary.ComplementaryEstimator` for
  complementary-pair sampling and CC Shapley aggregation.
- :class:`~nlp_shap.estimation.neyman.NeymanEstimator` for two-phase Neyman-CC
  sampling with :class:`~nlp_shap.pipeline.config.NeymanConfig` controls.
- Shared sampling helpers in ``nlp_shap.estimation._shared``.
- ``mc``, ``complementary``, and ``neyman_cc`` estimator entry points.
- Theory and guide pages for approximate estimation.
- ``estimator_comparison.ipynb`` comparing budget, sample count, and accuracy.

Changed
~~~~~~~

- :class:`~nlp_shap.pipeline.config.ExplanationConfig` adds ``neyman`` settings
  for Neyman-CC runs.
- Drop unnecessary ``from __future__ import annotations`` across the package;
  document the policy in development rules.
- Consolidate :doc:`examples` into a single list-table with hidden toctree.

.. _release-0-1-5:

0.1.5 (2026-07-13)
------------------

Added
~~~~~

- :class:`~nlp_shap.estimation.exact.ExactEstimator` for lazy coalition
  enumeration with estimand-plugin aggregation.
- :meth:`~nlp_shap.estimation.exact.ExactEstimator.iter_mask_ints` and
  :meth:`~nlp_shap.estimation.exact.ExactEstimator.iter_masks` for streaming
  masks without materializing ``2^n`` coalitions.
- :meth:`~nlp_shap.estimation.exact.ExactEstimator.estimate_attributions` wires
  coalition payoffs to Shapley or Banzhaf aggregators.
- Vectorized exact Shapley aggregation for complete characteristic tables.
- :meth:`~nlp_shap.runtime.scheduler.InferenceScheduler.run_iter` and
  ``run_stream`` for bounded pending-task scheduling.
- ``exact`` estimator entry point under ``nlp_shap.estimators``.

Changed
~~~~~~~

- :class:`~nlp_shap.runtime.dedup.build_coalition_key` uses compact binary
  hashing instead of JSON serialization.
- Marginal estimand aggregation uses integer bitmask keys and cached factorial
  weights.
- :class:`~nlp_shap.protocols.estimator.EstimatorStrategy.sample_masks` yields
  an iterator instead of a materialized tuple.

Documentation
~~~~~~~~~~~~~

- Theory and usage guides for exact estimation and streaming scheduler usage.
- API reference for :mod:`nlp_shap.estimation.exact`.
- Example notebook ``examples/exact_estimation.ipynb``.
- Performance review rules in contributor workflow.

.. _release-0-1-4:

0.1.4 (2026-07-13)
------------------

Added
~~~~~

- :class:`~nlp_shap.runtime.archive.RunArchive` with SQLite index, blob storage,
  and lazy :meth:`~nlp_shap.runtime.archive.RunArchive.history_lazy` iteration.
- :class:`~nlp_shap.runtime.archive.CoalitionRecord` rows with mask bytes,
  utility, timing, and cache-hit metadata.
- :class:`~nlp_shap.runtime.dedup.CoalitionDedupRegistry` and
  :func:`~nlp_shap.runtime.dedup.build_coalition_key` SHA256 coalition keys.
- :class:`~nlp_shap.runtime.store.HotResultStore` LRU cache for recent generations.
- :class:`~nlp_shap.runtime.scheduler.InferenceScheduler` with bounded
  ``max_inflight`` concurrency and :class:`~nlp_shap.runtime.scheduler.SchedulerMetrics`.

Documentation
~~~~~~~~~~~~~

- Theory and usage guides for the runtime archive, dedup, and scheduler.
- API reference page for :mod:`nlp_shap.runtime`.
- Example notebook ``examples/runtime_core.ipynb``.

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
