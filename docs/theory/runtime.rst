Runtime core
============

Phase 3 introduces the execution substrate for coalition evaluations:

1. **Run archive** — SQLite index plus blob storage for generation text.
2. **Dedup registry** — SHA256 coalition keys that collapse repeated work.
3. **Hot store** — in-memory LRU for recent generations.
4. **Inference scheduler** — async execution with bounded ``max_inflight``.

Run archive
-----------

:class:`~nlp_shap.runtime.archive.RunArchive` persists :class:`~nlp_shap.runtime.archive.CoalitionRecord`
rows with mask bytes, utility, timing, and cache-hit flags. Generation text is
stored under ``blobs/`` so :meth:`~nlp_shap.runtime.archive.RunArchive.history_lazy`
can stream rows without loading every blob into memory. Each archive directory
also contains ``manifest.json`` from :class:`~nlp_shap.pipeline.manifest.RunManifest`.

Coalition deduplication
-----------------------

:class:`~nlp_shap.runtime.dedup.build_coalition_key` hashes snapshot id, player
order, packed mask bytes, absence policy, model id, and generation settings.
:class:`~nlp_shap.runtime.dedup.CoalitionDedupRegistry` tracks which keys already
required backend execution. :func:`~nlp_shap.runtime.dedup.dedup_enabled` maps
``ExplainConfig`` dedup mode to deterministic generation (``auto`` follows
``temperature == 0``).

Scheduling
----------

:class:`~nlp_shap.runtime.scheduler.InferenceScheduler` checks the hot store,
then the dedup registry, then acquires a concurrency semaphore before calling
the backend. :class:`~nlp_shap.runtime.scheduler.SchedulerMetrics` reports
``requested``, ``executed``, ``deduplicated``, and ``cache_hits``.

See also
--------

- :doc:`../guides/runtime` — usage examples
- :doc:`../api/runtime` — module reference
- :doc:`../guides/config` — ``archive``, ``dedup``, and ``max_inflight`` settings
