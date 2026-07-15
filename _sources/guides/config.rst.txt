Configuration
=============

``nlp-shap`` uses a single :class:`~nlp_shap.pipeline.config.ExplainConfig`
schema with three top-level sections: ``backend``, ``generation``, and
``explanation``. The schema replaces split kwargs and separate model config
objects from legacy explain pipelines.

YAML skeleton
---------------

.. code-block:: yaml

   backend:
     kind: lmstudio
     model_id: qwen2-500m-instruct
     api_host: null

   generation:
     max_new_tokens: 128
     temperature: 0.0
     top_k: 1
     precompute_base: true

   explanation:
     estimand: shapley
     estimator: neyman_cc
     value_fn: tfidf_cosine
     normalizer: identity
     players: tokens
     absence_policy: delete
     budget:
       fraction: 0.3
     include_minimal_masks: false
     max_inflight: 2
     archive:
       path: ./runs/{run_id}
       flush_every: 50
     dedup:
       enabled: auto
     kv_cache:
       enabled: true
     embedding_mode: static
     seed: 42

Load and save
-------------

.. code-block:: python

   from pathlib import Path

   from nlp_shap import ExplainConfig, explain_config_from_yaml, explain_config_to_yaml

   text = Path("config.yaml").read_text(encoding="utf-8")
   config = explain_config_from_yaml(text)
   assert isinstance(config, ExplainConfig)

   Path("config.roundtrip.yaml").write_text(
       explain_config_to_yaml(config),
       encoding="utf-8",
   )

.. guide-result:: config_roundtrip

Budget flags
------------

Legacy ``standard`` and ``limited`` estimator variants map to
``explanation.budget.fraction`` and ``explanation.include_minimal_masks``:

- **Standard sampling:** ``fraction: 1.0``, ``include_minimal_masks: false``
- **Limited sampling:** ``fraction < 1.0`` and/or ``include_minimal_masks: true``

Further reading
---------------

- Domain types: :doc:`../api/domain`
- Plugin registry: :doc:`../api/plugins`
- Masking and absence policies: :doc:`masking`
- Runtime archive and scheduler: :doc:`runtime`
- Exact estimation: :doc:`exact`
- Estimand guide: :doc:`estimands`
