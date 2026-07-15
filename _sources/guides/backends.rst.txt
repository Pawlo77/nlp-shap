Generative backends
===================

``nlp-shap`` routes coalition evaluations through pluggable **generative
backends**. Each backend implements :class:`~nlp_shap.protocols.backend.GenerativeBackend`
and is selected with ``backend.kind`` in :class:`~nlp_shap.ExplainConfig`.

Built-in backends
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - ``kind``
     - Extra
     - Use when
   * - ``mock``
     - *(core)*
     - Deterministic CI runs, unit tests, and offline demos.
   * - ``api``
     - ``[api]``
     - Remote OpenAI-compatible HTTP APIs (OpenAI, vLLM, local proxies).
   * - ``lmstudio``
     - ``[lmstudio]``
     - Local LM Studio models via the official async SDK.
   * - ``transformers``
     - ``[transformers]``
     - On-device Hugging Face causal LMs with optional prefix-cache metrics.

HTTP API backend
----------------

:class:`~nlp_shap.backends.api.ApiBackend` posts chat-completions requests to
``{api_host}/chat/completions``. Configure the base URL with ``backend.api_host``
or the ``OPENAI_BASE_URL`` environment variable (default
``http://127.0.0.1:1234/v1`` for local OpenAI-compatible servers).

Authentication uses ``OPENAI_API_KEY`` or ``NLP_SHAP_API_KEY`` when present.

At ``temperature: 0.0`` the backend caches responses keyed by the canonical JSON
request body so repeated identical coalition prompts avoid duplicate HTTP calls.
Scheduler-level dedup still applies for coalition keys across the full explain run.

Example configuration:

.. code-block:: yaml

   backend:
     kind: api
     model_id: gpt-4o-mini
     api_host: https://api.openai.com/v1

   generation:
     temperature: 0.0

Install the optional dependency:

.. code-block:: bash

   pip install "nlp-shap[api]"

Contract tests in CI mock the HTTP transport; point ``api_host`` at a live server
only for manual smoke runs.

Further reading
---------------

- Configuration schema: :doc:`config`
- End-to-end pipeline: :doc:`pipeline`
- API module reference: :doc:`../api/backends`
