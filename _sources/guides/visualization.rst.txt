Visualization
=============

Per-token Shapley and Banzhaf attributions are numeric tuples in
:class:`~nlp_shap.pipeline.result.ExplainRunOutput`. Renderer plugins turn those
values into review-ready figures for notebooks, reports, and stakeholder decks.

Renderers
---------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Plugin key
     - Output
   * - ``token_text``
     - SHAP-style inline colored tokens (diverging red/blue scale)
   * - ``token_bar``
     - Horizontal bar chart sorted by absolute attribution

Install the optional visualization extra:

.. code-block:: bash

   pip install "nlp-shap[viz]"

Quick start
-----------

.. code-block:: python

   from nlp_shap import ExplainRunner, ExplainConfig, render_attribution
   from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
   from nlp_shap.domain.enums import Role
   from nlp_shap.masking.partitions import TokenPartitioner

   snapshot = ConversationSnapshot.from_turns((
       Turn(messages=(Message(role=Role.USER, text="refund within thirty days"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "estimator": "exact",
           "estimand": "shapley",
           "value_fn": "tfidf_cosine",
           "absence_policy": "pad",
       },
   })
   output = ExplainRunner(config).explain_sync(snapshot)
   player_set = TokenPartitioner().partition(snapshot)

   figure = render_attribution(output, snapshot, player_set, renderer="token_bar")
   figure.savefig("attribution.png", dpi=150, bbox_inches="tight")

Jupyter HTML
------------

For inline notebook display, use :func:`~nlp_shap.render_attribution_html` or
:func:`~nlp_shap.viz.display_attribution_html` when IPython is available.

Color scale
-----------

Positive attributions trend red; negative attributions trend blue. Color
intensity scales with the largest absolute value in the rendered vector so
small-magnitude tokens remain readable.

Further reading
---------------

- End-to-end explain notebook: :doc:`../examples`
- Generative backends: :doc:`backends`
- API reference: :doc:`../api/viz`
