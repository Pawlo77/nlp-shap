End-to-end explain pipeline
============================

The explain pipeline wires masking, runtime, estimation, and value-function plugins
into a single run.

Quick start
-----------

.. code-block:: python

   from nlp_shap import ExplainConfig, ExplainRunner
   from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
   from nlp_shap.domain.enums import Role

   snapshot = ConversationSnapshot.from_turns((
       Turn(messages=(Message(role=Role.USER, text="hello world"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "estimator": "exact",
           "estimand": "shapley",
           "value_fn": "tfidf_cosine",
           "normalizer": "identity",
       },
   })

   output = ExplainRunner(config).explain_sync(snapshot)
   print(output.result.values)
   print(output.metrics)

The mock backend is deterministic and requires no GPU or optional extras. Use
``pad`` absence policy when the estimator samples the empty coalition.

For remote or local model servers see :doc:`backends`.

See :doc:`config` for the full configuration schema.
