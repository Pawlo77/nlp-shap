Using value functions
=====================

Value functions score coalition model outputs relative to a base generation.
They are separate from estimands (Shapley/Banzhaf) and estimators (sampling).

TF-IDF cosine (U3)
------------------

.. code-block:: python

   from nlp_shap import GenerationRecord, TfIdfCosineValue

   base = GenerationRecord(
       text="hello world",
       text_token_rows=((1, 0), (0, 1)),
   )
   candidate = GenerationRecord(
       text="hello there",
       text_token_rows=((1, 0), (2, 3)),
   )

   value_fn = TfIdfCosineValue()
   value_fn.fit((base, candidate))
   utility = value_fn.score(base, candidate)
   print(round(utility, 4))

.. guide-result:: value_tfidf

Call :meth:`~nlp_shap.value.tfidf.TfIdfCosineValue.fit` once at run start to
freeze inverse document frequency weights for the full archive corpus.

Embedding utilities (U1/U2/U4)
------------------------------

.. code-block:: python

   from nlp_shap import CosineEmbeddingValue, GenerationRecord
   from nlp_shap.domain.enums import EmbeddingMode

   base = GenerationRecord(text="base", embedding=(1.0, 0.0))
   candidate = GenerationRecord(text="candidate", embedding=(0.9, 0.1))

   value_fn = CosineEmbeddingValue(embedding_mode=EmbeddingMode.STATIC)
   similarity = value_fn.score(base, candidate)
   print(round(similarity, 4))

.. guide-result:: value_embedding

Use ``embedding_euclidean`` for U4-style ``1 / (1 + distance)`` scoring.
Set ``embedding_mode=EmbeddingMode.CONTEXTUAL`` for U2 contextual embeddings.

Logprob utility
---------------

.. code-block:: python

   from nlp_shap import GenerationRecord, LogprobValue

   candidate = GenerationRecord(text="answer", logprobs=(-0.1, -0.2, -0.3))
   utility = LogprobValue().score(candidate, candidate)
   print(round(utility, 4))

.. guide-result:: value_logprob

When logprobs are absent, :class:`~nlp_shap.value.logprob.LogprobValue` falls
back to a deterministic stub so reanalysis paths can swap utilities without
new backend calls.

Attribution normalizers
-----------------------

Normalizers scale aggregated attributions for presentation. They do **not**
change archived coalition utilities.

.. code-block:: python

   from nlp_shap import MinMaxNormalizer

   raw = [1.0, -2.0, 3.0]
   displayed = MinMaxNormalizer().normalize(raw)
   print(displayed)

.. guide-result:: value_normalizer

Config keys: ``identity``, ``abs_sum``, ``power_shift``, ``min_max``. See
:doc:`config` and :doc:`../theory/value_functions`.
