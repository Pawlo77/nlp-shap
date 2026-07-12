Masking
=======

Coalition masking turns a :class:`~nlp_shap.ConversationSnapshot` and a
:class:`~nlp_shap.CoalitionMask` into a coalition-specific prompt view. The
masking layer separates three concerns:

1. **Partitioning** — which text units are explainability players.
2. **Absence policy** — how absent players appear in the rendered prompt.
3. **Codec** — stable packed representation for deduplication keys.

Partitioning
------------

:class:`~nlp_shap.masking.partitions.TokenPartitioner` splits each message into
whitespace-delimited tokens. Each token becomes one player with a stable
identifier derived from the base snapshot id and token coordinates.

Absence policies
----------------

Three policies ship in ``0.1.3``:

Delete
~~~~~~

Absent tokens are removed from the rendered text. This matches the default
``explanation.absence_policy: delete`` setting in :class:`~nlp_shap.ExplainConfig`
and changes later token positions when a model consumes the prompt sequentially.

Pad
~~~

Absent tokens are replaced with a fixed placeholder (default ``[MASK]``).
Position shifts are avoided relative to deletion, but the placeholder must be
valid for the downstream tokenizer and model.

Neutral
~~~~~~~

Absent tokens are replaced with width-matched neutral fillers (default ``_`` per
character). This preserves string length while signalling absence without
introducing a special mask token.

Structural sharing
------------------

:class:`~nlp_shap.masking.builder.MaskedSnapshot` references the original
snapshot object. Many coalition views therefore share the same
``snapshot_id`` and base conversation payload; only the coalition mask and
policy metadata differ until :class:`~nlp_shap.masking.builder.MaskBuilder`
materializes a rendered snapshot.

Packed masks
------------

:class:`~nlp_shap.masking.codec.MaskCodec` packs boolean masks into little-endian
bytes and exposes a stable integer hash compatible with the legacy MLLM-Shap
codec. :class:`~nlp_shap.masking.space.MaskSpace` projects feature-level splits
back into full-length masks when some positions are always present.

See also
--------

- :doc:`../guides/masking` — usage examples
- :doc:`../api/masking` — module reference
- :doc:`../guides/config` — ``players`` and ``absence_policy`` settings
