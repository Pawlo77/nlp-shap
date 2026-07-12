Masking views
=============

This guide shows how to partition a conversation into token players, build
shared masked views, and render coalition-specific prompts.

Partition tokens
----------------

.. code-block:: python

   from nlp_shap import ConversationSnapshot, Message, Role, Turn
   from nlp_shap.masking import TokenPartitioner

   turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
   snapshot = ConversationSnapshot.from_turns((turn,))

   players = TokenPartitioner().partition(snapshot)
   print(players.player_ids)
   # ('<snapshot_id>:0:0:0', '<snapshot_id>:0:0:1', '<snapshot_id>:0:0:2')

Render a coalition
------------------

.. code-block:: python

   from nlp_shap import CoalitionMask
   from nlp_shap.masking import DeletePolicy, MaskBuilder

   mask = CoalitionMask.from_sequence((True, False, True))
   builder = MaskBuilder(DeletePolicy())
   view = builder.view(snapshot, players, mask)
   rendered = builder.render(view)

   print(rendered.turns[0].messages[0].text)
   # Who you?

Compare absence policies
------------------------

.. code-block:: python

   from nlp_shap.masking import NeutralPolicy, PadPolicy

   pad_view = MaskBuilder(PadPolicy()).view(snapshot, players, mask)
   neutral_view = MaskBuilder(NeutralPolicy()).view(snapshot, players, mask)

   print(MaskBuilder(PadPolicy()).render(pad_view).turns[0].messages[0].text)
   # Who [MASK] you?

   print(MaskBuilder(NeutralPolicy()).render(neutral_view).turns[0].messages[0].text)
   # Who ___ you?

Resolve plugins from config
---------------------------

Built-in plugins register the ``tokens`` partition and ``delete`` / ``pad`` /
``neutral`` absence policies. Load them through :class:`~nlp_shap.PluginRegistry`:

.. code-block:: python

   from nlp_shap import ExplainConfig, PluginGroup, PluginRegistry
   from nlp_shap.plugins import register_builtin_plugins

   registry = PluginRegistry()
   register_builtin_plugins(registry)

   config = ExplainConfig.model_validate(
       {
           "backend": {"kind": "mock", "model_id": "stub"},
           "explanation": {"players": "tokens", "absence_policy": "pad"},
       }
   )

   partitioner = registry.resolve(PluginGroup.PARTITIONS, config.explanation.players)
   policy = registry.resolve(
       PluginGroup.ABSENCE_POLICIES,
       config.explanation.absence_policy,
   )

See :doc:`../guides/config` for the full configuration schema.

Notebook
--------

For a runnable walkthrough with stored outputs, see
``examples/masking_views.ipynb`` (also embedded on :doc:`../examples`).
