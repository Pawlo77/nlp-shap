Runtime core
============

This guide shows how to persist coalition history, deduplicate repeated masks,
and schedule async coalition generation with bounded concurrency.

Open a run archive
------------------

.. code-block:: python

   from pathlib import Path

   from nlp_shap import Estimand
   from nlp_shap.masking.codec import MaskCodec
   from nlp_shap.pipeline.manifest import RunManifest
   from nlp_shap.runtime import CoalitionRecordDraft, RunArchive

   manifest = RunManifest(estimand=Estimand.SHAPLEY, run_id="demo-run")
   root = Path("./runs/demo-run")

   with RunArchive.open(root, manifest, flush_every=25) as archive:
       packed = MaskCodec.pack((True, False, True))
       archive.append(
           CoalitionRecordDraft(
               snapshot_id="snap-1",
               coalition_key="coalition-1",
               mask=packed,
               absence_policy="delete",
               model_id="mock",
               generation_text="Who you?",
               utility=0.8,
               elapsed_ms=12.5,
               cache_hit=False,
           )
       )

   with RunArchive.open(root, manifest) as archive:
       for record in archive.history_lazy():
           print(record.record_id, record.generation_text)

Build coalition dedup keys
--------------------------

.. code-block:: python

   from nlp_shap.pipeline.config import DedupConfig, GenerationConfig
   from nlp_shap.runtime import build_coalition_key, dedup_enabled

   generation = GenerationConfig(temperature=0.0)
   key = build_coalition_key(
       snapshot_id="snap-1",
       player_ids=("snap-1:0:0:0", "snap-1:0:0:1"),
       mask_present=(True, False),
       absence_policy="delete",
       model_id="mock",
       generation=generation,
   )
   print(dedup_enabled(DedupConfig(enabled="auto"), generation), key[:12])

Schedule async coalition jobs
-----------------------------

.. code-block:: python

   import asyncio

   from nlp_shap.domain.conversation import ConversationSnapshot, Message, Role, Turn
   from nlp_shap.masking.codec import MaskCodec
   from nlp_shap.pipeline.config import GenerationConfig
   from nlp_shap.runtime import (
       CoalitionDedupRegistry,
       CoalitionJob,
       HotResultStore,
       InferenceScheduler,
   )

   turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
   snapshot = ConversationSnapshot.from_turns((turn,))
   packed = MaskCodec.pack((True, False, True))

   async def generate(snapshot: ConversationSnapshot) -> str:
       return snapshot.turns[0].messages[0].text

   scheduler = InferenceScheduler(
       max_inflight=2,
       generation=GenerationConfig(temperature=0.0),
       store=HotResultStore(),
       dedup=CoalitionDedupRegistry(),
   )
   jobs = [
       CoalitionJob(
           coalition_key=f"key-{index % 3}",
           snapshot_id=snapshot.snapshot_id,
           snapshot=snapshot,
           absence_policy="delete",
           mask_words=packed.words,
           mask_n_bits=packed.n_bits,
           model_id="mock",
           utility=1.0,
       )
       for index in range(9)
   ]
   metrics = asyncio.run(scheduler.run(jobs, generate))
   print(metrics)

Notebook
--------

For a runnable walkthrough with stored outputs, see
``examples/runtime_core.ipynb`` (also embedded on :doc:`../examples`).
