Audio alignment
===============

Use :class:`~nlp_shap.alignment.sgpa.SpectrogramGuidedAligner` to force-align a
transcript to waveform bytes with Wav2Vec2 CTC and spectrogram boundary
refinement. Install the audio extra first:

.. code-block:: bash

   pip install "nlp-shap[audio]"

Minimal usage (device can be a ``torch.device`` or a string such as ``"cpu"``):

.. code-block:: python

   from nlp_shap.alignment import SpectrogramGuidedAligner

   aligner = SpectrogramGuidedAligner(device="cpu")
   segments = aligner(
       transcript="hello world",
       audio_content=open("utterance.wav", "rb").read(),
       audio_format="wav",
       attach_audio=False,
   )
   for segment in segments:
       print(segment.token, segment.start_time, segment.end_time, segment.confidence)

Each :class:`~nlp_shap.alignment.segments.AudioSegment` carries optional sample
indices and WAV bytes when ``attach_audio=True``. See :doc:`../api/alignment`
for the full module reference.

Pass a token filter to skip punctuation (or a custom exclude set) when splitting
the transcript into alignment targets:

.. code-block:: python

   from nlp_shap.alignment import SpectrogramGuidedAligner
   from nlp_shap.masking import ExcludePunctuationTokensFilter

   aligner = SpectrogramGuidedAligner(
       device="cpu",
       token_filter=ExcludePunctuationTokensFilter(),
   )

SGPA segment players
--------------------

After alignment, build coalition players with the ``sgpa_segments`` partition
plugin (:class:`~nlp_shap.alignment.partitions.SgpaSegmentPartitioner`) and
render absences with :class:`~nlp_shap.masking.policies.SegmentDeletePolicy`:

.. code-block:: python

   from nlp_shap import AudioPayload, CoalitionMask, ConversationSnapshot, Message, ModalityFlag, Role, Turn
   from nlp_shap.alignment import SgpaSegmentPartitioner
   from nlp_shap.masking import MaskBuilder, SegmentDeletePolicy

   snapshot = ConversationSnapshot.from_turns((
       Turn(
           messages=(
               Message(
                   role=Role.USER,
                   text="hello world",
                   modality=ModalityFlag.AUDIO,
                   audio=AudioPayload(data=open("utterance.wav", "rb").read(), sample_rate_hz=16_000, audio_format="wav"),
               ),
           )
       ),
   ))
   segments = tuple(aligner(transcript="hello world", audio_content=snapshot.turns[0].messages[0].audio.data))
   partitioner = SgpaSegmentPartitioner(segments=segments)
   players = partitioner.partition(snapshot)
   mask = CoalitionMask.from_sequence((True,) + (False,) * (players.num_players - 1))
   policy = SegmentDeletePolicy(segments=partitioner.filtered_segments())
   view = MaskBuilder(policy).view(snapshot, players, mask)
