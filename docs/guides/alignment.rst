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
