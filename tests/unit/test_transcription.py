import json
from types import SimpleNamespace

import numpy as np

from src.services.processing.transcription import Transcriber


def test_transcribe_returns_plain_floats(tmp_path):
    """faster-whisper yields numpy floats; they must not leak into DB rows / JSON."""
    word = SimpleNamespace(word=" AI", start=np.float64(1.23456), end=np.float64(1.5), probability=np.float32(0.98765))
    fake_model = SimpleNamespace(
        transcribe=lambda *a, **k: ([SimpleNamespace(words=[word])], SimpleNamespace(language="en", language_probability=0.99))
    )
    transcriber = Transcriber("tiny")
    transcriber._model = fake_model

    transcript = transcriber.transcribe(tmp_path / "audio.wav")

    w = transcript.words[0]
    assert (type(w.start), type(w.end), type(w.prob)) == (float, float, float)
    assert (w.start, w.end, w.prob) == (1.235, 1.5, 0.988)
    json.dumps(w.__dict__)  # serialisable
    assert transcript.language == "en"
