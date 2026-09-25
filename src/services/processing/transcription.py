import logging
from dataclasses import dataclass
from pathlib import Path

from src.services.processing.segmentation import Word

logger = logging.getLogger(__name__)


@dataclass
class Transcript:
    language: str | None
    words: list[Word]


class Transcriber:
    """faster-whisper wrapper. The model is loaded on first use and reused for the life of the worker process."""

    def __init__(self, model_size: str, device: str = "cpu", compute_type: str = "int8", beam_size: int = 5):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import; only the worker needs it

            logger.info("Loading whisper model '%s' (%s, %s)", self.model_size, self.device, self.compute_type)
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: Path) -> Transcript:
        model = self._load()
        segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,  # skip silence/music: fewer hallucinated words
            beam_size=self.beam_size,
        )
        # float(): faster-whisper returns numpy floats, which database drivers and JSON can't serialise
        words = [
            Word(word=w.word, start=round(float(w.start), 3), end=round(float(w.end), 3), prob=round(float(w.probability), 3))
            for segment in segments
            for w in (segment.words or [])
        ]
        logger.info("Transcribed %s words (language=%s, p=%.2f)", len(words), info.language, info.language_probability)
        return Transcript(language=info.language if words else None, words=words)
