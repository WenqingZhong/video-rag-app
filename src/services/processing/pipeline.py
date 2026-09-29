"""Turns one stored video into search segments. Runs inside the Celery worker."""

import logging
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.models import SegmentKind, VideoStatus
from src.repositories import VideoRepository
from src.services.limits import Principal, acting_for
from src.services.processing import ffmpeg
from src.services.processing.segmentation import build_shots, build_speech_windows
from src.services.processing.transcription import Transcriber
from src.services.processing.visual import VisualEnricher
from src.services.storage import StorageClient
from src.services.tracing import Span, span

logger = logging.getLogger(__name__)


class VideoTooLong(ValueError):
    """A user's upload over UPLOAD_MAX_DURATION_SEC. Not retried."""


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def frames_prefix(video_id: str) -> str:
    return f"frames/{video_id}/"


class VideoPipeline:
    """download → probe → scenes → keyframes → embeddings + captions → transcript → segments → search index.

    Idempotent: re-running for the same video deletes its old frames and replaces its segments,
    so Celery redeliveries (acks_late) and manual re-processing are safe.
    """

    def __init__(
        self,
        database: BaseDatabase,
        storage: StorageClient,
        settings: Settings,
        transcriber_factory: Callable[[], Transcriber],
        indexer: Callable[[str], int] | None = None,
        enricher: VisualEnricher | None = None,
    ):
        self.database = database
        self.storage = storage
        self.settings = settings
        self._transcriber_factory = transcriber_factory
        self._indexer = indexer  # video_id -> documents indexed; None = skip (e.g. tests)
        self._enricher = enricher  # keyframes -> CLIP vectors + captions; None = skip

    def _set_stage(self, video_id: str, stage: str) -> None:
        with self.database.get_session() as session:
            VideoRepository(session).set_status(video_id, VideoStatus.PROCESSING, stage=stage)
            session.commit()
        logger.info("video %s: %s", video_id, stage)

    @contextmanager
    def _stage(self, video_id: str, stage: str, **attributes) -> Iterator[Span]:
        """Show the stage on the video row (for GET /videos/{id}) and time it as a span of the task's trace."""
        self._set_stage(video_id, stage)
        with span(f"stage.{stage}", **attributes) as step:
            yield step

    def process(self, video_id: str) -> dict:
        with self.database.get_session() as session:
            video = VideoRepository(session).get(video_id)
            if video is None:
                raise ValueError(f"video {video_id} not found")
            if not video.s3_key:
                raise ValueError(f"video {video_id} has no stored file")
            s3_key, owner = video.s3_key, video.owner_id
        # Model calls made while processing (captions) count against the uploader's daily tokens.
        with acting_for(Principal(viewer=owner)):
            return self._process(video_id, s3_key, owner)

    def _process(self, video_id: str, s3_key: str, owner: str | None) -> dict:
        with tempfile.TemporaryDirectory(prefix=f"video-{video_id[:8]}-") as tmp:
            workdir = Path(tmp)

            with self._stage(video_id, "downloading_source"):
                source = self.storage.download_file(s3_key, workdir / Path(s3_key).name)

            with self._stage(video_id, "probing") as step:
                meta = ffmpeg.probe(source)
                step.set(duration_sec=meta.duration_sec, has_audio=meta.has_audio)
                longest = self.settings.upload_max_duration_sec
                if owner is not None and meta.duration_sec > longest:  # before any model call: nothing is spent
                    raise VideoTooLong(f"it's {mmss(meta.duration_sec)} long, and uploads can be up to {mmss(longest)}")
                with self.database.get_session() as session:
                    VideoRepository(session).update(
                        video_id,
                        duration_sec=meta.duration_sec,
                        width=meta.width,
                        height=meta.height,
                        fps=meta.fps,
                        has_audio=meta.has_audio,
                    )
                    session.commit()

            with self._stage(video_id, "detecting_scenes") as step:
                cuts = ffmpeg.detect_scene_changes(source, self.settings.scene_threshold)
                shots = build_shots(
                    cuts, meta.duration_sec, self.settings.visual_min_segment_sec, self.settings.visual_max_segment_sec
                )
                step.set(shots=len(shots))

            with self._stage(video_id, "extracting_keyframes"):
                self.storage.delete_prefix(frames_prefix(video_id))
                visual_segments, frame_images = [], []
                for idx, shot in enumerate(shots):
                    frame_time = min(shot.mid, max(meta.duration_sec - 0.05, 0))
                    frame_path = ffmpeg.extract_frame(
                        source, frame_time, workdir / f"frame_{idx:05d}.jpg", self.settings.frame_max_side
                    )
                    key = f"{frames_prefix(video_id)}{int(frame_time * 1000):09d}.jpg"
                    self.storage.upload_file(frame_path, key, content_type="image/jpeg")
                    frame_images.append(frame_path.read_bytes())
                    visual_segments.append(
                        {
                            "kind": SegmentKind.VISUAL,
                            "idx": idx,
                            "start_sec": shot.start,
                            "end_sec": shot.end,
                            "frame_key": key,
                            "frame_time_sec": round(frame_time, 3),
                        }
                    )

            if self._enricher is not None:
                enriched = self._enricher.enrich(
                    frame_images, on_stage=lambda stage: self._set_stage(video_id, stage), video_id=video_id
                )
                for segment, fields in zip(visual_segments, enriched, strict=True):
                    segment.update(fields)

            speech_segments, language = [], None
            if meta.has_audio:
                with self._stage(video_id, "transcribing") as step:
                    audio = ffmpeg.extract_audio(source, workdir / "audio.wav")
                    transcript = self._transcriber_factory().transcribe(audio)
                    language = transcript.language
                    windows = build_speech_windows(
                        transcript.words, self.settings.speech_window_sec, self.settings.speech_stride_sec
                    )
                    speech_segments = [{"kind": SegmentKind.SPEECH, "idx": i, **window} for i, window in enumerate(windows)]
                    step.set(words=len(transcript.words), language=language)

        with self._stage(video_id, "saving_segments"), self.database.get_session() as session:
            repo = VideoRepository(session)
            count = repo.replace_segments(video_id, visual_segments + speech_segments)
            repo.update(video_id, language=language)
            session.commit()

        # "ready" means searchable: index before flipping the status. Postgres stays the source of truth,
        # so if indexing fails the video is marked failed and a retry/reprocess/rebuild repairs it.
        indexed = 0
        if self._indexer is not None:
            with self._stage(video_id, "indexing") as step:
                indexed = self._indexer(video_id)
                step.set(documents=indexed)

        with self.database.get_session() as session:
            VideoRepository(session).set_status(video_id, VideoStatus.READY)
            session.commit()

        summary = {
            "video_id": video_id,
            "duration_sec": meta.duration_sec,
            "visual_segments": len(visual_segments),
            "speech_segments": len(speech_segments),
            "segments": count,
            "indexed": indexed,
        }
        logger.info("video %s ready: %s", video_id, summary)
        return summary
