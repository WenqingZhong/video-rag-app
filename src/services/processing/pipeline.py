"""Turns one stored video into search segments. Runs inside the Celery worker."""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.models import SegmentKind, VideoStatus
from src.repositories import VideoRepository
from src.services.processing import ffmpeg
from src.services.processing.segmentation import build_shots, build_speech_windows
from src.services.processing.transcription import Transcriber
from src.services.storage import StorageClient

logger = logging.getLogger(__name__)


def frames_prefix(video_id: str) -> str:
    return f"frames/{video_id}/"


class VideoPipeline:
    """download → probe → scenes → keyframes → transcript → segments.

    Idempotent: re-running for the same video deletes its old frames and replaces its segments,
    so Celery redeliveries (acks_late) and manual re-processing are safe.
    """

    def __init__(
        self, database: BaseDatabase, storage: StorageClient, settings: Settings, transcriber_factory: Callable[[], Transcriber]
    ):
        self.database = database
        self.storage = storage
        self.settings = settings
        self._transcriber_factory = transcriber_factory

    def _set_stage(self, video_id: str, stage: str) -> None:
        with self.database.get_session() as session:
            VideoRepository(session).set_status(video_id, VideoStatus.PROCESSING, stage=stage)
            session.commit()
        logger.info("video %s: %s", video_id, stage)

    def process(self, video_id: str) -> dict:
        with self.database.get_session() as session:
            video = VideoRepository(session).get(video_id)
            if video is None:
                raise ValueError(f"video {video_id} not found")
            if not video.s3_key:
                raise ValueError(f"video {video_id} has no stored file")
            s3_key = video.s3_key

        with tempfile.TemporaryDirectory(prefix=f"video-{video_id[:8]}-") as tmp:
            workdir = Path(tmp)

            self._set_stage(video_id, "downloading_source")
            source = self.storage.download_file(s3_key, workdir / Path(s3_key).name)

            self._set_stage(video_id, "probing")
            meta = ffmpeg.probe(source)
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

            self._set_stage(video_id, "detecting_scenes")
            cuts = ffmpeg.detect_scene_changes(source, self.settings.scene_threshold)
            shots = build_shots(
                cuts, meta.duration_sec, self.settings.visual_min_segment_sec, self.settings.visual_max_segment_sec
            )

            self._set_stage(video_id, "extracting_keyframes")
            self.storage.delete_prefix(frames_prefix(video_id))
            visual_segments = []
            for idx, shot in enumerate(shots):
                frame_time = min(shot.mid, max(meta.duration_sec - 0.05, 0))
                frame_path = ffmpeg.extract_frame(source, frame_time, workdir / f"frame_{idx:05d}.jpg", self.settings.frame_width)
                key = f"{frames_prefix(video_id)}{int(frame_time * 1000):09d}.jpg"
                self.storage.upload_file(frame_path, key, content_type="image/jpeg")
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

            speech_segments, language = [], None
            if meta.has_audio:
                self._set_stage(video_id, "transcribing")
                audio = ffmpeg.extract_audio(source, workdir / "audio.wav")
                transcript = self._transcriber_factory().transcribe(audio)
                language = transcript.language
                windows = build_speech_windows(transcript.words, self.settings.speech_window_sec, self.settings.speech_stride_sec)
                speech_segments = [{"kind": SegmentKind.SPEECH, "idx": i, **window} for i, window in enumerate(windows)]

        self._set_stage(video_id, "saving_segments")
        with self.database.get_session() as session:
            repo = VideoRepository(session)
            count = repo.replace_segments(video_id, visual_segments + speech_segments)
            repo.update(video_id, language=language)
            repo.set_status(video_id, VideoStatus.READY)
            session.commit()

        summary = {
            "video_id": video_id,
            "duration_sec": meta.duration_sec,
            "visual_segments": len(visual_segments),
            "speech_segments": len(speech_segments),
            "segments": count,
        }
        logger.info("video %s ready: %s", video_id, summary)
        return summary
