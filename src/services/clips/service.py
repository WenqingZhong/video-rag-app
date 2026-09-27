"""Getting a clip: from the S3 cache if it exists, otherwise cut by the clip worker."""

from collections.abc import Callable

from src.services.clips.boundaries import ClipRange, clip_key
from src.services.storage import StorageClient
from src.services.tracing import span

# (video_id, start_sec, end_sec) -> S3 key of the cut clip. Injected: the API passes "run clip.cut on the
# clips queue and wait"; tests pass a fake.
CutFn = Callable[[str, float, float], str]


class ClipService:
    def __init__(self, storage: StorageClient, cut: CutFn):
        self.storage = storage
        self.cut = cut

    def get_or_cut(self, video_id: str, clip: ClipRange) -> tuple[str, bool]:
        """→ (S3 key, served from cache?)."""
        key = clip_key(video_id, clip)
        with span("clip", video_id=video_id, start_sec=clip.start_sec, end_sec=clip.end_sec) as step:
            cached = self.storage.exists(key)
            step.set(cached=cached)
            if cached:
                return key, True
            # the cut itself runs on the clip worker and appears in this trace as its own spans
            return self.cut(video_id, clip.start_sec, clip.end_sec), False
