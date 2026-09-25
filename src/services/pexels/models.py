from urllib.parse import urlparse

from pydantic import BaseModel, Field


class PexelsUser(BaseModel):
    id: int
    name: str
    url: str


class PexelsVideoFile(BaseModel):
    id: int
    quality: str | None = None
    file_type: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    size: int | None = None
    link: str


class PexelsVideo(BaseModel):
    id: int
    width: int
    height: int
    duration: int
    url: str
    image: str | None = None
    tags: list[str] = Field(default_factory=list)
    user: PexelsUser
    video_files: list[PexelsVideoFile]

    @property
    def title(self) -> str:
        """Pexels has no title field; the page slug is a human description ('dogs-with-their-tongues-out-5340598')."""
        slug = urlparse(self.url).path.rstrip("/").split("/")[-1]
        words = [w for w in slug.split("-") if w and w != str(self.id)]
        return " ".join(words).capitalize() if words else f"Pexels video {self.id}"

    def best_file(self, max_height: int) -> PexelsVideoFile | None:
        """Largest mp4 rendition at or below `max_height`; falls back to the smallest mp4 if all are larger."""
        mp4s = [f for f in self.video_files if (f.file_type or "").endswith("mp4") and f.height]
        if not mp4s:
            return None
        fitting = [f for f in mp4s if f.height <= max_height]
        if fitting:
            return max(fitting, key=lambda f: (f.height, f.fps or 0))
        return min(mp4s, key=lambda f: f.height)


class PexelsSearchResponse(BaseModel):
    page: int
    per_page: int
    total_results: int
    videos: list[PexelsVideo]
    next_page: str | None = None
