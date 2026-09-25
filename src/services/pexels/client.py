import logging
import time

import httpx

from src.services.pexels.models import PexelsSearchResponse, PexelsVideo

logger = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class PexelsError(RuntimeError):
    pass


class PexelsClient:
    """Minimal Pexels Videos API client with retries and rate-limit awareness."""

    def __init__(self, api_key: str, base_url: str = "https://api.pexels.com", timeout: float = 15.0, max_retries: int = 3):
        if not api_key:
            raise PexelsError("PEXELS_API_KEY is not set")
        self.max_retries = max_retries
        self.http = httpx.Client(base_url=base_url, headers={"Authorization": api_key}, timeout=timeout)
        self.rate_limit_remaining: int | None = None

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.get(path, params=params)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise PexelsError(f"Pexels unreachable: {exc}") from exc
                time.sleep(2**attempt)
                continue

            remaining = response.headers.get("x-ratelimit-remaining")
            if remaining is not None:
                self.rate_limit_remaining = int(remaining)

            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                wait = 2**attempt
                logger.warning("Pexels %s returned %s; retrying in %ss", path, response.status_code, wait)
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                raise PexelsError(f"Pexels {path} failed: HTTP {response.status_code} {response.text[:200]}")
            return response.json()
        raise PexelsError(f"Pexels {path} failed after {self.max_retries} retries")

    def search_videos(
        self,
        query: str,
        per_page: int = 15,
        page: int = 1,
        orientation: str | None = None,
        size: str | None = None,
    ) -> PexelsSearchResponse:
        params = {"query": query, "per_page": min(per_page, 80), "page": page}
        if orientation:
            params["orientation"] = orientation
        if size:
            params["size"] = size
        return PexelsSearchResponse.model_validate(self._get("/videos/search", params))

    def get_video(self, video_id: int) -> PexelsVideo:
        return PexelsVideo.model_validate(self._get(f"/videos/videos/{video_id}"))

    def close(self) -> None:
        self.http.close()
