from src.config import Settings
from src.services.pexels.client import PexelsClient


def make_pexels_client(settings: Settings) -> PexelsClient:
    return PexelsClient(api_key=settings.pexels_api_key, base_url=settings.pexels_base_url)
