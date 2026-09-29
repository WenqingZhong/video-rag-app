from fastapi import APIRouter

from src.dependencies import LimiterDep, SettingsDep, ViewerDep
from src.schemas.api.me import UsageOut

router = APIRouter(prefix="/me", tags=["Me"])


@router.get("/usage", response_model=UsageOut)
def my_usage(viewer: ViewerDep, limiter: LimiterDep, settings: SettingsDep) -> UsageOut:
    """Today's allowance: tokens, uploads and Pexels downloads used and left (resets at midnight UTC)."""
    if limiter is None:
        return UsageOut(tokens_used=0, tokens_limit=settings.limit_tokens_per_day, tokens_left=settings.limit_tokens_per_day,
                        uploads_used=0, uploads_limit=settings.limit_uploads_per_day, pexels_used=0,
                        pexels_limit=settings.limit_pexels_per_day, resets_in_sec=0)  # fmt: skip
    usage = limiter.usage(viewer)
    return UsageOut(**vars(usage), tokens_left=usage.tokens_left)
