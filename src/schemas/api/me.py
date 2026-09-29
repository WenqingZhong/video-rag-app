from pydantic import BaseModel


class UsageOut(BaseModel):
    tokens_used: int
    tokens_limit: int
    tokens_left: int
    uploads_used: int
    uploads_limit: int
    pexels_used: int
    pexels_limit: int
    resets_in_sec: int
