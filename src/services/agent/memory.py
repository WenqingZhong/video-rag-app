"""What the agent remembers between turns of one conversation, in Redis (24 h after the last turn).

Deliberately small: the router is a 3B model, and only the last request, the last clip and an open offer matter
for follow-ups ("now without people", "yes, fetch them", "what colour is the dog in that clip?").
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field

import redis

from src.services.agent.decide import TurnContext
from src.services.cache import CacheClient

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    id: str
    last_request: str | None = None
    last_status: str | None = None
    last_action: str | None = None  # the last tool used: follow-ups ("what about X?") repeat the same kind
    last_video: str | None = None  # title, for the router
    last_video_id: str | None = None  # id, to scope follow-up questions ("in that clip")
    offered_fetch: str | None = None  # a Pexels download we offered; cleared on the next turn
    fetching: list[str] = field(default_factory=list)  # video ids being downloaded/processed for this conversation
    fetch_request: str | None = None  # the request to retry once they're ready
    fetch_subject: str | None = None  # what was downloaded ("horse"), for the messages
    fetch_started: float | None = None  # epoch seconds: gives up after a while
    uploading: str | None = None  # a video the user uploaded in this chat, still processing
    upload_name: str | None = None
    upload_request: str | None = None  # a request sent with the upload, or while it's processing: run once ready
    upload_request_action: str | None = None  # "find_clip" | "answer_question"
    upload_started: float | None = None
    focus_video_id: str | None = None  # the user's own video: searched first, then the library
    focus_title: str | None = None
    shown: list[str] = field(default_factory=list)  # clips already shown ("video_id@start"): "another one" skips them
    turns: int = 0

    def context(self, image_attached: bool) -> TurnContext:
        return TurnContext(
            self.last_request, self.last_status, self.last_action, self.last_video, self.offered_fetch, image_attached
        )


class ConversationStore:
    def __init__(self, cache: CacheClient, ttl_seconds: int = 24 * 3600):
        self.cache = cache
        self.ttl = ttl_seconds

    def load(self, conversation_id: str | None) -> Conversation:
        """A new conversation when there's no id, it expired, or Redis is down (the turn still works, without memory)."""
        if conversation_id:
            try:
                raw = self.cache.get(f"chat:{conversation_id}")
            except redis.RedisError as exc:
                logger.warning("conversation memory unavailable: %s", exc)
                raw = None
            if raw:
                return Conversation(**json.loads(raw))
        return Conversation(id=conversation_id or uuid.uuid4().hex)

    def save(self, conversation: Conversation) -> None:
        conversation.shown = conversation.shown[-50:]
        try:
            self.cache.set(f"chat:{conversation.id}", json.dumps(asdict(conversation)), ttl_seconds=self.ttl)
        except redis.RedisError as exc:
            logger.warning("conversation not saved: %s", exc)
