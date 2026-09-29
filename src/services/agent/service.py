"""A chat turn end to end: load the conversation → run the graph → remember → answer."""

import time
from dataclasses import dataclass, field

from src.services.agent.decide import LLMRouter
from src.services.agent.graph import ChatGraph
from src.services.agent.memory import ConversationStore
from src.services.agent.tools import Toolbox
from src.services.tracing import span
from src.services.usage import UsageRecorder


@dataclass
class ChatTurn:
    conversation_id: str
    reply: str
    action: str
    decided_by: str  # "rules" (a clear pattern, or the model's choice was rejected) | "llm"
    notes: list[str] = field(default_factory=list)
    clips: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    videos: list[dict] = field(default_factory=list)
    fetching: list[str] = field(default_factory=list)
    uploading: str | None = None  # the id of a video the user just uploaded (still processing)
    seconds: float = 0.0


@dataclass
class ChatUpdate:
    conversation_id: str
    status: str  # "idle" (nothing downloading) | "pending" | "done" | "timed_out"
    reply: str | None = None  # "done" / "timed_out": the message to show
    clips: list[dict] = field(default_factory=list)
    progress: dict[str, int] = field(default_factory=dict)  # ready / pending / failed
    status_text: str | None = None  # an upload in progress: "Step 6 of 9: describing the frames (3 of 11) · 0:42"


class ChatService:
    def __init__(
        self,
        toolbox: Toolbox,
        store: ConversationStore,
        router: LLMRouter | None,
        recorder: UsageRecorder | None,
        fetch_timeout_sec: float = 600,
    ):
        self.graph = ChatGraph(toolbox, router, recorder, fetch_timeout_sec)
        self.store = store

    def poll(self, conversation_id: str) -> ChatUpdate:
        """Anything new? After a Pexels download: the clip once the videos are processed (clients call every few s)."""
        conversation = self.store.load(conversation_id)
        state = self.graph.poll(conversation)
        if state["update"] != "idle":
            self.store.save(conversation)
        return ChatUpdate(
            conversation_id=conversation.id,
            status=state["update"],
            reply=state.get("reply"),
            clips=state.get("clips", []),
            progress=state.get("progress", {}),
            status_text=state.get("status_text"),
        )

    def turn(
        self, conversation_id: str | None, message: str, image_key: str | None = None, video: tuple | None = None
    ) -> ChatTurn:
        """`video`: (file, filename, content type, size) of a video attached to the message."""
        started = time.perf_counter()
        conversation = self.store.load(conversation_id)
        with span("agent.turn", conversation_id=conversation.id, turn=conversation.turns + 1) as step:
            state = self.graph.run(conversation, message, image_key, video)
            decision = state["decision"]
            step.set(action=decision.action, decided_by=decision.source)
        self.store.save(conversation)
        return ChatTurn(
            conversation_id=conversation.id,
            reply=state.get("reply", ""),
            action=decision.action,
            decided_by=decision.source,
            notes=decision.notes,
            clips=state.get("clips", []),
            citations=state.get("citations", []),
            videos=state.get("videos", []),
            fetching=conversation.fetching if decision.action == "fetch_from_pexels" else [],
            # also when a request is waiting for it: clients (re)start polling until it's delivered
            uploading=conversation.uploading if video is not None or state.get("queued") else None,
            seconds=round(time.perf_counter() - started, 3),
        )
