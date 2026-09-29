"""A chat turn, or a check for news, as one LangGraph graph:

    message:  decide ──(reply)─────────────────────▶ respond ─────────▶ END
                 └──(a tool)──▶ act ──────────────▶ respond
    video:    upload (store + queue processing) ─────────────────────▶ END
    poll:     check_fetch ──(still processing, or ready with nothing to find)──▶ END
                 └──(processed, a request waiting)──▶ act ──▶ respond_fetched ──▶ END

decide           rules first; the model only when the rules can't decide; guards check it (decide.py)
act              runs one tool (tools.py)
respond          a fixed-template reply from the tool's result, and what to remember for the next turn
check_fetch      is the user's upload, or the Pexels download, processed yet? (gives up after a while)
respond_fetched  the clip from the new videos, or why there isn't one
upload           a video attached to a message: it becomes the conversation's focus once processed; requests then
                 search it first, and fall back to the library

Downloading and processing take minutes, so a turn never waits for them: clients poll (GET /chat/{id}/updates).
"""

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.services.agent import replies
from src.services.agent.decide import Decision, LLMRouter, decide
from src.services.agent.memory import Conversation
from src.services.agent.tools import Toolbox, ToolResult
from src.services.tracing import span
from src.services.usage import UsageRecorder


class TurnState(TypedDict, total=False):
    poll: bool  # a check for news, not a message
    video: tuple | None  # (file, filename, content type, size): a video attached to the message
    kind: str  # poll: "upload" | "fetch": what finished
    status_text: str  # poll, upload in progress: which step, for how long
    queued: str  # a request that must wait for the user's video: the reply saying so
    update: str  # poll only: "idle" (nothing fetching) | "pending" | "done" | "timed_out"
    progress: dict[str, int]  # poll only: ready / pending / failed
    message: str
    image_key: str | None
    conversation: Conversation
    decision: Decision
    result: ToolResult | None
    reply: str
    clips: list[dict]
    citations: list[dict]
    videos: list[dict]


def _clip_id(clip: dict) -> str:
    return f"{clip['video_id']}@{clip['start_sec']}"


def _clear_fetch(conversation: Conversation) -> None:
    conversation.fetching, conversation.fetch_request = [], None
    conversation.fetch_subject, conversation.fetch_started = None, None


def _route_outcome(decision: Decision) -> str:
    if decision.source == "llm":
        return "adjusted" if decision.notes else "accepted"
    return "invalid" if any("invalid" in n for n in decision.notes) else "rejected"


class ChatGraph:
    def __init__(
        self, toolbox: Toolbox, router: LLMRouter | None, recorder: UsageRecorder | None = None, fetch_timeout_sec: float = 600
    ):
        self.toolbox = toolbox
        self.router = router
        self.recorder = recorder
        self.fetch_timeout_sec = fetch_timeout_sec
        graph = StateGraph(TurnState)
        graph.add_node("decide", self._decide)
        graph.add_node("act", self._act)
        graph.add_node("respond", self._respond)
        graph.add_node("check_fetch", self._check_fetch)
        graph.add_node("respond_fetched", self._respond_fetched)
        graph.add_node("upload", self._upload)
        graph.add_conditional_edges(START, lambda s: "check_fetch" if s.get("poll") else "upload" if s.get("video") else "decide")
        graph.add_conditional_edges("decide", lambda s: "respond" if s["decision"].action == "reply" else "act")
        graph.add_conditional_edges("check_fetch", lambda s: "act" if s["update"] == "done" and s.get("decision") else END)
        graph.add_edge("upload", END)
        graph.add_conditional_edges("act", lambda s: "respond_fetched" if s.get("poll") else "respond")
        graph.add_edge("respond", END)
        graph.add_edge("respond_fetched", END)
        self.graph = graph.compile()

    def run(
        self, conversation: Conversation, message: str, image_key: str | None = None, video: tuple | None = None
    ) -> TurnState:
        return self.graph.invoke({"conversation": conversation, "message": message, "image_key": image_key, "video": video})

    def poll(self, conversation: Conversation) -> TurnState:
        return self.graph.invoke({"conversation": conversation, "poll": True, "message": ""})

    # ---- nodes -------------------------------------------------------------------------------------------------
    def _decide(self, state: TurnState) -> dict[str, Any]:
        conversation = state["conversation"]
        with span("agent.decide") as step:
            decision = decide(state["message"], conversation.context(bool(state.get("image_key"))), self.router)
            step.set(
                action=decision.action, decided_by=decision.source, confident=decision.confident, notes=decision.notes or None
            )
        if decision.llm_call is not None and self.recorder is not None:
            self.recorder.record(decision.llm_call, _route_outcome(decision))
        return {"decision": decision}

    def _act(self, state: TurnState) -> dict[str, Any]:
        decision, conversation = state["decision"], state["conversation"]
        if conversation.uploading and decision.action in ("find_clip", "answer_question") and not state.get("poll"):
            # Video first, request later: the video is still processing, so the request waits for it.
            text = decision.text or state["message"]
            conversation.upload_request, conversation.upload_request_action = text, decision.action
            status = self.toolbox.video_status(conversation.uploading)
            elapsed = time.time() - (conversation.upload_started or time.time())
            progress = replies.processing_status(status["status"], status.get("stage"), elapsed)
            return {"result": None, "queued": replies.queued_request(text, progress)}
        scope = conversation.last_video_id if decision.about_last_video else None
        # The user's own video is what they're asking about: requests and questions search only it.
        focused = scope is None and conversation.focus_video_id and decision.action in ("find_clip", "answer_question")
        if focused:
            scope = conversation.focus_video_id
        arguments: dict[str, Any] = {
            "find_clip": {"request": decision.text or state["message"], "video_id": scope},
            "find_by_image": {"image_key": state.get("image_key")},
            "answer_question": {"question": decision.text or state["message"], "video_id": scope},
            "fetch_from_pexels": {"query": decision.text, "count": 3},
            "list_videos": {},
        }[decision.action]
        if decision.action == "find_clip" and decision.text == conversation.last_request:
            arguments["max_clips"] = 3  # "another one": fetch a few, show the first not shown yet
        result = self.toolbox.run(decision.action, {k: v for k, v in arguments.items() if v is not None})
        if focused and result.ok and result.data.get("status") in ("no_match", "not_found"):
            result.data["not_in_upload"] = conversation.focus_title or "your video"  # say so; no library, no Pexels
        return {"result": result}

    def _upload(self, state: TurnState) -> dict[str, Any]:
        conversation = state["conversation"]
        fileobj, filename, content_type, size = state["video"]
        video_id = self.toolbox.upload_video(fileobj, filename, content_type, size)
        name = filename or "your video"
        conversation.uploading, conversation.upload_name, conversation.upload_started = video_id, name, time.time()
        conversation.upload_request = state["message"].strip() or None
        conversation.turns += 1
        reply = replies.upload_received(name, conversation.upload_request)
        return {"reply": reply, "clips": [], "citations": [], "videos": [], "decision": Decision("reply", source="rules")}

    def _check_upload(self, conversation: Conversation) -> dict[str, Any]:
        video_id = conversation.uploading
        status = self.toolbox.video_status(video_id)
        name = conversation.upload_name or "your video"
        if status["status"] in ("queued", "downloading", "processing"):
            if time.time() - (conversation.upload_started or time.time()) > 3 * self.fetch_timeout_sec:
                conversation.uploading = None
                return {"update": "timed_out", "kind": "upload", "reply": replies.fetch_timed_out(name)}
            elapsed = time.time() - (conversation.upload_started or time.time())
            text = replies.processing_status(status["status"], status.get("stage"), elapsed)
            return {"update": "pending", "kind": "upload", "progress": {"ready": 0, "pending": 1, "failed": 0},
                    "status_text": text}  # fmt: skip
        conversation.uploading = None
        if status["status"] != "ready":
            return {"update": "done", "kind": "upload", "progress": {"ready": 0, "pending": 0, "failed": 1},
                    "reply": replies.upload_failed(name, status.get("error"))}  # fmt: skip
        conversation.focus_video_id, conversation.focus_title = video_id, name
        ready = replies.upload_ready(name, status.get("duration_sec"), status.get("speech"), bool(conversation.upload_request))
        out = {"update": "done", "kind": "upload", "progress": {"ready": 1, "pending": 0, "failed": 0}, "reply": ready,
               "clips": [], "citations": [], "videos": []}  # fmt: skip
        if conversation.upload_request:  # a request came with the upload or while it processed: answer it now
            action = conversation.upload_request_action or "find_clip"
            out["decision"] = Decision(action, text=conversation.upload_request)
            conversation.upload_request, conversation.upload_request_action = None, None
        return out

    def _check_fetch(self, state: TurnState) -> dict[str, Any]:
        conversation = state["conversation"]
        if conversation.uploading:
            return self._check_upload(conversation)
        if not conversation.fetching:
            return {"update": "idle"}
        with span("agent.check_fetch", videos=len(conversation.fetching)) as step:
            data = self.toolbox.run("check_videos", {"video_ids": conversation.fetching}).data
            progress = {k: data.get(k, 0) for k in ("ready", "pending", "failed")}
            step.set(**progress)
        if progress["pending"] == 0:
            # Retry the request that found nothing, now over the new videos too.
            decision = Decision("find_clip", text=conversation.fetch_request or conversation.fetch_subject)
            return {"update": "done", "kind": "fetch", "progress": progress, "decision": decision}
        if time.time() - (conversation.fetch_started or time.time()) > self.fetch_timeout_sec:
            subject = conversation.fetch_subject or "requested"
            _clear_fetch(conversation)
            return {"update": "timed_out", "progress": progress, "reply": replies.fetch_timed_out(subject)}
        return {"update": "pending", "progress": progress}

    def _respond_fetched(self, state: TurnState) -> dict[str, Any]:
        conversation, result = state["conversation"], state.get("result")
        if state.get("kind") == "upload":  # the upload is ready and a request was waiting: answer it like any turn
            out = self._respond(state)
            return {**out, "reply": f"{state['reply']} {out['reply']}".strip()}
        subject, request = conversation.fetch_subject or "requested", state["decision"].text
        ready = state["progress"]["ready"]
        _clear_fetch(conversation)
        data = result.data if result is not None and result.ok else {}
        if data.get("status") == "answered" and data.get("clips"):
            clip = data["clips"][0]
            conversation.shown.append(_clip_id(clip))
            conversation.last_video, conversation.last_video_id = clip["title"], clip["video_id"]
            conversation.last_request, conversation.last_status = request, "answered"
            return {"reply": replies.fetched(subject, clip["explanation"]), "clips": [clip], "citations": [], "videos": []}
        return {"reply": replies.fetched_no_match(subject, ready), "clips": [], "citations": [], "videos": []}

    def _respond(self, state: TurnState) -> dict[str, Any]:
        decision, conversation, result = state["decision"], state["conversation"], state.get("result")
        if state.get("queued"):  # waiting for the user's video to finish processing
            conversation.turns += 1
            return {"reply": state["queued"], "clips": [], "citations": [], "videos": []}
        conversation.turns += 1
        offered, conversation.offered_fetch = conversation.offered_fetch, None  # an offer is open for one turn only
        out: dict[str, Any] = {"clips": [], "citations": [], "videos": []}

        if decision.action == "reply":
            out["reply"] = replies.CANNED.get(decision.reply or "", replies.CANNED["what_to_show"])
            if decision.reply == "declined":
                conversation.offered_fetch = None
            return out
        if result is None or not result.ok:
            out["reply"] = f"Sorry, that didn't work: {result.summary if result else 'no result'}"
            return out

        data = result.data
        conversation.last_action = decision.action
        if decision.action == "find_clip":
            request = decision.text or state["message"]
            status = data.get("status")
            if status == "answered":
                # "another one" (the same request again) skips clips already shown; a refined request may show one again
                again = request == conversation.last_request
                fresh = [c for c in data["clips"] if not again or _clip_id(c) not in conversation.shown]
                if not fresh:
                    out["reply"] = replies.no_more(request)
                else:
                    clip = fresh[0]
                    conversation.shown.append(_clip_id(clip))
                    conversation.last_video, conversation.last_video_id = clip["title"], clip["video_id"]
                    out["clips"], out["reply"] = [clip], clip["explanation"]
            elif status == "no_match" and data.get("not_in_upload"):
                out["reply"] = replies.not_in_upload(request, data["not_in_upload"])
            elif status == "no_match":
                intent = data.get("intent") or {}
                subject = intent.get("visual") or intent.get("topic") or intent.get("phrase") or request
                conversation.offered_fetch = subject
                out["reply"] = replies.offer_fetch(data["answer"], subject)
            else:
                out["reply"] = data.get("answer") or result.summary
            conversation.last_request, conversation.last_status = request, status
        elif decision.action == "find_by_image":
            out["clips"] = data.get("clips", [])
            if out["clips"]:
                clip = out["clips"][0]
                conversation.shown.append(_clip_id(clip))
                conversation.last_video, conversation.last_video_id = clip["title"], clip["video_id"]
            out["reply"] = data.get("answer") or result.summary
        elif decision.action == "answer_question":
            out["citations"] = data.get("citations", [])
            if data.get("clip_url"):
                out["clips"] = [
                    {
                        "url": data["clip_url"],
                        "key": data.get("clip_key"),
                        "video_id": data.get("clip_video_id"),
                        "title": data.get("clip_title"),
                        "start_sec": data.get("clip_start_sec"),
                        "end_sec": data.get("clip_end_sec"),
                        "explanation": "The moment this answer comes from.",
                    }
                ]
            if data.get("status") != "answered" and data.get("not_in_upload"):
                out["reply"] = f"I couldn't find that in your video “{data['not_in_upload']}”."
            else:
                out["reply"] = result.summary
            conversation.last_request, conversation.last_status = decision.text or state["message"], data.get("status")
            if out["citations"]:  # "that part" / "that clip" now means the moment the answer came from
                first = out["citations"][0]
                conversation.last_video, conversation.last_video_id = first["title"], first["video_id"]
        elif decision.action == "fetch_from_pexels":
            ids = data.get("video_ids", [])
            if ids:
                conversation.fetching = ids
                conversation.fetch_request = conversation.last_request if offered else decision.text
                conversation.fetch_subject, conversation.fetch_started = data["query"], time.time()
                out["reply"] = replies.fetching(len(ids), data["query"])
            else:
                out["reply"] = replies.nothing_new(data["query"])
        elif decision.action == "list_videos":
            out["videos"], out["reply"] = data.get("videos", []), result.summary
        return out
