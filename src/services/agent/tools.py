"""What the agent can do. Each tool wraps a service the API already has, and returns:
- `summary`: a few short lines for the model (a 3B model reads long tool output slowly and gets lost in it);
- `data`: structured results for the chat page and the Telegram bot (clip URLs, times, credits).

The tools know nothing about LangGraph: the agent's graph (graph.py) adapts them. Tools never raise: a failure is a result
the model can read ("search is unavailable") and react to.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.db.interfaces.base import BaseDatabase
from src.models import VideoStatus
from src.repositories import VideoRepository
from src.services.answering import Answer, AnsweredClip, AskService, ImageAskService, mmss
from src.services.ingestion import IngestionService
from src.services.pexels import PexelsClient
from src.services.qa import QAService
from src.services.storage import StorageClient
from src.services.tracing import span

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # what the model reads to choose the tool: short, concrete, with an example
    parameters: dict[str, Any]  # JSON schema of the arguments


def _clip_data(storage: StorageClient, clip: AnsweredClip) -> dict[str, Any]:
    doc = clip.hit.source
    return {
        "url": storage.presigned_url(clip.key),
        "key": clip.key,  # S3 key: clients that can't open the presigned URL (the Telegram bot) read the file directly
        "video_id": doc["video_id"],
        "title": doc.get("video_title"),
        "source": doc.get("video_source"),
        "author": doc.get("video_author"),
        "source_url": doc.get("video_source_url"),
        "start_sec": clip.clip.start_sec,
        "end_sec": clip.clip.end_sec,
        "explanation": clip.explanation,
    }


class Toolbox:
    SPECS: ClassVar[list[ToolSpec]] = [
        ToolSpec(
            "find_clip",
            'Find a video clip from a description, a quote or a topic. Example: {"request": "a dog on a beach"}.',
            {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "what to show, what someone says, or a topic"},
                    "video_id": {"type": "string", "description": "optional: search only this video"},
                },
                "required": ["request"],
            },
        ),
        ToolSpec(
            "find_by_image",
            "Find the clip that looks most like the image the user uploaded. Use only when the user sent an image.",
            {"type": "object", "properties": {"image_key": {"type": "string"}}, "required": ["image_key"]},
        ),
        ToolSpec(
            "answer_question",
            'Answer a question about what videos say or show, citing the moments. Example: {"question": "how long is '
            'a sleep cycle?"}.',
            {
                "type": "object",
                "properties": {"question": {"type": "string"}, "video_id": {"type": "string"}},
                "required": ["question"],
            },
        ),
        ToolSpec(
            "fetch_from_pexels",
            "Download new stock videos from Pexels for a subject missing from the library. Only after the user agreed.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}, "count": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["query"],
            },
        ),
        ToolSpec(
            "check_videos",
            "Check whether fetched or uploaded videos have finished processing.",
            {
                "type": "object",
                "properties": {"video_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["video_ids"],
            },
        ),
        ToolSpec(
            "list_videos",
            "List the videos in the library (titles, lengths, whether they have speech).",
            {"type": "object", "properties": {"source": {"type": "string", "enum": ["pexels", "upload"]}}},
        ),
    ]

    def __init__(
        self,
        database: BaseDatabase,
        storage: StorageClient,
        ask: AskService,
        image_ask: ImageAskService,
        qa: QAService,
        make_ingestion: Callable[[Any], IngestionService],  # session → IngestionService
        pexels: PexelsClient | None,
    ):
        self.database = database
        self.storage = storage
        self.ask = ask
        self.image_ask = image_ask
        self.qa = qa
        self.make_ingestion = make_ingestion
        self.pexels = pexels

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = getattr(self, f"_{name}", None)
        if handler is None or name not in {s.name for s in self.SPECS}:
            return ToolResult(False, f"There is no tool called {name!r}.")
        with span(f"tool.{name}", arguments=arguments) as step:
            try:
                result = handler(**arguments)
            except TypeError as exc:  # the model passed arguments the tool doesn't take
                result = ToolResult(False, f"Wrong arguments for {name}: {exc}")
            except Exception as exc:  # a failed tool is a result the model can react to, not a crash
                logger.exception("tool %s failed", name)
                result = ToolResult(False, f"{name} failed: {type(exc).__name__}")
            step.set(ok=result.ok)
            return result

    # ---- not for the model: called by the graph when a video is attached ------------------------------
    def upload_video(self, fileobj: Any, filename: str | None, content_type: str | None, size: int | None) -> str:
        with span("tool.upload_video", filename=filename, size=size), self.database.get_session() as session:
            return self.make_ingestion(session).create_upload(fileobj, filename, content_type, size).id

    def video_status(self, video_id: str) -> dict[str, Any]:
        with self.database.get_session() as session:
            video = VideoRepository(session).get(video_id)
            if video is None:
                return {"status": "missing"}
            return {"status": video.status, "stage": video.stage, "title": video.title, "duration_sec": video.duration_sec,
                    "speech": bool(video.language), "error": video.error}  # fmt: skip

    # ---- tools -----------------------------------------------------------------------------------------
    def _find_clip(self, request: str, video_id: str | None = None, max_clips: int = 1) -> ToolResult:
        answer: Answer = self.ask.ask(request, video_id=video_id, max_clips=max(1, min(int(max_clips), 3)))
        intent = answer.understanding.intent
        data = {
            "status": answer.status,
            "answer": answer.answer,
            "intent": intent.model_dump(),
            "clips": [_clip_data(self.storage, c) for c in answer.clips],
        }
        if answer.status == "answered":
            return ToolResult(True, f"Found: {answer.answer}", data)
        if answer.status == "needs_subject":
            return ToolResult(True, "The request doesn't say what to show. Ask the user what they want to see.", data)
        return ToolResult(True, f"No match in the library for {intent.text!r}. You may offer to fetch videos from Pexels.", data)

    def _find_by_image(self, image_key: str) -> ToolResult:
        answer = self.image_ask.ask(self.storage.get_bytes(image_key))
        data = {"status": answer.status, "answer": answer.answer, "clips": [_clip_data(self.storage, c) for c in answer.clips]}
        return ToolResult(True, answer.answer if answer.status == "answered" else "Nothing in the library looks like it.", data)

    def _answer_question(self, question: str, video_id: str | None = None) -> ToolResult:
        result = self.qa.answer(question, video_id=video_id)
        citations = [
            {
                "video_id": e.video_id,
                "title": e.title,
                "kind": e.kind,
                "start_sec": e.start_sec,
                "end_sec": e.end_sec,
                "text": e.text,
            }
            for e in result.citations
        ]
        data = {
            "status": result.status,
            "answer": result.answer,
            "citations": citations,
            "clip_url": self.storage.presigned_url(result.clip_key) if result.clip_key else None,
            "clip_key": result.clip_key,
            "clip_start_sec": result.clip.start_sec if result.clip else None,
            "clip_end_sec": result.clip.end_sec if result.clip else None,
            "clip_video_id": result.citations[0].video_id if result.citations else None,
            "clip_title": result.citations[0].title if result.citations else None,
        }
        if result.status != "answered":
            return ToolResult(result.status != "unavailable", result.answer, data)
        sources = "; ".join(f'"{e.title}" {mmss(e.start_sec)}–{mmss(e.end_sec)}' for e in result.citations)
        return ToolResult(True, f"{result.answer} (from {sources})", data)

    def _fetch_from_pexels(self, query: str, count: int = 3) -> ToolResult:
        if self.pexels is None:
            return ToolResult(False, "Pexels is not configured (no API key).")
        count = max(1, min(int(count), 5))
        with self.database.get_session() as session:
            result = self.make_ingestion(session).ingest_pexels(self.pexels, query, count)
            ids = [v.id for v in result.queued]
        data = {"query": query, "video_ids": ids, "skipped_existing": result.skipped_existing}
        if not ids:
            return ToolResult(True, f"Pexels had no new suitable videos for {query!r}.", data)
        return ToolResult(True, f"Fetching {len(ids)} videos for {query!r}; they take about a minute to process.", data)

    def _check_videos(self, video_ids: list[str]) -> ToolResult:
        with self.database.get_session() as session:
            repo = VideoRepository(session)
            statuses = {vid: (v.status if (v := repo.get(vid)) else "missing") for vid in video_ids}
        ready = sum(1 for s in statuses.values() if s == VideoStatus.READY)
        failed = sum(1 for s in statuses.values() if s in (VideoStatus.FAILED, "missing"))
        pending = len(statuses) - ready - failed
        summary = f"{ready} ready, {pending} still processing, {failed} failed."
        return ToolResult(True, summary, {"statuses": statuses, "ready": ready, "pending": pending, "failed": failed})

    def _list_videos(self, source: str | None = None) -> ToolResult:
        with self.database.get_session() as session:
            videos, total = VideoRepository(session).list_videos(limit=50, status=VideoStatus.READY, source=source)
            items = [
                {
                    "id": v.id,
                    "title": v.title or v.original_filename,
                    "source": v.source,
                    "duration_sec": v.duration_sec,
                    "speech": bool(v.language),
                }
                for v in videos
            ]
        lines = [f"- {i['title']} ({i['duration_sec'] or 0:.0f} s{', speech' if i['speech'] else ''})" for i in items[:15]]
        more = f"\n… and {total - 15} more" if total > 15 else ""
        return ToolResult(True, f"{total} videos:\n" + "\n".join(lines) + more, {"videos": items, "total": total})
