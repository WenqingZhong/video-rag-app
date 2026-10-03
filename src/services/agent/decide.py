"""Each chat turn → one action. The model fills a small form (Ollama JSON schema: qwen2.5vl has no native tool
calling); plain-code guards check it against the message and the conversation; rules are the fallback.

The same design as request understanding (services/understanding), one level up: there the model chose quote/topic/visual,
here it chooses which tool to use and rewrites follow-ups ("now without people") into a full request.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

from src.services.llm import ChatModel, ModelError
from src.services.understanding.llm import InvalidReply
from src.services.usage import LLMCall

logger = logging.getLogger(__name__)

Action = Literal["find_clip", "find_by_image", "answer_question", "fetch_from_pexels", "list_videos", "reply"]
ACTIONS: tuple[Action, ...] = ("find_clip", "find_by_image", "answer_question", "fetch_from_pexels", "list_videos", "reply")


@dataclass
class TurnContext:
    """What the router needs to know about the conversation so far (kept short: a 3B model reads slowly)."""

    last_request: str | None = None  # the last full request or question
    last_status: str | None = None  # "answered" | "no_match" | ...
    last_action: str | None = None  # "find_clip" | "answer_question" | ...: follow-ups repeat the same kind
    last_video: str | None = None  # title of the video last shown
    offered_fetch: str | None = None  # a Pexels fetch we offered and the user hasn't answered yet
    image_attached: bool = False

    def render(self) -> str:
        lines = []
        if self.last_request:
            lines.append(
                f'Last request ({self.last_action or "find_clip"}): "{self.last_request}" → {self.last_status or "answered"}'
            )
        if self.last_video:
            lines.append(f'Last clip shown: "{self.last_video}"')
        if self.offered_fetch:
            lines.append(f'You offered to download videos of "{self.offered_fetch}" from Pexels; the user has not answered.')
        lines.append(f"The user attached an image: {'yes' if self.image_attached else 'no'}")
        return "\n".join(lines)


@dataclass
class Decision:
    action: Action
    text: str | None = None  # find_clip: the full request · answer_question: the question · fetch: what to fetch
    about_last_video: bool = False  # the question/request is about the clip just shown
    reply: str | None = None  # action "reply": which canned reply (see the agent's templates)
    source: str = "rules"  # "llm" | "rules"
    confident: bool = False  # rules only: a pattern clear by construction, decided without the model
    notes: list[str] = field(default_factory=list)
    llm_call: LLMCall | None = None


# ---- the model ---------------------------------------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "text": {"type": ["string", "null"]},
        "about_last_video": {"type": "boolean"},
    },
    "required": ["action", "text", "about_last_video"],
}

SYSTEM_PROMPT = """You route messages in a video-clip chat app. Pick ONE action and reply with JSON only.
You do NOT answer the user yourself: the tools do. Never put an answer or a fact in text.

Actions:
- find_clip: the user wants to SEE a moment: something shown, someone saying words, where something is discussed.
  Messages with "where", "the part", "show", "find", "clip" are find_clip.
- answer_question: the user asks for a FACT the videos state ("how long…?", "what does she say about…?").
- find_by_image: the user attached an image and wants a similar clip.
- fetch_from_pexels: ONLY if the user accepts your offer to download videos, or explicitly asks to download from Pexels.
- list_videos: the user asks what videos exist.
- reply: greetings, thanks, questions about you, yes/no with nothing to answer, or anything the tools can't do.

text: copy the user's own words.
- find_clip: the full request on its own. If the message continues the last request, combine them:
  last "<X>" + "now without <Y>" → "<X> without <Y>"; last "<X>" + "what about <Z>?" → "<Z>".
- answer_question: the user's question, as they wrote it. fetch_from_pexels: the subject to download.
- Otherwise null. If the user names nothing to show, use action reply.
about_last_video: true only if the message refers to the clip just shown ("that clip", "in it")."""


class RouterReply(BaseModel):
    action: Action
    text: str | None = None
    about_last_video: bool = False


class LLMRouter:
    def __init__(self, chat: ChatModel):
        self.chat = chat

    @property
    def model(self) -> str:
        return self.chat.name

    def route(self, message: str, context: TurnContext) -> tuple[RouterReply, LLMCall]:
        user = f"{context.render()}\nMessage: {message!r}"
        # 150, not 60: a model that rambles in `text` got cut off mid-JSON (an invalid reply)
        text, call = self.chat.json_reply(SYSTEM_PROMPT, user, SCHEMA, "route", max_tokens=150)
        try:
            return RouterReply.model_validate(json.loads(text)), call
        except (ValidationError, ValueError, KeyError) as exc:
            raise InvalidReply(f"{type(exc).__name__}: {exc}", call) from exc

    def close(self) -> None:
        self.chat.close()


# ---- rules ---------------------------------------------------------------------------------------------------------
_YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|please do|please|fine|sounds good)\b", re.IGNORECASE)
_NO = re.compile(r"^(?:no|nope|nah|not now|no thanks|don'?t)\b[\s,.!]*", re.IGNORECASE)
_PEXELS = re.compile(r"\b(?:pexels|download|fetch)\b", re.IGNORECASE)
_GREETING = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|cheers|great|who are you|what can you do|help)\b|\b(?:delete|remove|erase)\b",
    re.IGNORECASE,
)
_LIBRARY = re.compile(
    r"\b(?:list|what|which)\b.*\b(?:videos|talks|library|clips)\b.*(?:have|there|library|\?)|^list\b", re.IGNORECASE
)
_QUESTION = re.compile(r"^(?:what|how|why|when|who|which|does|do|did|is|are|can|was|were)\b", re.IGNORECASE)
_LOCATE = re.compile(r"\b(?:where|the part|show|clip|footage|find|give me)\b", re.IGNORECASE)
_REFERS = re.compile(r"\b(?:that|this|the) (?:clip|video|one|scene|shot)\b|\bin it\b|\bthat\?", re.IGNORECASE)
_IMAGE = re.compile(r"\b(?:this|looks? like|similar|photo|picture|image)\b", re.IGNORECASE)
_ANOTHER = re.compile(r"^(?:show me )?(?:another|one more|more|next)(?: one)?\b", re.IGNORECASE)
_SWITCH = re.compile(r"^(?:what|how) about\s+|^(?:no,?\s*)?i meant\s+|^instead,?\s*", re.IGNORECASE)
_NEW_SUBJECT = re.compile(r"^(?:and|also|now)\s+(?=(?:a|an|some|the)\s+(?!one\b))", re.IGNORECASE)  # "and a balloon"
_THAT_MOMENT = re.compile(r"\b(?:show|play|see)\b.*\bthat (?:part|bit|moment|clip)\b", re.IGNORECASE)
_CONTINUE = re.compile(r"^(?:now|and|but|same but|the same but|also)\b\s*(?:one\s+)?", re.IGNORECASE)
_REFINE = re.compile(r"\b(?:without|with no|no|at night|in the|with a|with)\b", re.IGNORECASE)


def _pexels_subject(message: str) -> str:
    words = re.sub(
        r"\b(?:please|download|fetch|get|some|videos?|clips?|footage|from|pexels|of|me|can you|for)\b",
        " ",
        message,
        flags=re.IGNORECASE,
    )
    return " ".join(words.split()).strip(" ?.!,") or message


FOLLOW_UP = "follow-up of the last request"
SWITCH = "same kind of request, new subject"


def decide_with_rules(message: str, context: TurnContext) -> Decision:
    """Rules first. `confident=True` marks patterns that are unambiguous by construction (an image with no other
    request, a yes/no to our offer, an explicit "download from Pexels", a greeting, a library question): those are
    decided here, instantly. Everything else is only a best guess, used when the model fails or a guard rejects it."""
    text = message.strip()

    def sure(action: Action, **kwargs) -> Decision:
        return Decision(action, confident=True, **kwargs)

    if context.image_attached and (not text or _IMAGE.search(text) or len(text.split()) <= 3):
        return sure("find_by_image")
    if context.offered_fetch and _YES.match(text):
        return sure("fetch_from_pexels", text=context.offered_fetch)
    if _NO.match(text):
        rest = _NO.sub("", text).strip()
        if len(rest.split()) < 2 or re.match(r"^(?:thanks|thank you)\b", rest, re.IGNORECASE):
            return sure("reply", reply="declined" if context.offered_fetch else "ok")
        rest = _SWITCH.sub("", rest)  # "no, I meant a calm sea" → "a calm sea"
        text = re.sub(r"^(?:show me|give me)\s+", "", re.sub(r"\binstead\b", "", rest, flags=re.IGNORECASE)).strip(" ,.")
        return Decision("find_clip", text=text)
    if _YES.match(text) and len(text.split()) <= 3:
        return sure("reply", reply="nothing_to_confirm")
    if _PEXELS.search(text):
        return sure("fetch_from_pexels", text=_pexels_subject(text))
    if _GREETING.search(text) or not text:
        return sure("reply", reply="greeting")
    if _IMAGE.search(text) and re.search(r"\blooks? like\b|\bthis (?:photo|picture|image)\b", text, re.IGNORECASE):
        return sure("reply", reply="attach_image")
    if _LIBRARY.search(text):
        return sure("list_videos")
    if context.last_request:
        repeat: Action = "answer_question" if context.last_action == "answer_question" else "find_clip"
        if _ANOTHER.match(text):
            return Decision("find_clip", text=context.last_request, notes=[FOLLOW_UP])
        if context.last_action == "answer_question" and _THAT_MOMENT.search(text):
            # "show me that part" after an answer: the moment the answer came from
            return sure("find_clip", text=context.last_request, about_last_video=True, notes=[FOLLOW_UP])
        if _SWITCH.match(text) or _NEW_SUBJECT.match(text):
            # "what about X" / "and a X": the same kind of request as last time, about X
            subject = _NEW_SUBJECT.sub("", _SWITCH.sub("", text)).strip(" ?.!")
            return sure(repeat, text=text.strip() if repeat == "answer_question" else subject, notes=[SWITCH])
        if _CONTINUE.match(text) or (len(text.split()) <= 4 and _REFINE.search(text)):
            addition = _CONTINUE.sub("", text).strip(" ?.!")
            return Decision("find_clip", text=f"{context.last_request} {addition}".strip(), notes=[FOLLOW_UP])
    if _QUESTION.match(text) and not _asks_to_see(text):
        return Decision("answer_question", text=text, about_last_video=bool(_REFERS.search(text)))
    if not re.search(r"[a-z]{3,}", text, re.IGNORECASE) or re.fullmatch(r"(?:exclude|without|no)\s+\w+", text, re.IGNORECASE):
        return Decision("reply", reply="what_to_show")
    return Decision("find_clip", text=text)


def _asks_to_see(message: str) -> bool:
    """ "where…", "the part…", "show…": a request to see a moment. "in that clip" only refers to one, so it doesn't count."""
    return bool(_LOCATE.search(_REFERS.sub(" ", message)))


# ---- the model, checked --------------------------------------------------------------------------------------------
_SMALL_WORDS = {"a", "an", "the", "of", "on", "in", "at", "to", "for", "with", "and", "or", "is", "are", "me", "some", "any"}


def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9']+", text.lower())} - _SMALL_WORDS


def decide(message: str, context: TurnContext, router: LLMRouter | None) -> Decision:
    rules = decide_with_rules(message, context)
    if router is None or rules.confident:  # rules first: clear cases never wait for the model
        return rules
    try:
        reply, call = router.route(message, context)
    except InvalidReply as exc:
        rules.notes.append(f"router reply invalid: {exc}")
        rules.llm_call = exc.call
        return rules
    except ModelError as exc:
        rules.notes.append(f"router unavailable: {exc}")
        return rules

    def fallback(reason: str) -> Decision:
        rules.notes.append(f"model chose {reply.action} {reply.text!r}: {reason} → rules")
        rules.llm_call = call
        return rules

    consented = bool(context.offered_fetch and _YES.match(message.strip())) or bool(_PEXELS.search(message))
    # Guard 1 (consent): never download without an accepted offer or an explicit request.
    if reply.action == "fetch_from_pexels" and not consented:
        return fallback("no consent to download")
    # Guard 2 (image): image search needs an image; a bare "find this" with an image is image search.
    if reply.action == "find_by_image" and not context.image_attached:
        return fallback("no image attached")
    if context.image_attached and rules.action == "find_by_image" and reply.action != "find_by_image":
        return fallback("an image was attached with no other request")
    # Guard 3 (yes/no): a bare yes/no with nothing to answer is not a request.
    if (_YES.match(message.strip()) or _NO.match(message.strip())) and rules.action == "reply":
        return fallback("a yes/no with nothing to act on")
    # Guard 4 (seeing, not asking): "where…", "the part…", "show…" ask to SEE a moment; that's a clip, not a fact.
    if reply.action == "answer_question" and _asks_to_see(message):
        return fallback("the message asks to see a moment")
    # Guard 5 (a question stays the user's question): the model tends to write its own ANSWER into the text.
    # Its choice of action is usually still right, so keep the action and ask the user's own words.
    if reply.action == "answer_question":
        asked, wanted = _words(message), _words(reply.text or "")
        if not wanted or len(wanted & asked) / len(wanted) < 0.8:
            return Decision(
                "answer_question",
                text=message.strip(),
                about_last_video=reply.about_last_video and bool(context.last_video) and bool(_REFERS.search(message)),
                source="llm",
                notes=[f"model wrote {reply.text!r} as the question: asking the user's words instead"],
                llm_call=call,
            )
    # Guard 7 (follow-ups): "now without people" continues the last request; the model often passes the message on
    # alone ("now without people" → a search for "now"). If the rules see a follow-up and the model's request has
    # none of the last request's words, use the rules' combined request.
    if (
        reply.action == "find_clip"
        and FOLLOW_UP in rules.notes
        and context.last_request
        and not (_words(context.last_request) & _words(reply.text or ""))
    ):
        rules.notes.append(f"model's request {reply.text!r} dropped the last request → rules' {rules.text!r}")
        rules.llm_call = call
        return rules
    # Guard 9 (the message counts): a new request must use the message's own words. The model sometimes repeats the
    # last request instead ("show me cats instead" → "a unicorn" again).
    if (
        reply.action == "find_clip"
        and FOLLOW_UP not in rules.notes
        and _words(message) - _SMALL_WORDS
        and not (_words(message) & _words(reply.text or ""))
    ):
        return fallback("request ignores the message's own words")
    # Guard 6 (grounding): the text must use words from the message or the conversation, not invented ones.
    if reply.action in ("find_clip", "answer_question", "fetch_from_pexels"):
        said = _words(message) | _words(context.last_request or "") | _words(context.offered_fetch or "")
        wanted = _words(reply.text or "")
        if not wanted or len(wanted & said) / len(wanted) < 0.6:
            return fallback("text not grounded in the conversation")
    text = reply.text if reply.action in ("find_clip", "answer_question", "fetch_from_pexels") else None
    return Decision(
        reply.action,
        text=text,
        # Guard 8 (scope): only a message that points at the clip ("that clip", "in it") is about it. The model sets
        # the flag freely, which silently narrowed "a beach" to the last video shown (→ no match).
        about_last_video=reply.about_last_video and bool(context.last_video) and bool(_REFERS.search(message)),
        reply=rules.reply if reply.action == "reply" else None,
        source="llm",
        llm_call=call,
    )
