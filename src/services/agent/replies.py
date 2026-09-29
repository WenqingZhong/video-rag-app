"""The agent's own words: fixed templates (as for /ask). The model routes; it never writes the reply, so a reply
can't claim something the tools didn't return."""

CANNED = {
    "greeting": (
        "Hi! I find moments in videos. Try “a dog on a beach”, “the part where she talks about caffeine”, "
        "“how long is a sleep cycle?”, or send me a photo. Videos you upload are private to you, deleted after "
        "7 days, or sooner if you say “delete my video”."
    ),
    "ok": "Okay. What would you like to see?",
    "declined": "Okay, I won't download anything. What else would you like to see?",
    "nothing_to_confirm": "Sure. What would you like to see?",
    "attach_image": "Send me the image and I'll find the clip that looks most like it.",
    "what_to_show": "What would you like to see or hear?",
}


def offer_fetch(no_match_answer: str, subject: str) -> str:
    return f"{no_match_answer} Want me to download some “{subject}” videos from Pexels? It takes a minute or two."


def fetching(count: int, subject: str) -> str:
    return f"Downloading {count} “{subject}” videos from Pexels. Processing takes a few minutes; I'll show you the clip when they're ready."


def nothing_new(subject: str) -> str:
    return f"Pexels had no new videos for “{subject}” that fit (up to 60 s long)."


def fetched(subject: str, explanation: str) -> str:
    return f"Your “{subject}” videos are ready. {explanation}"


def fetched_no_match(subject: str, ready: int) -> str:
    if ready == 0:
        return f"Sorry, none of the “{subject}” videos could be processed."
    return f"The new “{subject}” videos are in ({ready} ready), but none matched your request closely enough."


def fetch_timed_out(subject: str) -> str:
    return f"The “{subject}” videos are taking too long to process. Ask me again in a few minutes."


# Pipeline stages (videos.stage) in the order they run, and what each means to someone waiting.
STAGES = [
    ("downloading_source", "reading the file"),
    ("probing", "checking the video"),
    ("detecting_scenes", "finding the scene changes"),
    ("extracting_keyframes", "taking keyframes"),
    ("embedding_frames", "understanding the frames"),
    ("captioning", "describing the frames"),
    ("transcribing", "transcribing the speech"),
    ("saving_segments", "saving"),
    ("indexing", "making it searchable"),
]


def processing_status(status: str, stage: str | None, elapsed_sec: float) -> str:
    """ "Step 6 of 9: describing the frames (3 of 11) · 0:42" — so a long wait never looks like a dead bot."""
    clock = f"{int(elapsed_sec // 60)}:{int(elapsed_sec % 60):02d}"
    if status == "queued" or not stage:
        return f"Waiting in line: another video is being processed first · {clock}"
    name, _, detail = stage.partition(" ")
    names = [s for s, _ in STAGES]
    if name not in names:
        return f"Processing · {clock}"
    step, label = names.index(name) + 1, dict(STAGES)[name]
    frames = f" ({detail.replace('/', ' of ')})" if detail else ""
    return f"Step {step} of {len(STAGES)}: {label}{frames} · {clock}"


def queued_request(request: str, status_text: str) -> str:
    return f"Your video is still processing ({status_text}). I'll find “{request}” as soon as it's ready."


def not_in_upload(request: str, name: str) -> str:
    return f"I couldn't find “{request}” in your video “{name}”."


def upload_received(name: str, request: str | None) -> str:
    then = f"Then I'll find “{request}”." if request else "Tell me what to find in it; you can ask now."
    return f"Got your video “{name}”. Processing takes a minute or two (longer for long videos). {then}"


def upload_ready(name: str, duration_sec: float | None, speech: bool, answering: bool = False) -> str:
    length = f" ({int((duration_sec or 0) // 60)}:{int((duration_sec or 0) % 60):02d})" if duration_sec else ""
    if answering:  # a request came with the upload: its clip follows
        return f"Your video “{name}”{length} is ready."
    example = "“the part where they say …”, “where do they talk about …”" if speech else "“a close-up of …”"
    return f"Your video “{name}”{length} is ready. Ask me for any moment in it, e.g. {example}, or ask a question about it."


def upload_failed(name: str, error: str | None) -> str:
    if error and error.startswith("VideoTooLong: "):
        return f"Sorry, “{name}” is too long: {error.removeprefix('VideoTooLong: ')}. Send a shorter clip."
    return f"Sorry, I couldn't process “{name}”{': ' + error[:120] if error else ''}. Try another file (MP4, MOV, WebM)."


def no_more(request: str) -> str:
    return f"That's every matching clip I have for “{request}”."


# ---- deleting the user's own videos ------------------------------------------------------------------------------
NOTHING_TO_DELETE = (
    "You haven't uploaded any videos, so there's nothing of yours to delete. "
    "(The videos in my library are shared stock footage and stay.)"
)
DELETE_KEPT = "Okay, I'll keep it."


def _names(titles: list[str]) -> str:
    quoted = [f"“{t}”" for t in titles]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def confirm_delete(titles: list[str]) -> str:
    what = _names(titles) if len(titles) == 1 else f"your {len(titles)} videos ({_names(titles)})"
    return (
        f"Delete {what}? It'll be removed from search and storage right away, and can't be undone. "
        "Reply “yes” to delete, or “no” to keep it."
    )


def deleted(done: list[str], busy: list[str]) -> str:
    parts = []
    if done:
        parts.append(f"Deleted {_names(done)}. {'It is' if len(done) == 1 else 'They are'} gone from search and storage.")
    if busy:
        parts.append(f"{_names(busy)} {'is' if len(busy) == 1 else 'are'} still being processed: ask again once it's done.")
    return " ".join(parts) or "That video was already gone."
