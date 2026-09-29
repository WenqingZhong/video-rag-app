"""Fixed-template answers, filled only from search results: instant, and they can't state anything not found."""

from src.services.understanding.intent import Intent


def mmss(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


def explain(intent: Intent, doc: dict, start: float, end: float, matched_text: str | None, match_score: float | None) -> str:
    """One line describing a returned clip: what it is and where it comes from."""
    title = doc.get("video_title") or "untitled video"
    when = f"{mmss(start)}–{mmss(end)}"
    if doc.get("kind") == "speech" and matched_text:
        heard = matched_text.strip()
        if match_score is not None and match_score < 0.999:
            return f'Closest match at {when} in "{title}": "{heard}" (you asked for "{intent.text}").'
        return f'At {when} in "{title}": "{heard}"'
    if doc.get("kind") == "speech":
        snippet = (doc.get("text") or "").strip()
        if len(snippet) > 160:
            snippet = snippet[:157].rsplit(" ", 1)[0] + "…"
        return f'At {when} in "{title}": "{snippet}"'
    credit = f" (Pexels, by {doc['video_author']})" if doc.get("video_source") == "pexels" and doc.get("video_author") else ""
    caption = (doc.get("caption") or "").strip()
    return f'"{title}"{credit}, {when}' + (f": {caption}" if caption else "")


def explain_image(doc: dict, start: float, end: float, similarity: float) -> str:
    """The clip that looks most like the user's photo."""
    title = doc.get("video_title") or "untitled video"
    credit = f" (Pexels, by {doc['video_author']})" if doc.get("video_source") == "pexels" and doc.get("video_author") else ""
    caption = (doc.get("caption") or "").strip()
    return f'Closest match to your image: "{title}"{credit}, {mmss(start)}–{mmss(end)} (similarity {similarity:.2f})' + (
        f": {caption}" if caption else ""
    )


def no_match(intent: Intent) -> str:
    what = {"quote": "anyone saying", "topic": "a discussion of", "visual": "footage of"}[intent.type]
    without = f" without {', '.join(intent.exclude)}" if intent.exclude else ""
    return f'No moment matched {what} "{intent.text}"{without}.'


def ask_for_subject(intent: Intent) -> str:
    avoid = f" I'll leave out {', '.join(intent.exclude)}." if intent.exclude else ""
    return f"What would you like to see or hear?{avoid}"
