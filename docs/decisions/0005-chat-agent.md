# ADR 0005: The chat agent: rules first, a local 3B model checked by guards, LangGraph, fixed-template replies

- **Status:** accepted
- **Date:** 2026-09-28
- **Evidence:** [`eval/results/agent_turns*.json`](../../eval/results/) (`make eval-agent`),
  [`conversations_first_run.json`](../../eval/results/conversations_first_run.json) (`make eval-chat`),
  [`images.json`](../../eval/results/images.json) (`make eval-images`), [`qa.json`](../../eval/results/qa.json) (`make eval-qa`)

## In one minute

The app became a **conversation**: follow-ups ("now without people", "another one", "what colour is the dog in that
clip?"), questions answered from the videos with citations, search by photo, and fetching from Pexels **after asking**.
It's available as a web page (`/app`) and a Telegram bot.

**Each turn:**
1. **Rules first.** Clear patterns (an image with no other request, yes/no to our offer, "download from Pexels", greetings,
   library questions, "what about X") are decided instantly, with no model call.
2. **The model for the rest** (`qwen2.5vl:3b`, JSON-schema form). **Eight guards** check its choice against the message and
   the conversation. It can't download without consent, answer instead of routing, or narrow a request to the last clip.
3. **One tool** runs.
4. **A fixed-template reply** is built from the tool's result. The model never writes the reply.

**Results:**
- whole conversations, first run with labels fixed beforehand: **23 of 28 turns** and **8 of 12 conversations** right;
- **0 downloads without consent** in any evaluation;
- median **8.9 s** per turn, because the model runs on the CPU in Docker.

## Context

Before this, the app answered one request with one clip. The goals:
- follow-ups;
- answering questions about the videos;
- search by photo;
- fetching from Pexels when the library has nothing, only with the user's consent;
- a web chat and a Telegram bot.

It had to stay local (`qwen2.5vl:3b`, already used for captions and request understanding), but be ready for AWS.

## What was measured, and what it decided

**1. The model can't call tools natively.** Ollama reports that `qwen2.5vl:3b` "does not support tools". So the agent
uses the same pattern as request understanding (ADR 0002): the model fills a JSON-schema form, and plain code checks it.

**2. The model alone isn't reliable enough.** Choosing the action, on labelled turns:

| | first set (40) | held-out 1 (20) | held-out 2 (20) | unsafe downloads |
|---|---|---|---|---|
| rules only | 1.00 | 0.60 | 0.75 | 0 |
| model only | 0.70 | 0.75 | 0.85 | **1** |
| model + guards | 0.85 | 0.70 | — | 0 |
| **rules first + model + guards** (chosen) | **0.95** | **0.85** | **0.95** (0.80 unseen) | **0** |

- The rules fit the phrasings they were written for (1.00), but fall to 0.60 on new ones.
- The model generalises better, but it:
  - **writes its own answer into the request** ("A puppy is a cute and cuddly pet…");
  - **copies prompt examples** (it downloaded "horse" after a bare "yes");
  - **drops the context of follow-ups**.
- Rules first + model: the model is consulted on 50–70% of turns.

Held-out 2's clean score is **0.80**; the later numbers come after fixes that had seen it. Each set was used for one round
of changes; the conversation set below was written fresh after that.

**3. The guards** (in `src/services/agent/decide.py`):

| # | Guard | Why |
|---|---|---|
| 1 | download only after an accepted offer or an explicit "download from Pexels" | the model downloaded on a bare "yes" |
| 2 | image search only with an image; an image alone means image search | the model chose image search for "a beach" |
| 3 | a bare yes/no with nothing to answer isn't a request | |
| 4 | "where…", "the part…", "show…" ask to *see* a moment, not a fact | the model routed them to questions |
| 5 | a question stays the user's question (the model's action is kept) | the model wrote answers as the question |
| 6 | the request must use words from the message or the conversation | invented requests |
| 7 | a follow-up keeps the last request ("now without people" → "a beach without people") | a live test searched for "now" |
| 8 | only a message pointing at the clip ("that clip", "in it") is about the last video | "a beach" was silently narrowed to the dog video |
| 9 | a new request uses the message's own words | "show me cats instead" → "a unicorn" again |

Guards 7–9 came from **live use**, not the evaluation sets. Every guard change was checked against all three routing
sets for regressions.

**4. Whole conversations** (`eval/conversations.json`, 12 conversations and 28 turns through `/chat`, labelled first):

| run | turns right | conversations right | p50 / p95 per turn | tokens per turn |
|---|---|---|---|---|
| **first run (unseen)** | **23 / 28** | **8 / 12** | 8.9 s / 13.2 s | 562 |
| after five general fixes (seen) | 28 / 28 | 12 / 12 | 5.4 s / 12.9 s (warm caches) | 399 |

The five misses had general causes:
- a routing reply cut off mid-JSON (60 output tokens; now 150);
- "what about X" should repeat the previous *kind* of request;
- "and a X" names a new subject;
- the model repeating the last request (guard 9);
- no handling of "show me that part" after an answer.

## The tools

| Tool | What | Measured |
|---|---|---|
| `find_clip` | the `/ask` pipeline | ADR 0002, ADR 0003 |
| `find_by_image` (**new**) | CLIP photo vector vs keyframe vectors, cut-off **0.57** | frames of library videos: best match right **10/10**; Pexels photos of subjects in the library: **18/20**; photos of absent subjects: **0/15** false answers (`make eval-images`) |
| `answer_question` (**new**) | answers **only** from numbered, timed excerpts; must cite one; every number must be in the cited text | **0/7 made up** (incl. "does alcohol affect sleep?", which the talk never mentions); facts right 16/17; citations 17/17 (`make eval-qa`) |
| `fetch_from_pexels` | queue downloads, then poll `/chat/{id}/updates`; retry the original request when processed | live: "a hot air balloon" → 3 videos → clip in **78 s** |
| `list_videos` | the library | |
| *(upload, not a model choice)* | a video attached to a message is stored and processed, then becomes the conversation's focus: requests and questions search **only** it, and a miss says so ("I couldn't find … in your video"), with no library results and no Pexels offer. A request sent with the video, **or any time while it's processing**, waits and runs when it's ready. While it processes, clients show which step it's on ("Step 6 of 9: describing the frames (3 of 11) · 0:42"); the bot edits one status message in place | live: upload, then a separate request 30 s later → "still processing (Step 7 of 9…)" → the 2.9 s clip of "AI is changing everything" at 0:42 |

**The image cut-off is thin:** the strongest unrelated photo scored 0.55, and 0.55–0.58 all gave the same result. Fifteen
negatives is a small sample.

**Test material for questions:** a 102-second talk generated with macOS `say` (`eval/talks/sleep_talk.txt`), so every
answer is known. Synthetic speech is easier to transcribe than a real speaker.

## Other decisions

- **Fixed-template replies** (as for `/ask`, ADR 0003). The model routes; it never writes to the user.
- **Memory** in Redis for 24 hours: last request, last action, last clip, an open offer (for one turn only), and clips
  shown ("another one" skips them).
- **Blank frames get no caption or vector.** A plain grey background was captioned "a person in a dark room holding a
  smartphone", so "a horse" matched it. Blank frames measured 0.0 and the least varied real frame 13.5, so the cut-off is 4.0.
  Search results were unchanged.
- **The model is a setting (`LLM_PROVIDER`).** Locally it's Ollama; on AWS it's Amazon Bedrock (the Converse API, with a
  forced tool call for the form). The Bedrock path has **unit tests only** (a stubbed client), since there are no AWS
  credentials here.
- **Telegram bot:**
  - it's a client of `/chat` and long-polls Telegram, so it needs no public URL;
  - clips are uploaded from S3, because Telegram can't open localhost links;
  - the token is redacted from all logs.
- **Web page:** one static file at `/app`. It was driven in headless Chrome: clips play, photo search and dark mode work,
  and there are no console errors.

## Consequences

- **Latency:** 1–3 s when the rules decide or caches hit; **8–15 s** when the model runs twice (routing plus understanding
  or answering). The model runs on the CPU inside Docker. Two alternating prompts evict each other from Ollama's single
  prompt cache (4–5 s to re-read each), and more slots didn't help in Ollama 0.11.2. **Production needs a GPU or Bedrock**;
  a Mac-native Ollama would help only locally.
- **Accuracy on free phrasing is ~0.8–0.85 per turn.** The page shows how each turn was decided, so a wrong turn is
  visible and easy to correct.
- **Evaluation numbers move as the library grows.** Three balloon videos moved one query ("drone footage above the
  coast") from rank 2 to 3.
- **Found by the dashboards and live use, not by tests:**
  - invented captions on blank frames;
  - a routing reply cut off by the token limit;
  - my own test loop never sending the conversation id (zsh doesn't split `${CID:+-F …}`). Live tests now go through Python.

## When to revisit

- **Bedrock is available:** run the same evaluations (routing, conversations, questions, understanding) on a hosted
  model, and compare accuracy, time and cost before switching.
- **Real conversations exist:** measure on those; the evaluation turns were written by us.
- **A GPU:** remeasure latency; the ~2 s per model call and the prompt-cache eviction change completely.
