# Video RAG: project deep dive

A complete technical walkthrough of the system: what it does, how every component works, why each design was chosen
over the alternatives, **how each component is tested and measured**, what broke along the way, and what's left.
Section 20 is a bank of likely interview questions with answers; Section 21 is a one-page numbers cheat sheet.

---

## Contents

1. [The pitch](#1-the-pitch)
2. [Requirements and constraints](#2-requirements-and-constraints)
3. [Architecture](#3-architecture)
4. [Ingestion pipeline](#4-ingestion-pipeline)
5. [Indexing and search](#5-indexing-and-search)
6. [Request understanding](#6-request-understanding)
7. [Clips](#7-clips)
8. [Question answering](#8-question-answering)
9. [Search by photo](#9-search-by-photo)
10. [The chat agent](#10-the-chat-agent)
11. [Uploads, Pexels fetching and progress](#11-uploads-pexels-fetching-and-progress)
12. [Interfaces: web chat, Telegram bot, REST API](#12-interfaces-web-chat-telegram-bot-rest-api)
13. [Observability: tokens, traces, metrics](#13-observability-tokens-traces-metrics)
14. [Caching](#14-caching)
15. [Model backend and the AWS path](#15-model-backend-and-the-aws-path)
16. [Testing and evaluation strategy](#16-testing-and-evaluation-strategy)
17. [Performance and cost](#17-performance-and-cost)
18. [Bugs and lessons](#18-bugs-and-lessons)
19. [Limitations, security gaps and next steps](#19-limitations-security-gaps-and-next-steps)
20. [Likely interview questions](#20-likely-interview-questions)
21. [Numbers cheat sheet](#21-numbers-cheat-sheet)
22. [Glossary of metrics](#22-glossary-of-metrics)

---

## 1. The pitch

**30 seconds.** Video RAG finds the exact moment you're looking for inside a video and returns just that clip, a few
seconds long. You can ask by what's **said** ("the part where the host says AI is changing everything"), by what's
**shown** ("a dog in the snow", "a beach with no people"), ask a **question** ("how long is a sleep cycle?"), or send
a **photo**. It works on your own uploads and on a stock-footage library it can grow from Pexels. It runs fully
locally, including the language model, as a web chat, a Telegram bot and a REST API.

**2 minutes: what's technically interesting:**
- **Multimodal indexing.** Every video is indexed twice:
  - frames: scene detection → keyframes → a CLIP vector and a caption from a vision-language model;
  - speech: Whisper with a timestamp on every word.
- **Word-accurate clips.** Search locates the exact words, and ffmpeg re-encodes so the clip starts on the first
  word, not the nearest video keyframe.
- **A small local LLM that's never trusted blindly.** It fills structured forms (JSON schema), and plain-code
  **guards** check its output against the user's own words, with rules as the fallback. The chat agent decides with
  **rules first**, and uses the model only when rules can't.
- **Evaluation-driven.** Every component has a labelled test set, with labels fixed before running; held-out sets
  are reported separately from anything that informed a fix. Decisions are written up as ADRs with the numbers.
- **Production-shaped.** Postgres is the source of truth and the search index is rebuildable, blue/green. The pipeline
  is idempotent, with a dedicated queue for interactive work. Tracing crosses Celery. There's a token and cost ledger,
  caches with correct invalidation, dashboards as code, and a switchable model backend (Ollama or Bedrock).

---

## 2. Requirements and constraints

**Functional:**
1. Return a clip for a visual request ("a dog").
2. Return the part of an uploaded video where specific words are said.
3. Handle negation ("no people"), synonyms ("canine"), style ("drone footage").
4. Say "nothing matched" instead of returning the nearest wrong thing.
5. Answer questions from the videos with citations, and never from the model's general knowledge.
6. Search by photo.
7. Hold a conversation: follow-ups, "another one", questions about "that clip".
8. Fetch missing stock footage, **only with consent**.
9. Accept uploads in the chat, show progress, and accept a request before processing finishes.

**Non-functional:**
- Runs entirely on one laptop: Apple M4 Pro, 24 GB RAM, Docker given 16 GB, **CPU-only inference** (Docker on macOS
  can't use the Apple GPU).
- Requests stay on the machine: a local LLM, no hosted APIs by default.
- Deployable to AWS later: the model backend is swappable, and storage uses the S3 API.
- Every quality claim is backed by a reproducible measurement (`make eval-*`).

**Out of scope for now:** authentication, multi-tenancy (uploads are currently shared; see §19), long-form video
(hours), non-English speech, GPU inference.

---

## 3. Architecture

### 3.1 Services (Docker Compose, 14 containers)

| Service | Role | Why separate |
|---|---|---|
| `api` | FastAPI, 4 uvicorn workers | the request path |
| `worker` | Celery, `celery` queue, concurrency 1 | heavy processing: Whisper holds ~1 GB, so one job at a time |
| `clip-worker` | Celery, `clips` queue only, concurrency 2 | **interactive** clip cutting never waits behind a video being processed |
| `embedder` | CLIP ViT-B/32 (laion2b) HTTP service | one model copy shared by the API (text queries) and worker (frames); independent scaling |
| `ollama` | local LLM server (`qwen2.5vl:3b`) | captions, request understanding, answers, routing |
| `postgres` | source of truth: videos, segments, token ledger, traces | relational, transactional, rebuildable index |
| `opensearch` (+ dashboards) | BM25 + HNSW kNN search | hybrid retrieval in one engine |
| `redis` | db0 cache + conversation memory, db1 broker, db2 results | |
| `seaweedfs` | S3-compatible object storage | videos, keyframes, clips, chat images (MinIO stopped publishing community images) |
| `airflow` | schedules: daily Pexels ingestion, trace retention | DAGs only call the API |
| `prometheus`, `grafana` | metrics, dashboards | |
| `telegram-bot` | Telegram client of `/chat` | thin client, no agent logic |

### 3.2 Data flow

```text
                       ┌─────────────────────── worker (Celery) ────────────────────────┐
POST /videos ─▶ S3 ─▶  │ download → ffprobe → scene cuts → shots → keyframes (S3 frames/) │
Pexels ─▶ download ─▶  │ → CLIP vectors (embedder) + captions (Ollama)                    │─▶ Postgres segments
                       │ → audio → Whisper (word timestamps) → overlapping speech windows │─▶ OpenSearch index
                       └───────────────────────────────────────────────────────────────────┘

/chat turn ─▶ decide (rules │ LLM + guards) ─▶ tool ─▶ template reply ─▶ memory (Redis)
   find_clip:  understand ─▶ search ─▶ clip range ─▶ S3 cache? ─▶ clip-worker ffmpeg ─▶ presigned URL
   answer:     excerpts ─▶ LLM (cite) ─▶ guards ─▶ sentence clip
   image:      CLIP(photo) ─▶ kNN over keyframes ─▶ cut-off 0.57 ─▶ clip
```

### 3.3 Data model

| Table | Holds | Notes |
|---|---|---|
| `videos` | source (pexels or upload), Pexels id and credit, S3 key, duration, fps, has_audio, language, **status + stage + error** | unique (source, source_id) makes Pexels ingestion idempotent |
| `segments` | kind (speech or visual), start/end, transcript + `words` (JSON, per-word times), keyframe key, caption, CLIP vector (JSON), model names | the searchable unit; Postgres keeps vectors so the index can be rebuilt without re-running models |
| `llm_calls` | every model call: operation, outcome, origin, tokens in/out, load/prompt/output seconds, **cost priced at write time**, video, trace id, saved tokens (cache hits) | the token ledger |
| `trace_spans` | timed steps: trace id, parent, name, service, duration, status, attributes | 14-day retention |

Migrations: Alembic (5 revisions), applied automatically at startup, with a test that migrations produce exactly what
the ORM models describe.

**S3 layout:** `raw/{video}/source.mp4`, `frames/{video}/{ms}.jpg`,
`clips/{video}/{start_ms:09d}-{end_ms:09d}.mp4` (the cache key), `chat-images/{uuid}.jpg`.

### 3.4 Why this shape (trade-offs)

| Decision | Pros | Cons |
|---|---|---|
| Postgres as the source of truth, OpenSearch as a derived index | the index can be rebuilt or migrated at any time; no dual-write consistency problem | two stores; indexing is an extra pipeline stage |
| Celery + Redis instead of in-process background tasks | retries, `acks_late` (a crashed worker's job is redelivered), separate queues, horizontal scaling | more moving parts |
| A dedicated clip queue and worker | an interactive cut never waits behind a 5-minute processing job (measured: 0.55 s cut while the processing worker was busy) | one more container; if it's down, `/ask` times out (504), and health reports it |
| CLIP as its own service | one model in memory for API + worker; independent scaling; language-agnostic interface | a network hop (~60–90 ms per text embedding) |
| One local LLM (`qwen2.5vl:3b`) for everything | no model swapping (Ollama holds one model); private; free | slow on CPU; weak at tool use; ~5 GB RAM |

---

## 4. Ingestion pipeline

### 4.1 Steps

1. **Store and queue** (`POST /videos`, or a Pexels download). Bytes go to S3 first, then the row is committed, then
   the job is queued. **Commit before enqueue**, so the worker always finds the row. The queue message carries only
   the id.
2. **Pexels:** choose the best MP4 at or below 720p, and skip anything over 60 s (cost scales with duration).
   Idempotent: already-ingested Pexels ids are skipped.
3. **Probe** (ffprobe): duration, resolution, fps, audio.
4. **Scene detection:** ffmpeg `select='gt(scene,0.3)'` on a 320-px-wide copy, for speed.
5. **Shots:**
   - cuts less than 1 s apart are ignored (flicker, fast pans);
   - a tail under 1 s merges into the previous shot;
   - shots over 10 s are split evenly, so each keyframe represents its segment.
6. **Keyframes:** the middle of each shot, scaled to fit 640×640, stored in S3.
7. **Blank frames:** grey-level standard deviation ≤ 4.0 means no caption and no vector (see §18).
8. **CLIP vectors:** batched 32 per call to the embedder.
9. **Captions:** `qwen2.5vl:3b`, image downscaled to 448 px (~4.5 s per frame on CPU instead of 10–30 s),
   temperature 0, ≤ 80 tokens.
10. **Speech:** 16 kHz audio → faster-whisper `small`, int8, beam 5, `word_timestamps=True`, VAD filter (skips
    silence and music, so fewer hallucinated words).
11. **Speech windows:** **15 s windows every 10 s (5 s overlap)**, so any phrase shorter than 5 s appears whole in at
    least one window. Word timings are kept for exact cutting.
12. **Save segments** (replace all of the video's segments), then **index**, then status `ready`. "Ready" means
    searchable.

### 4.2 Reliability

- **Idempotent:** re-running deletes old frames and replaces segments, so Celery redelivery and manual reprocessing
  are safe.
- **Retries** only for transient errors (network, S3, OpenSearch connection, model unavailable), with exponential
  backoff (10 × 2ⁿ s, 3 attempts). Bad input (corrupt file, missing row) fails fast, with the error stored on the video.
- **Visible progress:** `videos.stage` is updated at each step (captioning even reports "captioning 3/11"). This
  drives the user-facing progress messages (§11).
- A failed **re-enrichment** doesn't mark a working video as failed; it stays searchable as before.

### 4.3 How it's tested

| Test | What it checks |
|---|---|
| Unit: segmentation | shots from cut times (min/max length, tail merge, splitting); speech windows (overlap; every short phrase is contained) |
| Unit: ffmpeg parsing | scene-change timestamps parsed from ffmpeg's output |
| Unit: ingestion | Pexels dedupe (existing ids skipped), best-rendition choice, commit-before-enqueue |
| Unit: enricher | vectors and captions attached; **blank frames skipped** (flat grey vs noise images); per-frame progress calls; one usage row per caption |
| Unit: migrations | Alembic upgrade == ORM metadata (no drift) |
| Live | re-processing a video produced identical captions (58/58) and segments; vectors differed by ≤ 1e-6 (batch vs single embedding: floating-point noise) |

### 4.4 Pros and cons

| Choice | Pros | Cons / alternatives |
|---|---|---|
| Scene detection + one keyframe per shot | cheap; representative; typically 2–11 frames per video | misses changes within a shot (hence the 10 s split); dense sampling (1 fps) would cost 5–10× the captioning |
| Whisper `small` int8 on CPU | good accuracy/speed trade-off on CPU; word timestamps | English-focused in practice; larger models are slower |
| Overlapping windows (15/10) | a quote never falls across a boundary | duplicate text: dedupe at search time; the QA transcript is rebuilt from word timings |
| Captions from a local VLM | adds a keyword signal for details CLIP misses | ~4.5 s per frame (**92% of processing time** for a short silent video); can hallucinate |

---

## 5. Indexing and search

### 5.1 The index

- **Alias** `video_segments` → `video_segments_v2`. Clients only use the alias. Changing the mapping means a new
  version, filled from Postgres, then an atomic alias switch (blue/green, old index kept for rollback).
- **`dynamic: strict`:** unknown fields are rejected.
- **Transcript indexed twice:**
  - `text` with a custom analyser (lowercase + asciifolding, **keeps stop words and positions**), for exact phrases
    like "it is what it is";
  - `text.stemmed` with the English analyser ("changing" ~ "change"), for keywords.
- **Caption** (English analyser), and **video title** (English + a keyword subfield).
- **`image_embedding`:** `knn_vector`, 512 dimensions, **HNSW** (m=16, ef_construction=128), cosine similarity,
  **Lucene engine**, chosen because it applies filters (video, kind, source) *inside* the kNN search rather than
  filtering a small top-k afterwards.
- **Denormalised video fields** (title, author, duration, source): results render without a database round trip.

### 5.2 Retrieval strategies

| Strategy | Query | Used for |
|---|---|---|
| **Phrase cascade** | exact `match_phrase` (slop 0) → slop 2 → fuzzy (fuzziness AUTO, 75% of words) | quotes; stops at the first strategy that finds something |
| **Word-level phrase location** | inside the matched window, align the phrase to Whisper's words (sequence matching, token similarity ≥ 0.75, e.g. "chaining" ≈ "changing") | exact start/end times plus a match score |
| **Keyword (BM25)** | `multi_match` best_fields over `text.stemmed`, `caption`, `video_title^2`, tie_breaker 0.3 | topics, visual details |
| **Vector** | CLIP text embedding of the request (request words like "give me a clip of" stripped first) → kNN | visual meaning, synonyms, style |
| **Hybrid** | keyword + vector fused with **Reciprocal Rank Fusion, k = 60** | visual requests |

**Filters and cut-offs:**
- Vector hits need cosine ≥ 0.15.
- A hit found **only** by vector search needs cosine ≥ **0.20**, chosen by a sweep (§6.4).
- Results are over-fetched ×3, overlapping windows are collapsed, and there's one result per video (across the
  library) or several from one video (inside an upload).
- Exclusions ("no people") drop hits whose caption, transcript or title mentions them.

**Why RRF rather than a weighted sum of scores:** BM25 scores and cosine similarities aren't on comparable scales,
and their distributions change per query. RRF uses only ranks, needs no tuning, and is robust. The cost: it ignores
how *confident* each retriever is, which is why the vector-only cut-off exists.

### 5.3 How search is tested

**Test set** (`eval/queries.json`):
- **36 queries**, with relevance labelled **per video by Pexels id**, decided by **looking at the actual keyframes**,
  not titles or captions.
- **16 basic + 20 hard**, grouped by the weakness they probe: style (3), detail (6), synonym (4), count (4),
  colour (2), negation (1).
- Plus **12 requests that should return nothing** (`eval/no_answer.json`): greetings, and subjects not in the library
  ("an elephant", "a spaceship landing on mars").

**Metrics:**
- **Recall@5:** relevant videos in the top 5 / min(5, number of relevant videos);
- **MRR:** 1 / rank of the first relevant video, averaged;
- the same **per category**;
- **false answers:** how many of the 12 returned anything.

**Results** (`make eval`):

| Mode | Recall@5 | MRR | Style | Detail | Synonym | Negation | False answers |
|---|---|---|---|---|---|---|---|
| keyword | 0.717 | 0.722 | 0.33 | 1.00 | 0.00 | 0.50 | 4/12 |
| vector (CLIP) | 0.989 | 0.940 | 1.00 | 0.83 | 1.00 | 0.50 | 9/12 |
| hybrid | 0.970 | 0.935 | 0.83 | 1.00 | 1.00 | 0.50 | 7/12 |
| **understood** (what the app uses) | **0.970** | **0.949** | 0.83 | 1.00 | 1.00 | **1.00** | **2/12** |

**How to read it:**
- keyword and CLIP are **complementary**: CLIP wins on style and synonyms (in the pixels, not the words); keyword wins
  on named details;
- vector search alone "answers everything" (9/12 false answers), because kNN always returns a nearest neighbour;
- request understanding (§6) fixes negation and most false answers.

**ADR 0001, a rejected idea:** embedding the captions as a third retriever. On 36 queries it changed 2 (one better,
one worse); overall MRR 0.940 → 0.945 (noise); hard-query MRR 0.917 → 0.902 (worse). Why:
- caption embeddings overlap with both existing retrievers and **inherit caption errors**. A cat seen from behind was
  captioned "a light brown dog"; caption search then ranked the cat video first for "a dog";
- it would cost a second model, a migration, a new index version and an extra query per request.

Rejected; the experiment is kept reproducible (`make experiment-captions`).

**Unit tests:**
- query bodies (filters inside kNN, exact field for phrases);
- the phrase locator (misheard words, "state-of-the-art" split into tokens);
- RRF;
- dedupe of overlapping windows;
- the vector-only cut-off;
- exclusion matching;
- the index lifecycle (alias, blue/green migration, writing to an older mapping).

---

## 6. Request understanding

### 6.1 The intent

Every request becomes an `Intent`:
- `type`: **quote** (words someone says), **topic** (something discussed), or **visual** (something shown);
- the text for that type;
- `exclude`: a list of things that must not appear.

Search then depends on the type:
- **quote** → phrase cascade over speech;
- **topic** → keyword search over speech;
- **visual** → hybrid search over keyframes only (a transcript saying "dog" isn't footage of a dog).

If a quote or topic search finds nothing, it's retried as visual.

### 6.2 Design: the LLM proposes, the guards check, the rules catch

```text
request ─▶ LLM (Ollama JSON-schema mode, temperature 0) ─▶ guards ─▶ intent
             │ error / invalid              1. type:      quote/topic need a speech word ("says", "talks about"…) or quote marks
             ▼                              2. fields:    keep only the field for the chosen type
           rules ◀──────────────────────── 3. grounding: the text must use the user's own meaningful words
                                                          (≥ 0.8 of words for quotes, ≥ 0.5 otherwise), else rules
                                            4. exclusions: the LLM's ∪ the rules' "no X / without X", only words the
                                                          user said, overlapping terms merged
no subject ("exclude people", "hi") → ask "What would you like to see or hear?" instead of searching
```

**Why an LLM at all:** rules handled the phrasings they were written for (1.00) but only **0.17** of free phrasings.
**Why guards:** the 3B model alone
- invented details ("a dog" → "a dog **sitting on a couch**");
- dropped negations;
- **copied the prompt's own example** ("exclude people" → "a horse running");
- routed bare nouns like "canine" to speech search.

**Why JSON-schema mode:** the reply always parses. Ollama constrains decoding to the schema.

### 6.3 How it's tested

**Test set** (`eval/intents.json`, 58 requests):

| Category | n |
|---|---|
| phrasings the rules were designed for (5 categories) | 30 |
| free phrasing (written to avoid the rule patterns) | 12 |
| held out (labels fixed before any run) | 8 |
| no subject (exclusion-only, placeholders, greetings) | 8 |

**Metrics:** per request, is the **type** right, is the **text** right (compared after lowercasing and dropping
articles and plurals), are the **exclusions** right, and are **all three** right. Also latency (p50) and tokens.

| System | All correct | Free phrasing | Held out | p50 |
|---|---|---|---|---|
| rules only | 0.78 | 0.17 | 0.75 | 0 s |
| **LLM + guards + rules** | **0.81** | **0.58** | 0.62 → 0.75* | 1.2 s |

\* The held-out run exposed a missing speech word ("tells"). It was fixed, and the 0.75 is **reported as measured
after seeing the held-out set**, so it's no longer unseen.

### 6.4 Effect on search, and the cut-off sweep

| Mode | Recall@5 | MRR | Negation | False answers |
|---|---|---|---|---|
| hybrid on the raw request | 0.970 | 0.940 | 0.50 | 7/12 |
| **understood** | 0.970 | **0.954** | **1.00** | **2/12** |

(MRR 0.954 at the time; 0.949 now, because three new videos moved one query down a rank; see §18.)

**False answers were cut in two ways:**
- **Fix A:** chit-chat counts as "no subject".
- **Fix B:** a sweep of the vector-only cut-off:

| Cut-off | Recall@5 | MRR | False answers |
|---|---|---|---|
| 0.15 | 0.989 | 0.954 | 5/12 |
| **0.20 (chosen)** | 0.970 | 0.954 | **2/12** |
| 0.22 | 0.906 | 0.926 | 2/12 |
| 0.25 | 0.860 | 0.870 | 2/12 |

0.20 removes three false answers at the cost of one relevant video on one query; stricter values only lose recall.

**Known misses:**
- descriptive words dropped ("city traffic at night" → "city traffic");
- spoken requests without a speech word ("which part covers…") go to visual search;
- 2 of 12 false answers remain ("someone playing the piano" matches common words in keyword search).

**Unit tests:** each guard; grounding of quotes with quote marks; exclusion merging; the rule parser's patterns; LLM
failures (HTTP error, invalid JSON) → rules; usage outcome recorded after the guards (accepted, adjusted, rejected,
invalid).

---

## 7. Clips

### 7.1 Decisions (ADR 0003)

| Question | Decision | Rejected, and why |
|---|---|---|
| What's returned | a **real MP4** in S3 via a presigned URL | `#t=start,end` on the full video: not a shareable clip |
| How it's cut | **re-encode** (libx264 veryfast, CRF 23, AAC, `+faststart`) | **stream copy** can only start at a video keyframe: asked for 8.73–11.61 s (2.88 s), a copy started at 7.96 s and ran 3.11 s |
| Boundaries | quote: **words ± 0.75 s**; visual: **5 s** centred on the matched keyframe, inside its shot; topic or answer: **the sentence** (from word timings) ± 0.75 s, ≤ 8 s | whole shots (≤ 15 s) and whole excerpts (~20 s) felt like short videos, not clips (user feedback) |
| Repeat requests | cached by key `clips/{video}/{start_ms}-{end_ms}.mp4` (times rounded to 10 ms) | re-cutting every time |
| Who cuts | the `clip-worker` on its own queue | the processing worker: users would wait behind minutes-long jobs |
| Text with the clip | a **fixed template** from search results ("At 0:09–0:11 in "host_talk": "AI is changing everything."") | LLM-written prose: seconds slower on CPU, and it could state things not in the video |

**The sentence finder:**
- Whisper words are split into sentences at `. ? !`;
- pick the sentence sharing the most meaningful words with the topic, or with the question plus its answer;
- join a sentence under 2.5 s with the next one.

### 7.2 How it's tested

- **Accuracy, end to end:** the cut quote clip was **transcribed back with Whisper**. "AI" lands at **9.49 s** of video
  time, where search placed it at **9.48 s**.
- **`+faststart`:** the MP4's index (`moov` atom) sits at byte 36 instead of at the end (byte 28,431 of 31,674 without
  it), so playback starts before the download finishes.
- **Isolation:** with the processing worker busy on a 46 s video, a new clip was delivered in **0.55 s**.
- **Speed:** a fresh short cut takes 0.1–0.9 s; a cache hit takes ~1–8 ms.
- **Lengths, live, after the change:** "a dog in the snow" 5.0 s (was 10 s); "how long is a sleep cycle?" 4.9 s (was
  ~15 s); the caffeine topic 5.3 s; a quote 2.9 s.
- **Unit tests:**
  - boundaries (quote padding, a visual clip centred and kept inside its shot, full length kept near the end of a
    shot, focus sentence, degenerate ranges);
  - cache keys;
  - the sentence finder (topic sentence found; a short sentence joined with the next; no match → None);
  - re-joining Whisper tokens ("half" + "-life" → "half-life").

**Pros and cons:** re-encoding costs CPU (~0.1–1 s per clip) but is frame-accurate. The worker downloads the whole
source for each cut: fine for short videos, wasteful for hour-long ones (ffmpeg could read a byte range from a signed
URL instead).

---

## 8. Question answering

### 8.1 Design

```text
question ─▶ excerpts
              one video:    its whole transcript rebuilt from word timings (no duplicate text from overlapping
                            windows), cut into ~20 s chunks at sentence ends, plus its keyframe captions (≤ 8 excerpts;
                            if more, the ones sharing most question words)
              the library:  keyword search after dropping question words ("what does the host say about caffeine"
                            → "caffeine"); top 8 speech windows / captions
         ─▶ LLM (JSON schema {found, answer, cited[]}, temperature 0, ≤ 160 tokens):
              "answer ONLY from these numbered excerpts; if they don't contain it, found=false"
         ─▶ guards: found=false → "I couldn't find that in the videos."
                    citations must be excerpt numbers it was given
                    every number in the answer must appear in the cited text ("ninety" = "90")
         ─▶ answer + citations + a clip of the sentence that answers
```

**Why this design:** the most damaging failure is a **plausible answer from general knowledge** presented as coming
from the video. Excerpts-only prompting, required citations and the number check make that visible and blockable.

**Pros and cons:**
- A single-video question sees the whole (short) transcript: robust.
- A library question depends on keyword retrieval: it misses paraphrases, and long videos would need semantic
  retrieval.
- The number guard is narrow; it doesn't catch invented non-numeric claims (a word-overlap guard is the next step).

### 8.2 How it's tested

**Test material:** a 102-second talk with **known content**, generated with macOS `say` from a written script
(`eval/talks/sleep_talk.txt`). Every correct answer is known exactly.
- **Limitation:** synthetic speech is easier to transcribe than a real speaker. Whisper made one error ("replays" →
  "replace").
- The script deliberately **avoids words used in the search test set** ("night" was reworded), so adding the talk
  couldn't disturb search results. Checked afterwards: unchanged.

**Test set** (`eval/questions.json`, 24 questions, labelled before any run):

| Category | n | Example |
|---|---|---|
| facts from the talk | 12 | "What is the half-life of caffeine?" |
| across the library | 3 | "What does the host say AI is changing?" |
| **not in the video** | 5 | "Does alcohol affect sleep?" (a model knows; the talk never mentions it) |
| not in the library | 2 | "What is the capital of France?" |
| visual (from captions) | 2 | "What colour is the dog?" |

**Metrics:**
- **status accuracy:** answered vs not found, as expected;
- **facts right:** the answer contains every required fact, e.g. "90" or "ninety";
- **citation right:** a cited excerpt contains the labelled evidence;
- **made up:** answered when it should have said not found. This matters most;
- **refused an answerable question;**
- p50 latency, tokens per question, and how many answers the guards rejected.

**Results:**

| Metric | Result |
|---|---|
| made-up answers | **0 of 7** |
| refused answerable questions | 0 |
| facts right | **16/17** |
| citations right | **17/17** |
| p50 | 1.9–2.3 s |
| tokens per question | ~515–606 |

**The one miss** is a label issue, **kept as a miss**. The label required "everything"; the model answered from the
next sentence ("AI is changing how we write, how we code, and how we learn"), which is true to the video. Labels are
never edited after seeing results.

**Unit tests:** each guard (not found, an invalid citation, an invented number, an invalid reply); no excerpts means no
model call; transcript rebuilding (overlap removal, sentence chunks); retrieval text; number normalisation; the clip is
the answering sentence.

---

## 9. Search by photo

### 9.1 Design

The photo gets a CLIP **image** vector, compared with keyframe vectors: the same model, the same space. Photo-to-frame
similarities run much higher than text-to-frame ones, so there's **its own cut-off (0.57)**.

### 9.2 How it's tested

**Test set** (`eval/images.json`, 45 images):

| Kind | n | Relevant = | Why |
|---|---|---|---|
| near-duplicate | 10 | the video it came from | a frame taken ¼ into a shot, **not** the stored keyframe |
| semantic | 20 | the videos labelled relevant for the matching text query in `eval/queries.json` | Pexels **photos** of subjects the library has (2 per subject, first results, not hand-picked) |
| negative | 15 | nothing | photos of subjects the library lacks (elephant, piano, horse…) |

- **Label check:** all 35 photos were **reviewed on a contact sheet** before use (each shows its subject; two were
  flagged as ambiguous in advance).
- **Metrics:** **hit@1** (the best video is relevant and above the cut-off), per kind; **false answers** (a negative
  still gets a clip); swept over cut-offs from 0.40 to 0.80.

**Distribution of best-match similarity:**

| Kind | min / median / max |
|---|---|
| near-duplicate | 0.843 / 0.944 / 0.985 |
| semantic | 0.511 / 0.684 / 0.800 |
| negative | 0.300 / 0.469 / 0.550 |

**Sweep:** 0.55–0.58 all give **semantic 18/20, near-duplicates 10/10, false answers 0/15**. **0.57** sits mid-way.
- The two misses are exactly the two flagged in advance (a blurry night-lights photo, fried bananas).
- **Honest caveat:** the margin is thin (the strongest negative is 0.550), and 15 negatives is a small sample.

---

## 10. The chat agent

### 10.1 Measured first: can the local model choose tools?

- **Native tool calling:** Ollama reports `qwen2.5vl:3b` "does not support tools".
- **So it's the same pattern as request understanding:** the model fills a **JSON form** (`{action, text,
  about_last_video}`), and guards check it.

**Routing test sets:**
- **40 labelled turns:** each has a context (last request, last clip, an open download offer, an image attached) and
  a message, labelled with the expected action and required words.
- **Held-out 1 (20 turns)** and **held-out 2 (20 turns)**, free phrasing, written before the systems they evaluate
  were run.

**Metrics:**
- **action accuracy**;
- **all correct** (action + text: it contains the required words, lacks forbidden ones, keeps exclusions);
- **unsafe:** a Pexels download without consent (**must be 0**);
- the share of turns that call the model;
- latency and tokens.

| System | First set | Held-out 1 | Held-out 2 | Unsafe |
|---|---|---|---|---|
| rules only | 1.00 | 0.60 | 0.75 | 0 |
| model only | 0.70 | 0.75 | 0.85 | **1** |
| model + guards | 0.85 | 0.70 | — | 0 |
| **rules first + model + guards** | **0.95** | **0.85** | **0.95** (**0.80 unseen**) | **0** |

**How to read it:**
- **The rules overfit** to the phrasings they were written with (1.00 → 0.60 on new ones).
- **The model generalises better** but:
  - writes its own answer into the request ("A puppy is a cute and cuddly pet…");
  - copies prompt examples (downloaded "horse" after a bare "yes");
  - drops follow-up context.
- **Rules first:** clear patterns are decided instantly (30–50% of turns never call the model). The model handles
  open language, and guards check it.
- Held-out 2's **clean** score is 0.80; the 0.95 is after one repair designed with its misses in view.

### 10.2 The guards

| # | Guard | Found by |
|---|---|---|
| 1 | download only after an accepted offer or an explicit "download from Pexels" | a download on a bare "yes" |
| 2 | image search needs an image; an image alone means image search | image search chosen for "a beach" |
| 3 | a bare yes/no with nothing to answer isn't a request | |
| 4 | "where…", "the part…", "show…" are requests to *see* a moment | routed to questions |
| 5 | a question stays the user's own words (the model's action is kept) | the model writing answers as the question |
| 6 | the request must use words from the message or the conversation | invented requests |
| 7 | a follow-up keeps the last request ("now without people" → "a beach without people") | **live use** |
| 8 | only a message pointing at the clip ("that clip") is scoped to it | **live use**: "a beach" narrowed to the dog video |
| 9 | a new request uses the message's own words | "show me cats instead" → "a unicorn" again |

### 10.3 The graph (LangGraph)

```text
message:  decide ──(reply)──────────────────────────▶ respond ─▶ END
             └──(tool)──▶ act ──────────────────────▶ respond
video:    upload (store + queue) ─────────────────────────────────▶ END
poll:     check (upload or download progress) ──(pending)─────────▶ END
             └──(done, a request waiting)──▶ act ─▶ respond ─────▶ END
```

**Why LangGraph:** the flow branches (reply vs tool, fetch → wait → retry, upload → wait → answer), with multiple
entry points. Explicit state and nodes make each path testable. A plain loop would have worked for the first version.

**Memory** (Redis, 24 h), kept deliberately small because a 3B model reads slowly:
- last request, last action, last clip's video;
- an open Pexels offer (**one turn only**);
- clips already shown ("another one" skips them);
- uploading/fetching state, and the uploaded video in focus.

**Replies are fixed templates.** The model never writes to the user, so a reply can't claim anything the tools didn't
return.

### 10.4 Whole-conversation evaluation

**Test set** (`eval/conversations.json`):
- **12 conversations, 28 turns**, through the real `/chat` endpoint (routing + tools + memory + replies), labelled
  first;
- per turn: the expected action(s), a regex the clip's video title must match, reply must-have and must-not-have
  patterns, and "a different video than the last turn".

| Run | Turns right | Conversations right | p50 / p95 | Tokens per turn |
|---|---|---|---|---|
| **first run (unseen)** | **23/28** | **8/12** | 8.9 s / 13.2 s | 562 |
| after five general fixes (seen) | 28/28 | 12/12 | 5.4 s / 12.9 s (warm caches) | 399 |

**The five misses and their general causes:**
1. a routing reply **cut off mid-JSON** by a 60-token limit (now 150);
2. "what about X" should repeat the previous *kind* of request (clip vs question);
3. "and a X" names a new subject, not a refinement;
4. the model repeating the last request (guard 9);
5. "show me that part" after an answer.

**Unit tests:**
- the rules/model split (clear patterns never call the model);
- each guard;
- memory, including a Redis outage;
- a one-turn offer expiring;
- "another one" skipping clips already shown;
- "that clip" scoping;
- the fetch/upload flows (pending → done, a request queued during processing, not-found in an upload, timeout,
  failure);
- tool failures → an apology, not a crash.

---

## 11. Uploads, Pexels fetching and progress

### 11.1 Uploading in the chat

1. The video is attached → stored and queued (no model decision) → "Got your video. Tell me what to find in it; you
   can ask now."
2. **A request can come any time:**
   - still processing → the request is **queued in the conversation** (it survives a page reload or bot restart), and
     the reply shows the current step;
   - ready → searched immediately.
3. When processing finishes: "Your video is ready." + the clip for the waiting request.
4. The upload becomes the **conversation's focus**: every request and question searches only it. **Not found → "I
   couldn't find X in your video"**, with no library results and no Pexels offer (a result from someone else's video
   would be misleading).

**Live:** the upload was accepted in 0.2 s; a separate request 30 s later → "still processing (Step 7 of 9:
transcribing the speech · 0:30)" → the 2.9 s clip of "AI is changing everything" at 0:42.

### 11.2 Progress ("is it processing, or did the bot die?")

The pipeline's `stage` is mapped to a numbered step in plain words, with a clock:

```text
Waiting in line: another video is being processed first · 0:12
Step 6 of 9: describing the frames (3 of 11) · 0:42
Step 7 of 9: transcribing the speech · 0:55
✅ Processed in 1:12
```

- **Telegram:** **one status message edited in place** (`editMessageText`) every ~10 s. The clock proves it's alive,
  without spamming the chat.
- **Web:** one progress bubble with the same text.
- **Polls where nothing happened** discard their trace (§13), so polling every few seconds doesn't flood the trace
  table.

### 11.3 Fetching from Pexels (with consent)

"a hot air balloon" → no match → "Want me to download some from Pexels? It takes a minute or two." → "yes please"
(the rules decide instantly: a yes to our open offer) → 3 videos queued → polls: 0/3 → 2/3 → done → the original
request is retried → clip. **Live: 78 s.**

The retry can't hit a stale cached "no match", because every index write bumps the answer cache's index version (§14).

---

## 12. Interfaces: web chat, Telegram bot, REST API

- **Web chat** (`/app`): one static HTML file, no build step.
  - an intro explaining the project, with four cards;
  - 📎 accepts videos or photos;
  - clips play inline;
  - citations and "decided by rules/llm" per reply;
  - progress polling;
  - dark mode, phone width.
  - **Tested in headless Chrome (Playwright):** clips load (readyState 4), no console errors, screenshots at desktop
    and phone width in both themes. This found oversized portrait clips, an unreadable dark-mode link and a missing
    favicon.
- **Telegram bot:** long polling (no public URL needed), one conversation per chat, `/new`.
  - Clips are **uploaded from S3**, because Telegram's servers can't open `localhost` presigned URLs.
  - Videos over **20 MB** are refused politely (the Bot API download limit).
  - **The token is in every Bot API URL**, so HTTP request logging is off and a logging filter redacts the token from
    all log lines and tracebacks (unit-tested).
  - It's a **thin client of `/chat`**, so every agent change reaches both interfaces.
- **REST API** (OpenAPI at `/docs`):
  - `/chat`, `/chat/{id}/updates`, `/ask`, `/answer`, `/ask/image`, `/clips`, `/search`, `/videos`, `/traces`,
    `/health`, `/metrics`.
  - Health reports each dependency (database, Redis, S3, OpenSearch, embedder, Ollama, both workers).
  - The API starts in a **degraded mode** instead of crashing when a dependency is down.

---

## 13. Observability: tokens, traces, metrics

### 13.1 Choice (ADR 0004)

| Option | Verdict |
|---|---|
| Langfuse, self-hosted | rejected: needs ClickHouse plus several more containers on an already-full laptop |
| Langfuse Cloud | rejected: requests would leave the machine |
| OpenTelemetry + Jaeger/Tempo | not needed yet (one API host); revisit when it grows |
| **Own tables in Postgres + Prometheus + Grafana** | **chosen:** two light containers; SQL we control; workers report too |

### 13.2 Token ledger

- **One row per model call.**
  - Tokens come from Ollama's `prompt_eval_count` and `eval_count`; times from its load, prompt and output durations.
  - **Cost** is the tokens × a configured hosted-model list price, **stored with the row** like a bill, so changing the
    price doesn't rewrite history.
  - **Outcome:** accepted, adjusted (a guard changed it), rejected (rules answered: wasted tokens), invalid,
    not_found, or cache_hit (zero spent, with tokens *saved*).
  - **Origin:** api, worker or eval, so evaluation runs don't pollute real numbers.
- **Findings:**
  - an `/ask` costs **~448 tokens, 95% of them the fixed instructions**;
  - a keyframe caption costs ~225 tokens (≈196 in, with the image);
  - **Ollama counts the full prompt every call but reuses its work** (reading the prompt: 4.9 s cold → 0.07–0.15 s);
  - writing (~15 tokens/s) is where the time goes;
  - **6%** of understanding tokens were wasted on rejected answers.

### 13.3 Traces

- **A small tracer:** `with span("search"):` nests via contextvars. Each HTTP request is a trace (an ASGI middleware);
  its id is the `X-Request-ID` header and `request_id` in responses.
- **Across Celery:** `before_task_publish` puts the trace id and parent span id into the message headers;
  `task_prerun` continues the trace in the worker. A clip cut appears **inside the request that caused it**.
- **Saved after the response is sent**, one insert per trace, in a thread. A failed save never fails a request. Empty
  polls discard their trace.
- **Logs carry `[trace id]`.**
- **Retention:** 14 days (an Airflow DAG calls `/admin/cleanup-traces`).
- **What traces showed:**
  - the model is **~75% of `/ask` time** (2.95 s of 3.9 s);
  - captioning is **~92% of processing** for a short silent video;
  - chat turns alternate two prompts, which **evict each other from Ollama's single prompt cache** (4–5 s to re-read
    each). Four slots were tried and didn't help in this Ollama version, so that was reverted.

### 13.4 Metrics and dashboards

- **Prometheus metrics** (multiprocess mode across 4 uvicorn workers): request rate and latency by **route template**
  (not per video id), `/ask` results, what understood them, cache hits, chat turns by action and decider.
- **Grafana dashboards generated from Python** (`scripts/build_dashboards.py`):
  - **Tokens & cost:** tokens, cost, saved, cache hit rate, wasted share, per call, per video;
  - **Requests & latency:** rates, p50/p95 latency by route and by `/ask` stage, time per step from traces, slowest
    traces (linked), errors.
- **Every panel query was run through Grafana's API** to check it returns data. That found that stat panels needed
  instant queries, and ratio panels needed `or vector(0)` when nothing had happened yet.
- **The dashboards found a real bug:** the "Recent errors" panel showed failed jobs for non-existent videos every few
  minutes. A test had been sending **real jobs to the running broker** since the upload endpoint was written. Tests
  can no longer reach the broker.

---

## 14. Caching

| Cache | Key | TTL | Invalidated by |
|---|---|---|---|
| **understanding** (request → intent) | normalised request (lowercase, spaces collapsed) + model + **fingerprint of the understanding code** (hash of its source files: prompt and guards) | 7 days | any edit to that code |
| **answer** (the whole `/ask` result) | request + filters + **index version** + fingerprint of the search, answer and clip code and settings | 24 hours | **any index write** (new video, re-index) bumps the version; code or setting changes |
| **clip files** | S3 key from video + rounded times | — | derived data; clips are checked to still exist on each hit |

**Why the index version:** yesterday's cached "no match" must not hide a video indexed today. **Why fingerprints:**
nobody has to remember to bump a version number when a prompt changes. **Don't cache** results produced because the
model *failed*: next time it may work.

**How it's tested:**
- **Unit** (a fake in-memory Redis):
  - hits skip the model;
  - case and spacing are normalised;
  - filters are part of the key;
  - an index bump misses;
  - a deleted clip file misses;
  - model failures aren't cached;
  - a Redis outage means no cache, not an error;
  - saved tokens are recorded.
- **Live** (`make eval-cache`, 48 requests in 4 passes):

| Pass | p50 | Hits | Tokens spent | Tokens saved |
|---|---|---|---|---|
| nothing cached | 2,158 ms | – | 21,377 | 0 |
| repeated, in CAPITALS | **4 ms** | 48/48 answers | 0 | 21,377 |
| `max_clips=2` (new answer, same meaning) | 55 ms | 48/48 understandings | 0 | 21,377 |
| after re-indexing one video | 55 ms | 0 answers, 10/10 understandings | 0 | 4,444 |

- **Correctness:** cached answers were **identical** to fresh ones (48/48). After the re-index, **0 stale**.
- **Caveat:** best-case numbers (every request repeated); real hit rates depend on traffic.
- **Trade-off:** fingerprinting invalidates on *any* edit, even a comment. That's deliberate: a stale cache is worse
  than a cold one.

---

## 15. Model backend and the AWS path

- **`LLM_PROVIDER=ollama | bedrock`**, behind one interface with two operations: `json_reply(system, user, schema)`
  and `describe_image(prompt, image)`, each returning text plus an `LLMCall` for the ledger.
  - **Ollama:** `format` = the JSON schema.
  - **Bedrock:** the Converse API with **one tool whose input schema is the form, and `toolChoice` forcing it**,
    Bedrock's equivalent of constrained JSON.
- **Errors are normalised:**
  - `ModelUnavailable` (down, timeout, throttling, 5xx) → the worker retries, and chat falls back to rules;
  - `ModelRejected` (bad input, no access) → no retry.
- **Tested:** the Ollama implementation against mocked HTTP; Bedrock against a **stubbed boto3 client** (the forced
  tool call, token counts, image bytes, error mapping). **Not yet run against real Bedrock.**
- **Why not Ollama on the Mac?** It would be fast (Apple GPU), but AWS has no Apple GPU. The deployable pattern is the
  model as a replaceable service.

| Local | AWS |
|---|---|
| API, workers, bot | ECS Fargate (or one EC2 instance running Compose for a demo) |
| Postgres / Redis | RDS / ElastiCache |
| SeaweedFS | S3 (already the S3 API) |
| OpenSearch | Amazon OpenSearch Service |
| Ollama | **Bedrock** (pay per token) or a GPU instance (~$600–750/month running around the clock) |
| Airflow | EventBridge Scheduler (only two daily jobs) |
| Prometheus / Grafana | Amazon Managed Prometheus / Grafana |

---

## 16. Testing and evaluation strategy

### 16.1 Three layers

| Layer | What | How |
|---|---|---|
| **Unit and API tests** | **280 tests**, in about 3 s, **no infrastructure** | SQLite in-memory database with the same ORM; `httpx.MockTransport` for every HTTP client (Ollama, embedder, Pexels, Telegram); a fake Redis (with an outage switch); `MagicMock` services; a stubbed boto3 client; FastAPI `TestClient` with dependency overrides; tests are blocked from reaching the real Celery broker (autouse fixture) |
| **Evaluations** | labelled sets per component (table below), run against the live stack with `make eval-*` | metrics per component; results saved as JSON in `eval/results/` as evidence for ADRs |
| **Live checks** | real end-to-end runs | real Pexels downloads, uploads, Chrome (Playwright), Telegram token check (`getMe`), traces and dashboards |

**Evaluation sets:**

| Component | Set | Size | Key metrics |
|---|---|---|---|
| Search | `queries.json` + `no_answer.json` | 36 + 12 | Recall@5, MRR, per category, false answers |
| Request understanding | `intents.json` | 58 | type / text / exclusions / all correct, per category; p50; tokens |
| Question answering | `questions.json` + `talks/sleep_talk.txt` | 24 | facts right, citation right, **made up**, refusals, p50, tokens |
| Search by photo | `images.json` | 45 | hit@1 per kind, false answers, similarity distributions, cut-off sweep |
| Chat routing | `agent_turns*.json` | 40 + 20 + 20 | action accuracy, all correct, **unsafe downloads**, share calling the model |
| Conversations | `conversations.json` | 12 / 28 turns | turns right, conversations right, p50/p95, tokens per turn |
| Caching | the search requests, 4 passes | 48 | p50/p95, hit rates, tokens saved, **identical to fresh** |
| Caption embeddings | `queries.json` | 36 | Recall@5, MRR (overall, hard, per category) |

### 16.2 Discipline

- **Labels are written before any system runs on them,** and are never edited after seeing results. A correct-looking
  answer that misses its label stays a miss, with a note (§8.2).
- **Held-out sets** for anything with rules or prompts that could be tuned. Each set is used for **at most one round
  of changes**; after that it's reported as "seen".
- **Report both:** "first run (unseen)" and "after fixes (seen)", as in the conversations (23/28 → 28/28).
- **Sweeps, not guesses,** for every threshold: vector-only 0.20, image 0.57, blank frames 4.0 (the blank frames
  measured 0.0 against 13.5 for the least varied real frame).
- **Check labels themselves:** the photo contact sheet; search relevance from the actual keyframes.
- **Regression checks after data changes:** adding the talk video, clearing blank frames and fetching balloons were
  each followed by a search re-run and a diff.

### 16.3 Weaknesses of the evaluation (be upfront about these)

- **Small:** 36 search queries, and categories with 1–3 queries are suggestive only.
- **Written by the builder,** not by real users. Real conversations would be the next test.
- **Synthetic speech** for question answering is cleaner than real audio.
- **The library grows,** so fixed labels drift: three new balloon videos moved one query from rank 2 to 3 (MRR 0.954 →
  0.949) with a reasonable result that the labels count as worse.
- **What the evaluations missed was user experience:** clip *length*, discoverability of uploading, and a silent
  wait that feels broken were all found by using the app, not by metrics.

---

## 17. Performance and cost

| What | Measured |
|---|---|
| `/ask`, nothing cached | ~2.2 s p50 (model ~75%) |
| `/ask`, answer cached | **4 ms** |
| chat turn, rules decide / caches hit | 1–3 s |
| chat turn, model runs twice (routing + understanding or answering) | **8–15 s** on CPU |
| question answering | ~1.9–2.3 s |
| photo search | ~0.2 s |
| clip cut (fresh / cached) | 0.1–0.9 s / 1–8 ms |
| processing a 17 s talk | ~20–40 s |
| processing a 5 s silent stock clip | ~8 s (captioning 92%) |
| model speed (CPU) | ~15 tokens/s writing; 4–5 s to read a ~400-token prompt cold, ~0.1 s when reused |
| tokens | ~450 per `/ask`, ~225 per keyframe caption, ~1,700–2,400 per minute of footage, ~400–560 per chat turn |
| estimated cost (Claude Haiku 4.5 list price as the reference) | ~$0.54 per 1,000 `/ask`, ~$0.34 per 1,000 captions |
| memory | Docker 16 GB; the model ~5 GB; OpenSearch ~1.3 GB; worker (Whisper) ~1.1 GB; embedder ~1 GB |

**Bottlenecks and fixes:**
- **The LLM on CPU:** a GPU or Bedrock. Also shorten the fixed prompt (95% of tokens).
- **Captioning:** a GPU, or skip blank or near-duplicate frames (done for blank ones).
- **One processing worker** (Whisper memory): more worker replicas or machines.
- **Two prompts alternating** evict Ollama's cache: separate model servers per role, or a server with multi-slot
  prefix caching.

---

## 18. Bugs and lessons

These are good to tell as stories. Each was found by measuring or by using the app.

| Bug | How it was found | Fix | Lesson |
|---|---|---|---|
| **Blank frames got invented captions**: a plain grey frame became "a person in a dark room holding a smartphone", so "a horse" matched a talk video | live chat returned an absurd clip | frames with grey-level std ≤ 4.0 get no caption or vector (blank 0.0 vs real ≥ 13.5); search unchanged | VLMs hallucinate on empty input; don't send it |
| **The LLM copied its prompt's example** ("exclude people" → "a horse running"; "yes" → download "horse") | evaluation | placeholder examples, grounding guards, a consent guard | small models imitate examples; check output against the user's words |
| **A test was sending real jobs to the running worker** | the Grafana "Recent errors" panel | tests can't reach the broker (autouse fixture) | observability finds bugs tests can't |
| **"half-life" became "half -life"** (Whisper splits tokens) | a citation-check miss; later again in clip captions | one shared word-joining function | the same bug twice means the logic should live in one place |
| **A routing reply cut off mid-JSON** | the conversation evaluation | 60 → 150 output tokens | limits that are fine on average fail on outliers |
| **A new request narrowed to the last clip** ("a beach" searched only the dog video) | live use | scope only when the message points at the clip ("that clip") | the model sets flags freely; verify them against the text |
| **Follow-ups dropped their context** ("now without people" searched for "now") | live use | the rules' follow-up detection overrides the model's request | |
| **My test loop never sent the conversation id** (zsh doesn't split `${CID:+-F …}`) | puzzling "no memory" results | live tests through Python | suspect the harness too |
| **Ollama couldn't reload the model** (5.0 GiB needed, 4.6 free) after monitoring was added | 500 errors after idle | Docker 12 → 16 GB | capacity is part of the design |
| **Stat panels empty, ratio panels blank** | running every panel query through Grafana's API | instant queries; `or vector(0)` | test dashboards like code |
| **The library growing shifts metrics** (MRR 0.954 → 0.949) | re-running the evaluation after fetching | documented; one query moved one rank for a defensible reason | fixed labels drift as data grows |
| **Clips too long; upload not discoverable; silent waits** | user feedback | 5 s clips / sentence clips; upload first in the UI and greeting; step-by-step progress | metrics measure correctness, not experience |

---

## 19. Limitations, security gaps and next steps

**Must fix before public use:**
1. **Uploads are shared.** An uploaded video joins the shared library, so another user could get a clip of someone's
   private video. Needs per-user ownership, and filtering by owner in every query.
2. **No authentication, rate limits or quotas:** uploads, Pexels downloads and model cost are unbounded.
3. **No deletion or retention** for videos and derived clips.
4. **Local-dev shortcuts:** Grafana without a login, reading Postgres with the app's credentials (needs a read-only
   role); the Telegram bot open to anyone who finds it (an allow-list exists).

**Quality and scale:**
- latency on CPU (8–15 s turns);
- routing ~0.85 on free phrasing;
- the number-only hallucination guard;
- keyword-only retrieval for library questions;
- whole-file downloads for each cut;
- a single processing worker;
- English only.

**Evaluation:** small, self-written, synthetic speech; next is real conversations, and a Bedrock comparison on the
same sets.

**Next steps, in order:** private uploads + deletion → auth + limits → Bedrock evaluation → deploy (one EC2 instance
with Compose + Bedrock for a demo; Terraform for the managed-services version) → CI (tests, lint, image builds) →
automated UX checks (clip-length limits, progress interval).

---

## 20. Likely interview questions

### Architecture

**Why Postgres *and* OpenSearch? Why not a single vector database?**
Postgres is the source of truth (transactions, migrations, relations between videos, segments, token ledger and
traces). OpenSearch is a *derived* index I can rebuild or migrate at any time. It gives BM25 with analysers (exact and
stemmed transcripts) **and** HNSW kNN with filtering in one engine, which hybrid search needs. A pure vector database
would need a separate keyword engine, and the evaluation shows keyword search is essential for details and exact
phrases.

**Why Celery and two queues?**
Processing is slow (minutes) and memory-heavy (Whisper), so it runs one job at a time. Clip cutting is interactive
(sub-second). With a shared queue, a user would wait behind a processing job. Measured: a clip delivered in 0.55 s
while the processing worker was busy. `acks_late` plus an idempotent pipeline means a crashed worker's job is redone
safely.

**Why is CLIP a separate service?**
One model copy serves both the API (query text) and the worker (frames), it scales independently, and the API stays
light. The cost is a network hop of ~60–90 ms per query embedding.

**How do you change the index mapping without downtime?**
Clients use an alias. A mapping change bumps `INDEX_VERSION`; reindex builds `_v{n+1}` from Postgres, refreshes it,
then switches the alias atomically. The old index is kept for rollback. Indexing code also writes safely into an
older mapping by dropping unknown fields.

### Retrieval

**Why hybrid search? What does each retriever add?**
Measured per category: CLIP gets style and synonyms right (1.00; keyword 0.33 and 0.00), because the answer is in the
pixels. Keyword search over captions gets named details right (1.00; CLIP 0.83). Fused with RRF: Recall@5 0.97.

**Why RRF rather than weighting the scores?**
BM25 scores and cosine similarities aren't on the same scale, and their distributions change per query. RRF uses
ranks only and needs no tuning. The downside is that it ignores confidence, hence a separate cut-off for results only
the vector side found.

**How do you stop vector search from "answering everything"?**
kNN always returns neighbours: vector-only had 9 false answers out of 12. Three layers:
1. a similarity floor (0.15);
2. a stricter floor for results with no keyword support (0.20, chosen by a sweep: 5 → 2 false answers, losing one
   relevant video on one query);
3. request understanding marks greetings and requests with no subject.

Two of 12 still get through; I know which, and why.

**How do you get word-accurate times?**
Whisper word timestamps are stored per window. Search finds the window; a phrase locator aligns the query to the
window's words with sequence matching (tolerating misheard words at ≥ 0.75 token similarity) and returns the first
and last words' times. Verified by transcribing the cut clip back: 9.49 s against 9.48 s.

**Why not embed the captions too?**
I tested it (ADR 0001). It changed 2 of 36 queries, one better and one worse, and hard-query MRR fell. Caption
embeddings overlap both retrievers and inherit caption errors (a cat seen from behind, captioned as a dog).

### LLM and agent

**Why a 3B local model?**
Privacy (requests stay on the machine), zero marginal cost, and one model for captions and text, because Ollama holds
one model in memory and swapping costs seconds. Its weakness (unreliable structured decisions) is handled by design:
JSON-schema decoding, guards, rules first, template replies.

**How do you stop hallucination?**
Four mechanisms:
1. The model never writes to the user: replies are templates filled from tool results.
2. Question answering sees only numbered excerpts, must cite them, and any number it states must be in the cited
   text; otherwise the answer is "not found". Measured: 0 of 7 unanswerable questions answered.
3. Request understanding must use the user's own words (grounding).
4. Blank frames aren't captioned.

**What are the "guards"? Isn't this just rules again?**
They're checks on the model's output, not an alternative to it. For example: "is this download preceded by consent?",
"does the request text use words the user said?", "does a question stay the user's question?". The model does the
open-ended language work; guards catch its known failure modes. Each guard exists because of a measured or live
failure, and each was checked against all routing sets for regressions.

**Why rules first?**
Measured: model only 0.70–0.85 and one unsafe download; rules only 1.00 on known phrasings but 0.60 on new ones;
rules first + model + guards 0.85–0.95 with 0 unsafe downloads. It's also faster: 30–50% of turns skip the model.

**Why LangGraph?**
The flow has branches and several entry points (message, video upload, progress poll) with shared state. Explicit
nodes and edges keep each path testable. I'd accept a plain loop for a single-step agent.

**What did you learn about small models?**
They copy prompt examples, write answers where they should write a request, and drop context. Output limits must fit
the worst case (a 60-token cap cut a reply mid-JSON). Structured output removes parse errors but not wrong content.

### Evaluation

**How do you know it works?**
Each component has a labelled set and metrics (§16). The table I'd show: search Recall@5 0.97 / MRR 0.95 with 2 of
12 false answers; question answering 0/7 made up, 16/17 facts right, 17/17 citations right; photos 18/20 found and
0/15 false; routing 0.85–0.95 with 0 unsafe; conversations 23/28 on an unseen first run.

**How do you avoid overfitting your evaluations?**
Labels are fixed before runs and never edited afterwards. Held-out sets are separate from anything that informed a
fix. Each set is used for at most one round of changes. I report first-run and after-fix numbers side by side.
Thresholds come from sweeps. I checked the labels themselves (the contact sheet, keyframes).

**What are the weaknesses of your evaluation?**
It's small and self-written, the speech is synthetic, and labels drift as the library grows. It measured correctness
but missed user experience (clip length, discoverability, silent waits): those came from using the app.

**Why Recall@5 and MRR?**
Recall@5 asks whether the right video is among what a user would look at. MRR asks how high the first right one is,
which matters because `/ask` returns one clip. Per-category breakdowns show *which* weakness a change affects. False
answers measure the other failure: returning something when nothing matches.

### Operations

**How do you trace a request across the API and workers?**
The trace id is the request id. A Celery signal copies it (and the parent span id) into each message's headers, and
the worker continues the same trace. Traces are saved after the response (one insert), and logs carry the id.

**How do you know the cost?**
Every model call is a ledger row with tokens and a cost priced at write time, plus an outcome, so wasted tokens
(rejected answers, 6%) and cache savings are visible. About 450 tokens per `/ask`, 95% of them fixed instructions:
the biggest lever is a shorter prompt.

**How do you invalidate caches correctly?**
The answer key includes an index version bumped on every index write (new data) and a hash of the code and settings
behind the answer (new logic). Verified: 0 stale answers after a re-index, and cached answers identical to fresh ones
48/48.

**What happens when the model is down?**
Request understanding falls back to rules, routing falls back to rules, question answering reports "unavailable",
and captioning retries (`ModelUnavailable`). The API starts degraded instead of crashing, and health shows which
dependency is down.

### Scaling and production

**How would you deploy and scale this?**
For a demo: one EC2 instance running the same Compose file, with Bedrock for the model. For production: ECS Fargate
for the API and workers, RDS, ElastiCache, OpenSearch Service, S3, and Bedrock (or GPU instances). Scaling levers:
- more processing workers (the bottleneck is Whisper + captioning);
- a GPU or Bedrock for the model (latency);
- replicas for OpenSearch;
- clip workers scale independently;
- cut clips from byte ranges instead of whole files.

**What would you fix before letting the public use it?**
Private uploads (today they join the shared library: a real privacy issue), authentication, rate limits and quotas,
deletion and retention, a spending cap, and a read-only role for dashboards.

**What would you do differently?**
- Build user-experience checks into the evaluation from the start (clip length, progress interval).
- Design per-user data ownership from the first upload.
- Run the model evaluation on a hosted model early, to separate "the model's limits" from "CPU latency".
- Use a real speech corpus for question answering.

### Behavioural-style prompts about this project

- **A decision you reversed:** clip lengths. Whole shots were "correct" per the evaluation, but felt like short videos
  in use. I changed to 5 s clips and sentence clips, and documented it as an update to the ADR.
- **A time data changed your mind:** I expected the LLM alone to route well enough. It scored 0.57 (then 0.70) and
  made an unsafe download, so I switched to rules first.
- **Handling ambiguity:** "and a hot air balloon" (a new subject) vs "and one with a laptop" (a refinement). I made
  the rule general ("and a/an/some X" names a new subject; "and one with X" refines it) instead of patching one
  phrase.
- **Honesty about results:** I kept a miss that was arguably a labelling issue, rather than editing the label after
  seeing the answer.

---

## 21. Numbers cheat sheet

| Area | Number |
|---|---|
| Services / tests / ADRs | 14 containers · 280 tests (~3 s) · 5 ADRs |
| Search (app) | Recall@5 **0.97**, MRR **0.95**, false answers **2/12** (raw hybrid 7/12; vector alone 9/12) |
| Keyword vs CLIP | style 0.33 vs 1.00 · synonym 0.00 vs 1.00 · detail 1.00 vs 0.83 |
| Vector-only cut-off | **0.20** (false answers 5 → 2; one relevant video lost) |
| Understanding | 0.81 all correct (rules 0.78); free phrasing **0.58 vs 0.17** |
| Question answering | **0/7 made up** · facts 16/17 · citations 17/17 · ~2 s |
| Photo search | cut-off **0.57** · near-dup 10/10 · semantic 18/20 · negatives **0/15** false |
| Routing | model alone 0.70–0.85 (1 unsafe) · rules first **0.85–0.95** (0.80 unseen) · **0 unsafe** · 30–50% skip the model |
| Conversations | **23/28** turns, 8/12 conversations (first run) → 28/28 after fixes |
| Clips | quote ± 0.75 s (~3 s) · visual **5 s** · sentence ~5 s (≤ 8 s) · accuracy 9.49 vs 9.48 s · cut 0.1–0.9 s · cache hit 1–8 ms |
| Caching | 2.2 s → **4 ms** · 48/48 identical · 0 stale after re-index |
| Tokens / cost | ~450 per `/ask` (95% fixed prompt) · ~225 per caption · ~$0.54 per 1,000 `/ask` (estimate) · 6% wasted |
| Time | model ~75% of `/ask` · captioning ~92% of processing · chat turns 1–3 s (rules/cache) or 8–15 s (CPU model) |
| Pexels fetch | balloon clip in **78 s** · uploads: request queued, clip at 0:42 for a 17 s talk |
| Blank frames | grey std ≤ **4.0** (blank 0.0, least varied real 13.5) |
| Speech windows | 15 s every 10 s (5 s overlap) |
| Shots | scene threshold 0.3 · min 1 s · max 10 s |
| HNSW | m 16, ef_construction 128, cosine, Lucene (filters inside kNN) |
| RRF | k = 60 |
| Memory | Docker 16 GB; model ~5 GB |

---

## 22. Glossary of metrics

| Metric | Definition | Why it's used here |
|---|---|---|
| **Recall@5** | relevant videos in the top 5 / min(5, number relevant) | is the right video among what a user would look at |
| **MRR** (mean reciprocal rank) | mean of 1 / rank of the first relevant video | `/ask` returns one clip, so rank 1 matters most |
| **False answers** | requests that should return nothing but return a clip | the "nearest neighbour always exists" failure |
| **All correct** (intents, routing) | every labelled field right at once | partial credit hides unusable outputs |
| **hit@1** (photos) | the top video is relevant and above the cut-off | one clip is returned |
| **Made-up rate** | answered an unanswerable question | the most harmful QA failure |
| **Citation accuracy** | a cited excerpt contains the labelled evidence | the answer points at the right moment |
| **Unsafe downloads** | a Pexels fetch without consent | a hard safety requirement: must be 0 |
| **p50 / p95** | median / 95th-percentile latency | typical vs tail experience |
| **Tokens per call; cost per 1,000** | from Ollama's counts × a list price | comparable cost even with a free local model |
| **Wasted share** | tokens on answers the guards rejected / all understanding tokens | the cost of the safety net (6%) |
| **Cache hit rate; tokens saved** | hits / (hits + calls); tokens of avoided calls | what caching is worth |
