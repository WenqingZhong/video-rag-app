# Video RAG: find the exact moment in a video

Ask for a moment in plain language and get back **just that clip**, a few seconds long, cut at the right words or frames.
It works on your own uploads and on a library of stock footage, and answers questions about what the videos say,
citing where it heard it.

| You ask | You get |
|---|---|
| *(upload a talk)* "the part where the host says AI is changing everything" | a 3-second MP4 cut at exactly those words |
| "a dog in the snow" | a 5-second clip of a dog in the snow, with its Pexels credit |
| "a beach with no people" | the **empty** beach (negation is understood, not ignored) |
| "how long is a sleep cycle?" | "Each cycle lasts about 90 minutes." + the clip where it's said |
| "does alcohol affect sleep?" | "I couldn't find that in the videos." (it answers only from the library) |
| *(a photo of a kitten)* | the clip that looks most like it |
| "a hot air balloon" → "yes please" | nothing in the library → offers to fetch from Pexels → downloads, processes, sends the clip |

You can use it as a **web chat**, a **Telegram bot**, or through the **REST API**. Everything runs locally in Docker,
including the language model, and the model backend can be switched to Amazon Bedrock for AWS.

---

## How it works

```text
                      ┌────────────── ingestion (Celery worker) ───────────────┐
upload / Pexels ─────▶│ probe → scene detection → keyframes → CLIP vectors      │──▶ Postgres (source of truth)
                      │        → captions (VLM) → Whisper words → speech windows│──▶ OpenSearch (BM25 + kNN)
                      └──────────────────────────────────────────────────────────┘──▶ S3 (videos, frames, clips)

chat turn ─▶ decide: rules for clear patterns, else a local LLM (JSON form) checked by guards
          ─▶ one tool:  find_clip ─ understand (quote | topic | visual + exclusions)
                                  ─ search (phrase with word times · keywords · hybrid BM25+CLIP with RRF)
                                  ─ cut (clip worker, ffmpeg, cached in S3)
                        answer_question ─ excerpts → LLM answers only from them, must cite → guards
                        find_by_image ─ CLIP photo vector vs keyframe vectors
                        fetch_from_pexels ─ only after the user says yes; polls until processed
          ─▶ reply from a fixed template (the model never writes to the user)
```

**Every video is indexed two ways:**
- **What is shown:** keyframes at scene changes, each with a CLIP vector and a one-sentence caption.
- **What is said:** a Whisper transcript with a timestamp on every word.

A request is first **understood**:
- a *quote* ("says X") searches speech and returns the exact words' times;
- a *topic* ("talks about X") searches speech and returns the sentence about it;
- a *visual* request ("a dog") runs hybrid search over keyframes.

Exclusions ("no people") filter the results. The chosen moment is **cut into a real MP4**, re-encoded so it starts on
the first word rather than the nearest video keyframe.

## Results

Every component is judged against a labelled test set, with labels fixed before the first run. Held-out sets are
reported separately from sets that informed a fix.

| Component | Test set | Result |
|---|---|---|
| **Search** (as used by the app) | 36 labelled queries + 12 that should return nothing | Recall@5 **0.97**, MRR **0.95**, false answers **2 of 12** (hybrid search on raw text: 7 of 12) |
| **Request understanding** (LLM + guards + rules) | 58 requests | **0.81** all fields correct; free phrasing **0.58** vs rules alone 0.17 |
| **Question answering** | 24 questions, 7 unanswerable | facts right **16/17**, citation right **17/17**, **0/7** made-up answers |
| **Search by photo** | 45 images | library frames **10/10**; photos of library subjects **18/20**; unrelated photos **0/15** false matches |
| **Chat routing** (rules first + LLM + guards) | 40 turns + two held-out sets of 20 | **0.95 / 0.85 / 0.95** (0.80 on unseen phrasing); the LLM alone 0.70–0.85; **0** downloads without consent |
| **Whole conversations** | 12 conversations, 28 turns | **23/28** turns right on the first, unseen run |
| **Clip accuracy** | a cut quote transcribed back | the word lands at 9.49 s where search placed it at 9.48 s |
| **Caching** | 48 repeated requests | 4 ms instead of 2.2 s median; answers identical to fresh ones (48/48) |

**Cost and speed:**
- About **450 tokens per request**, roughly **$0.54 per 1,000 requests** at a small hosted model's list price. The
  model runs locally, so this is an estimate that's tracked per call.
- Turns take **1–3 s** when the rules decide or caches hit, and **8–15 s** when the local 3B model runs on the CPU.
- Uploads are processed in under a minute for a short clip, and a few minutes for longer ones, with progress
  reported throughout.

## Engineering decisions

Each decision is written up with the experiment behind it in [`docs/decisions/`](docs/decisions/):

- **An LLM that proposes, and plain code that checks.** A 3B local model is fast and private, but on its own it invents
  details, copies examples from its prompt and once downloaded videos without being asked. Guards compare its output
  with the user's own words (grounding, consent, scope) and fall back to rules. Clear patterns skip the model entirely.
  [ADR 0002](docs/decisions/0002-query-understanding.md), [ADR 0005](docs/decisions/0005-chat-agent.md)
- **Answers only from the videos.** Question answering sees numbered, timed excerpts. It must cite one, and every
  number in its answer must appear in the cited text; otherwise the reply is "not found". It never uses general
  knowledge.
- **Clips are moments.** A quote is cut at its words ± 0.75 s; a visual clip is 5 s around the matched frame; an answer
  is the sentence that answers, found using Whisper's word timings. Cutting re-encodes the video (a plain stream copy
  can only start at a video keyframe: asked for 8.73 s, it started at 7.96 s) and runs on a **dedicated queue**, so a
  user never waits behind video processing.
  [ADR 0003](docs/decisions/0003-clip-delivery.md)
- **Hybrid retrieval, measured.** BM25 and CLIP complement each other (keyword search misses synonyms, CLIP misses
  details), fused with Reciprocal Rank Fusion. Adding caption embeddings as a third retriever was tested and rejected.
  [ADR 0001](docs/decisions/0001-no-caption-embeddings.md)
- **Caches that can't serve stale answers.** Answer keys include an index version that every index write bumps, plus a
  fingerprint of the code and settings they depend on, so a new video or a prompt change invalidates exactly what it
  should. [ADR 0004](docs/decisions/0004-observability-and-caching.md)
- **Observability built in:**
  - every request and background task is a trace, with the trace ID carried through Celery message headers;
  - every model call is recorded with its tokens and cost;
  - Grafana dashboards are generated from code.

  The dashboards found real bugs, including a test that had been sending jobs to the running worker.
- **Production-shaped from the start:**
  - Postgres is the source of truth, and the search index can be rebuilt from it (blue/green, with an atomic alias
    switch);
  - the processing pipeline is idempotent;
  - schema changes go through Alembic migrations;
  - health checks report each dependency;
  - uploads and downloads are asynchronous, with progress reporting;
  - where the model runs is a setting (`LLM_PROVIDER=ollama | bedrock`).

## Running it

**Requirements:**
- Docker with **16 GB of memory** (the vision-language model needs about 5 GB);
- [uv](https://docs.astral.sh/uv/);
- a free [Pexels API key](https://www.pexels.com/api/);
- optionally, a Telegram bot token from @BotFather.

```bash
cp .env.example .env          # set PEXELS_API_KEY (and TELEGRAM_BOT_TOKEN for the bot)
uv sync
make start                    # docker compose up --build -d (14 services)
docker compose exec ollama ollama pull qwen2.5vl:3b
make health                   # every dependency's status
```

| Open | For |
|---|---|
| http://localhost:8000/app | the web chat: upload a video, ask, send a photo |
| http://localhost:8000/docs | the REST API (OpenAPI) |
| http://localhost:3000 | Grafana: tokens and cost, latency, cache hits, per-step timings, errors |
| http://localhost:8080 | Airflow: scheduled Pexels ingestion and trace retention |
| Telegram | message your bot: send a video, then ask for a moment |

To fill the library with stock footage: `POST /api/v1/videos/pexels {"query": "dogs", "count": 5}`, or enable the
`pexels_ingestion` DAG.

## API

| Endpoint | Does |
|---|---|
| `POST /api/v1/chat` | one conversation turn: `message`, optional `conversation_id`, `image` or `video` → reply, clips, citations |
| `GET /api/v1/chat/{id}/updates` | progress of an upload or download ("Step 6 of 9: describing the frames · 0:42"), then the result |
| `POST /api/v1/ask` | a single request → the best clip(s), what was understood, timings, tokens used |
| `POST /api/v1/answer` | a question → a cited answer, or "not found" |
| `POST /api/v1/ask/image` | a photo → the most similar clip |
| `POST /api/v1/clips` | cut any time range of a video |
| `POST /api/v1/search` | ranked segments with match times (keyword, vector, hybrid, or understood) |
| `POST /api/v1/videos` · `/videos/pexels` | upload a file, or queue Pexels videos (processed in the background) |
| `GET /api/v1/videos/{id}` | status, processing stage, metadata |
| `GET /api/v1/traces/{id}` | where a request spent its time, across the API and workers |
| `GET /api/v1/health` · `/metrics` | readiness per dependency · Prometheus metrics |

## Evaluation and development

```bash
make test           # 280 unit and API tests (no infrastructure needed)
make lint
make eval           # search quality: Recall@5 and MRR per mode and per failure category
make eval-intents   # request understanding: LLM + guards vs rules
make eval-qa        # question answering: facts, citations, made-up answers
make eval-images    # search by photo: hit rate and false matches per similarity cut-off
make eval-agent     # chat routing on the labelled and held-out turns
make eval-chat      # whole conversations through /chat
make eval-cache     # what caching saves, and that cached answers match fresh ones
make usage          # tokens and estimated cost per operation, per video
make trace ID=…     # one request as a timeline
```

Labelled sets live in [`eval/`](eval/), and saved results in [`eval/results/`](eval/results/).

## Tech stack

| Area | Tools |
|---|---|
| API and workers | FastAPI, Celery (processing queue + a dedicated clip queue), Redis |
| Storage | PostgreSQL + Alembic, S3 API (SeaweedFS locally, AWS S3 in production) |
| Search | OpenSearch: BM25 with English stemming, HNSW kNN (cosine), versioned index behind an alias |
| Models | faster-whisper (speech, word timestamps), CLIP ViT-B/32 (image and text vectors, own service), `qwen2.5vl:3b` via Ollama (captions, request understanding, answers, routing) |
| Agent | LangGraph, conversation memory in Redis |
| Video | ffmpeg (scene detection, keyframes, word-accurate re-encoded clips) |
| Operations | Prometheus, Grafana (dashboards as code), Airflow (schedules), request tracing in Postgres |
| Interfaces | web chat (a single static page), Telegram bot, REST API |

## Deploying to AWS

The services map onto managed ones:

| Local | AWS |
|---|---|
| API and workers | ECS (Fargate) |
| Postgres | RDS |
| Redis | ElastiCache |
| SeaweedFS | S3 (already using the S3 API) |
| OpenSearch | Amazon OpenSearch Service |
| Airflow | EventBridge Scheduler |
| Prometheus and Grafana | Amazon Managed Prometheus / Grafana |

**Models:** set `LLM_PROVIDER=bedrock` to use Amazon Bedrock (the Converse API, with structured output through a forced
tool call), or point Ollama at a GPU instance. The same evaluations can compare models before switching.

**Before public use, it still needs:**
- user accounts;
- rate and upload limits per user;
- a way to delete videos.

## Limitations

- **Latency:** the local model runs on the CPU; 8–15 s per turn when it's used. A GPU or Bedrock removes most of it.
- **Accuracy:** chat routing is about 0.85 accurate on free phrasing; each reply shows how the turn was decided.
- **Test data:** the evaluation sets were written by the author, and the spoken test material is synthetic speech.
  Real user conversations would be the next test.
- **Bedrock:** the client is unit-tested but hasn't been run against real Bedrock.

## Project layout

```text
src/
  routers/            REST endpoints (chat, ask, answer, clips, search, videos, traces, admin, health)
  services/agent/     chat agent: decide (rules + LLM + guards), LangGraph graph, tools, memory, reply templates
  services/understanding/  request → intent (LLM JSON schema + guards + rule fallback)
  services/search/    phrase / keyword / vector / hybrid search, RRF, word-level phrase location
  services/qa/        question answering from transcript and caption excerpts, with citations
  services/clips/     clip boundaries (words, sentences, keyframes) and get-or-cut
  services/processing/  the ingestion pipeline: ffmpeg, scenes, keyframes, Whisper, captions, vectors
  services/llm/       model client: Ollama or Bedrock
  services/tracing/, usage/, metrics/  traces, token ledger, Prometheus metrics
  worker/             Celery app and tasks (processing, clip cutting, indexing)
  web/chat.html       the web chat
  bot/telegram.py     the Telegram bot
embedder/             CLIP embedding service
eval/                 labelled evaluation sets and saved results
docs/decisions/       architecture decision records
infra/                Prometheus, Grafana (provisioned dashboards), Airflow, SeaweedFS config
scripts/              evaluations, reports, dashboard generation, test-material builders
tests/                unit and API tests
```
