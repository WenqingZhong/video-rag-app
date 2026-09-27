# Video RAG

A video-first RAG system: retrieve the right **clip** from a video library, either by what is
*shown* ("give me a clip of a dog") or by what is *said* ("the part where the host says
'AI is changing everything'"). Infrastructure follows the production-agentic-rag-course; the
multimodal retrieval approach follows the multimodal-agents-course.

## Services

| Service | Purpose | Port |
|---|---|---|
| `api` | FastAPI REST API | 8000 |
| `worker` | Celery worker for video ingestion/processing | – |
| `postgres` | Video / segment / job metadata | 5432 |
| `redis` | Cache (db 0), Celery broker (db 1) and results (db 2) | 6379 |
| `seaweedfs` | S3-compatible object storage for raw videos, frames and clips | 8333 |
| `opensearch` / `opensearch-dashboards` | Hybrid (BM25 + vector) segment index | 9200 / 5601 |
| `ollama` | Local models: `qwen2.5vl:3b` captions keyframes (one model loaded at a time) | 11434 |
| `airflow` | Scheduler: daily Pexels ingestion DAG (calls the API) | 8080 |
| `clip-worker` | Celery worker for the `clips` queue only: cuts clips (never waits behind video processing) | – |
| `embedder` | CLIP model server (ViT-B/32): text + image → 512-d vectors | 8001 |

Object storage is accessed purely through the S3 API (boto3). MinIO no longer publishes community
container images, so SeaweedFS is used locally; point `S3_ENDPOINT_URL` at any S3-compatible store,
or leave it empty to use AWS S3.

## Quick start

```bash
cp .env.example .env
uv sync
make start          # docker compose up --build -d
make health         # GET /api/v1/health -> every dependency's status
```

- `GET /api/v1/ping`: liveness (process is up)
- `GET /api/v1/health`: readiness (`ok`, or `degraded` with per-service details)
- Docs: http://localhost:8000/docs
- Airflow: http://localhost:8080 (local dev: no login; DAGs start paused)

## Ingestion (Week 2)

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/videos` | upload a file (multipart) → 202, processed in the background |
| `POST /api/v1/videos/pexels` `{query, count}` | queue `count` new Pexels videos (idempotent) |
| `GET /api/v1/videos[?status=&source=]` | list videos |
| `GET /api/v1/videos/{id}` | status / stage / error + metadata + presigned video URL (poll this) |
| `GET /api/v1/videos/{id}/segments[?kind=speech\|visual]` | segments with word timestamps / keyframe URLs |
| `POST /api/v1/videos/{id}/reprocess` | re-run the (idempotent) pipeline, e.g. after `failed` |

Worker pipeline (`video.process`): ffprobe → ffmpeg scene detection → shots → keyframes (S3 `frames/`) →
16 kHz audio → faster-whisper with word timestamps → overlapping speech windows → `segments` in Postgres.
Pexels videos first go through `video.download_pexels` (best mp4 ≤ 720p, max 60 s).

## Search (Week 3)

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/search` `{query, video_id?, kind?, source?, size?, group_by_video?}` | quoted text → phrase search with exact word times; otherwise BM25 keyword search |
| `POST /api/v1/admin/reindex` `{video_id?}` | rebuild the search index from Postgres (all ready videos, or one) |
| `GET /api/v1/admin/jobs/{job_id}` | state/result of any Celery job |

- Index: alias `video_segments` → `video_segments_v1`, one document per segment (Postgres stays the source of truth).
  Transcripts are indexed twice: `text` (exact, for quotes) and `text.stemmed` (English, for keywords).
- Quotes (`‘…’ “…” '…' "…"`) search speech with a cascade: exact phrase → phrase with slop → fuzzy. Each hit
  reports `match_start_sec`/`match_end_sec` from word timestamps and a `play_url` (`…#t=start,end`).
- The pipeline's last stage is `indexing`: a video is `ready` once it is searchable.

## Visual & hybrid search (Week 4)

- Each keyframe gets a **CLIP vector** (`embedder` service) and a **caption** (`qwen2.5vl:3b` via Ollama), stored in
  Postgres (`segments.image_embedding`, `caption`) and indexed (`knn_vector`, HNSW, cosine).
- Queries without quotes run **hybrid** search: BM25 (transcript + caption + title) and kNN (CLIP text vector vs
  keyframes), fused with **Reciprocal Rank Fusion**. `mode` = `auto` | `hybrid` | `keyword` | `vector`.
- `POST /api/v1/admin/enrich {video_id?, force?}`: backfill vectors + captions from keyframes already in S3.
- `POST /api/v1/admin/reindex`: when the index is older than the code (`INDEX_VERSION`), builds the new versioned
  index from Postgres and switches the alias atomically (blue/green; old index kept for rollback).
- Schema changes use **Alembic** (`src/db/migrations`), applied automatically at startup.
  New migration: `uv run alembic revision --autogenerate -m "..."`.

## Ask: a request in, a clip out (Week 5)

`POST /api/v1/ask {"query": "...", "video_id"?: "...", "max_clips"?: 1}`

```text
request ─▶ understand (qwen2.5vl:3b via Ollama JSON schema → quote | topic | visual + exclusions; guards; rule fallback)
        ─▶ search by intent (quote: phrase + word times · topic: speech keywords · visual: hybrid over keyframes; exclusions filtered)
        ─▶ cut the clip (clip-worker, ffmpeg re-encode, cached in S3 clips/) ─▶ template answer
```

| Request | Answer |
|---|---|
| "Give me the part where the host says AI is changing everything" | an MP4 cut at the exact words + *At 0:09–0:11 in "host_talk": "AI is changing everything."* |
| "give me a clip of a dog" | an MP4 of a dog + title, Pexels credit, caption |
| "a beach with no people" | the **empty** beach (negation understood) |
| "show me a dragon breathing fire" | `no_match`: *No moment matched footage of "a dragon breathing fire".* (no clip) |
| "exclude people" / "hi" | `needs_subject`: *What would you like to see or hear?* |

- `POST /api/v1/clips {video_id, start_sec, end_sec}`: cut any range (word-accurate, cached).
- `POST /api/v1/search {"understand": true}`: the same understanding, without cutting.
- `make eval-intents`: understanding accuracy (LLM vs rules) on `eval/intents.json`.

## Search quality & design decisions

Retrieval changes are judged with a labelled evaluation set, not by eyeballing results:

- [`eval/queries.json`](eval/queries.json): 36 queries with relevant videos (by Pexels id, labelled from the actual
  keyframes). 16 **basic** + 20 **hard**, grouped by the weakness they probe: `style`, `detail`, `synonym`, `count`,
  `color`, `negation`.
- `make eval`: Recall@5 and MRR for keyword / vector / hybrid, overall and **per category**.

| mode | Recall@5 | MRR | style | detail | synonym | negation | false answers |
|---|---|---|---|---|---|---|---|
| keyword | 0.717 | 0.722 | 0.33 | 1.00 | 0.00 | 0.50 | 4 / 12 |
| vector (CLIP) | 0.989 | 0.940 | 1.00 | 0.83 | 1.00 | 0.50 | 9 / 12 |
| hybrid | 0.970 | 0.940 | 0.83 | 1.00 | 1.00 | 0.50 | 7 / 12 |
| **understood** (`/ask`) | **0.970** | **0.954** | 0.83 | 1.00 | 1.00 | **1.00** | **2 / 12** |

*False answers:* requests that should return nothing ([`eval/no_answer.json`](eval/no_answer.json): greetings, and things
not in the library).

Decisions backed by experiments live in [`docs/decisions/`](docs/decisions/):

- [ADR 0001: Don't add caption embeddings](docs/decisions/0001-no-caption-embeddings.md). Embedding the keyframe
  captions as a third retriever changed 2 of 36 queries (one better, one worse) and lowered hard-query MRR
  (0.917 → 0.902). Keyword and CLIP already complement each other, and caption embeddings inherit caption errors.
  Reproduce: `make experiment-captions` → [`eval/results/caption_embeddings.json`](eval/results/caption_embeddings.json).
- [ADR 0002: Request understanding](docs/decisions/0002-query-understanding.md). A local LLM checked by plain-code
  guards, with rules as fallback. Rules alone: 1.00 on phrasings they were written for, 0.17 on free phrasing. The guards
  and a vector-only similarity cut-off (0.20, chosen by a sweep) cut false answers from 7 to 2 of 12.
- [ADR 0003: Clip delivery](docs/decisions/0003-clip-delivery.md). Real MP4s, re-encoded for word accuracy (verified
  by transcribing a clip back), cached, cut on a dedicated queue so users never wait behind video processing.

Requires `PEXELS_API_KEY` in `.env` (free at https://www.pexels.com/api/). Attribution (author, page URL) is stored per video.

## Local development

Run the infra in Docker and the Python code on the host (the `.env.example` values target `localhost`;
`compose.yml` overrides hosts for containers):

```bash
make run      # API with --reload
make worker   # Celery worker
make test     # unit + API tests (no infra needed)
make lint
```

## Layout

```text
src/
  main.py              # app + lifespan (creates shared clients)
  config.py            # pydantic-settings
  dependencies.py      # FastAPI dependency providers
  routers/ping.py      # /ping, /health
  schemas/api/         # response models
  db/                  # PostgreSQL interface + factory
  services/cache/      # Redis client
  services/storage/    # S3 client (raw/, frames/, clips/ prefixes)
  services/pexels/     # Pexels API client + response models
  services/ingestion/  # upload / Pexels ingestion (Phase 1: store + enqueue)
  services/processing/ # ffmpeg, segmentation, faster-whisper, VideoPipeline (Phase 2)
  models/, repositories/  # SQLAlchemy models (videos, segments) + data access
  routers/videos.py    # /videos endpoints
  routers/search.py, routers/admin.py  # /search, /admin/reindex, /admin/jobs
  services/opensearch/ # index definition (alias + versioned index), indexing, search, health
  services/indexing/   # Postgres rows -> OpenSearch documents; index_video()
  services/search/     # quote parsing, query builder, phrase locator (word times), SearchService
  worker/              # Celery app + tasks (video.download_pexels, video.process, index.rebuild)
  services/embeddings/ # client for the embedding service
  services/captioning/ # Ollama vision-model captioner
  services/processing/visual.py  # keyframes -> vectors + captions (pipeline + backfill)
  services/search/fusion.py      # Reciprocal Rank Fusion
  services/understanding/        # request → intent: LLM (Ollama JSON schema) + guards + rule fallback
  services/clips/                # clip boundaries, cache keys, get-or-cut
  services/answering/            # /ask orchestration + template answers
  routers/ask.py, routers/clips.py  # /ask, /clips
  db/migrations/       # Alembic migrations + startup runner
embedder/              # CLIP embedding service (own image: torch CPU + open_clip)
eval/queries.json      # labelled search queries (relevance by Pexels id, with categories)
eval/results/          # saved experiment results (evidence for docs/decisions)
docs/decisions/        # architecture decision records (ADRs)
scripts/evaluate_search.py  # Recall@k / MRR per mode and category + false answers (make eval)
scripts/evaluate_intents.py # understanding accuracy, LLM vs rules (make eval-intents)
eval/intents.json           # 58 labelled requests (incl. held-out, free phrasing, no subject)
eval/no_answer.json         # requests that should return no clip
scripts/experiment_caption_embeddings.py  # caption-embedding experiment (make experiment-captions)
airflow/dags/          # pexels_ingestion DAG
infra/airflow/start.sh # creates the airflow metadata DB, runs `airflow standalone`
infra/seaweedfs/s3.json  # local S3 credentials (dev only; must match .env)
tests/
```
