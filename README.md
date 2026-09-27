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

## Search quality & design decisions

Retrieval changes are judged with a labelled evaluation set, not by eyeballing results:

- [`eval/queries.json`](eval/queries.json): 36 queries with relevant videos (by Pexels id, labelled from the actual
  keyframes). 16 **basic** + 20 **hard**, grouped by the weakness they probe: `style`, `detail`, `synonym`, `count`,
  `color`, `negation`.
- `make eval`: Recall@5 and MRR for keyword / vector / hybrid, overall and **per category**.

| mode | Recall@5 | MRR | style | detail | synonym |
|---|---|---|---|---|---|
| keyword | 0.717 | 0.722 | 0.33 | 1.00 | 0.00 |
| vector (CLIP) | 0.989 | 0.926 | 1.00 | 0.83 | 1.00 |
| **hybrid** | **0.989** | **0.940** | 0.83 | 1.00 | 1.00 |

Decisions backed by experiments live in [`docs/decisions/`](docs/decisions/):

- [ADR 0001: Don't add caption embeddings](docs/decisions/0001-no-caption-embeddings.md). Embedding the keyframe
  captions as a third retriever changed 2 of 36 queries (one better, one worse) and lowered hard-query MRR
  (0.917 → 0.902). Keyword and CLIP already complement each other, and caption embeddings inherit caption errors.
  Reproduce: `make experiment-captions` → [`eval/results/caption_embeddings.json`](eval/results/caption_embeddings.json).

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
  db/migrations/       # Alembic migrations + startup runner
embedder/              # CLIP embedding service (own image: torch CPU + open_clip)
eval/queries.json      # labelled search queries (relevance by Pexels id, with categories)
eval/results/          # saved experiment results (evidence for docs/decisions)
docs/decisions/        # architecture decision records (ADRs)
scripts/evaluate_search.py  # Recall@k / MRR per search mode and category (make eval)
scripts/experiment_caption_embeddings.py  # caption-embedding experiment (make experiment-captions)
airflow/dags/          # pexels_ingestion DAG
infra/airflow/start.sh # creates the airflow metadata DB, runs `airflow standalone`
infra/seaweedfs/s3.json  # local S3 credentials (dev only; must match .env)
tests/
```
