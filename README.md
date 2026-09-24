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
| `ollama` | Local LLM | 11434 |

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
  worker/              # Celery app + tasks
infra/seaweedfs/s3.json  # local S3 credentials (dev only; must match .env)
tests/
```
