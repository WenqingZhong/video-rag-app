# ADR 0004: Observability and caching: a token ledger, our own traces, Prometheus + Grafana, two Redis caches

- **Status:** accepted
- **Date:** 2026-09-27
- **Evidence:** [`eval/results/usage_baseline.json`](../../eval/results/usage_baseline.json) (`make usage`),
  [`eval/results/cache.json`](../../eval/results/cache.json) (`make eval-cache`)

## In one minute

Week 5 answered requests but couldn't say what they cost, where the time went, or avoid repeating work. Now:
- every model call is recorded with its tokens and an **estimated cost**;
- every request and Celery task is a **trace** of timed steps, across the API and the workers;
- **Grafana** shows both, next to live request metrics from **Prometheus**;
- two **Redis caches** skip the model (and the search) for repeated requests.

What the measurements showed:
- an `/ask` costs **~448 tokens**, 95% of them the fixed instructions;
- a keyframe caption costs **~225 tokens**;
- the model is **~75% of `/ask` time**, and captioning is **~92% of processing time** for a short video;
- a cached answer takes **4 ms instead of 2.2 s** (median), spends **0 tokens**, and is identical to a fresh answer (48/48).

## Context

The model runs locally, so nothing billed us, but tokens are GPU/CPU time, and a hosted model would bill per token.
We needed to know:
- how many tokens each request and each video consumes, and what that would cost;
- which step is slow;
- how much a cache could save, without ever serving a wrong answer.

## Options for monitoring

| Option | Verdict |
|---|---|
| **Langfuse, self-hosted** (the course's choice) | Rejected: version 3 needs ClickHouse plus several more containers, on a Mac that already stalled Docker |
| **Langfuse Cloud** | Rejected: requests would leave the machine, one of the reasons for a local model |
| **OpenTelemetry + Jaeger/Tempo** | Not needed yet: two more services for a system with one API host. Revisit if it grows |
| **Our own: Postgres tables + Prometheus + Grafana** | **Chosen:** two light containers; tokens and traces in SQL we control; the workers report too |

## The design

```text
API request ──▶ TracingMiddleware: trace id = X-Request-ID ──▶ spans: understand · llm.understand · search.* · clip
                   │  Celery message header carries the trace id ──▶ worker spans: task clip.cut · ffmpeg.cut · stage.*
                   ▼
Postgres   llm_calls    one row per model call: tokens, time, cost priced when recorded; cache hits with tokens saved
           trace_spans  one row per timed step; deleted after 14 days (Airflow "maintenance" DAG)
Prometheus /metrics     request rate and latency by route, /ask results, what understood them, cache hits
Grafana    "Tokens & cost" (Postgres) · "Requests & latency" (Prometheus + Postgres), provisioned from code

/ask ──▶ answer cache (24 h) ── hit ──▶ response (no model, no search)
           │ miss
           ▼
        understanding cache (7 d) ── hit ──▶ skip the model ──▶ search ──▶ clip ──▶ store the answer
```

| Decision | Why |
|---|---|
| Cost = our token counts × a hosted list price (default: Claude Haiku 4.5, $1 / $5 per million in / out), **stored with each call** | The model is free locally; a reference price makes tokens comparable. Stored per row, like a bill: changing the price doesn't rewrite history |
| Record understanding calls **after the guards** | The outcome (`accepted` / `adjusted` / `rejected` / `invalid`) shows tokens spent on answers that were thrown away |
| Traces saved **after the response** is sent, one insert per trace | Tracing never adds to what the user waits for, and a failed save never fails a request |
| Trace id in the **Celery message header** | A clip cut or an ingestion appears inside the trace of the request that caused it |
| Metrics labelled by **route template** (`/videos/{video_id}`) | One time series per endpoint, not one per video |
| Cache keys include a **fingerprint of the code and settings** | Editing the prompt, guards or search changes the key: nobody has to remember a version number |
| The answer key includes an **index version**, bumped after every index write | A video added today can't be hidden behind yesterday's cached "no match" |
| Clips stay in S3; only their keys are cached. A missing file means answer again | The response's presigned URLs are always fresh |
| A result made because the model **failed** is not cached | Next time the model may work |

## What was measured

**Tokens** (`make usage`, baseline over the library and the 58 evaluation requests):

| | tokens per call | cost per 1,000 calls |
|---|---|---|
| understanding one `/ask` request | 448 (≈ 423 in, 25 out) | $0.54 |
| captioning one keyframe | 225 (≈ 196 in with the image, 29 out) | $0.34 |
| captioning footage | ~1,700–2,400 tokens per minute of video | |

- The understanding prompt is ~420 tokens whatever the request: **~95% of its tokens are the fixed instructions.**
- Ollama counts the full prompt on every call, but reuses its work: reading the prompt drops from **4.9 s** (cold) to
  **~0.1 s**. Writing the reply (~15 tokens/s on this CPU) is where the time goes.
- **6%** of understanding tokens went to answers the guards rejected.

**Time** (traces):

| | step | share |
|---|---|---|
| `/ask`, clip not cached (3.9 s) | model call 2.95 s · search 0.11 s · clip cut 0.84 s (ffmpeg 0.56 s) | model ≈ 75% |
| processing a 5 s video without audio (7.9 s) | captioning 7.2 s · every other stage < 0.15 s | captioning ≈ 92% |

**Caching** (`make eval-cache`, 48 requests: 36 search + 12 no-answer):

| pass | p50 | p95 | cache hits | tokens spent | tokens saved |
|---|---|---|---|---|---|
| nothing cached | 2,158 ms | 3,443 ms | – | 21,377 | 0 |
| same requests, in CAPITALS | **4 ms** | 5 ms | 48/48 answers | 0 | 21,377 |
| same requests, `max_clips=2` | **55 ms** | 584 ms | 48/48 understandings | 0 | 21,377 |
| after re-indexing one video | 55 ms | 65 ms | 0 answers · 10/10 understandings | 0 | 4,444 |

- **Cached answers were identical** to fresh ones: 48/48 (status, text, clips). After the re-index, none was
  served stale, and all 10 fresh answers matched.
- These are best-case numbers: every request was repeated. Real savings depend on how often users repeat requests;
  the dashboard shows the real hit rate.

**Found by the new dashboards:** the "Recent errors" table showed a failed `video.process` every few minutes, each
for a video that didn't exist. A test (since Week 2) was sending real jobs to the running broker; the worker picked them
up. Tests now never reach the broker.

## Consequences

- **Cost is an estimate:** a hosted model counts tokens differently (especially images), and would discount
  cached prompt tokens. The dashboard names the price it uses.
- **Any edit to the fingerprinted code empties that cache**, even a comment. Deliberate: a stale cache is worse than
  a cold one.
- **Case and spacing are ignored** in cache keys: a quote's capitals come from whoever asked first. Search ignores
  case, so results are the same.
- **Traces are kept 14 days** (`TRACE_RETENTION_DAYS`); token history is kept.
- **Local-dev shortcuts:** Grafana has no login and reads Postgres with the app's credentials. In production: a
  login, and a read-only database role on `llm_calls`, `trace_spans` and `videos`.
- **The first task in a freshly started worker** also connects to the database and checks migrations; that time
  shows as a gap before its first stage.

## When to revisit

- **Real traffic:** check the cache hit rate before tuning anything else.
- **The fixed instructions are 95% of understanding tokens:** a shorter prompt is the biggest token saving left, but
  needs the intent evaluation re-run (ADR 0002).
- **More than one API host, or services in other languages:** move to OpenTelemetry.
