# ADR 0007: Usage limits: a daily token allowance per user, request rates, and a global budget

- **Status:** accepted
- **Date:** 2026-09-29
- **Evidence:** `tests/unit/test_limits.py`, `tests/api/test_limits.py`, and a live run (below)

## In one minute

A public deployment with a hosted model (Bedrock) pays per token, and anyone can open a new anonymous session. So:
- each user gets a **daily allowance of 50,000 tokens**, counted from every model call they cause, including captioning
  their own uploads; cached answers cost nothing;
- **requests per minute** are limited per user (20) and per address (60), because clearing cookies creates a new user
  but not a new address;
- **uploads** (5 a day, up to 10 minutes long) and **Pexels downloads** (3 a day) are limited per user;
- a **global budget** (2 million tokens a day, ≈ $3 at the reference price) stops everything before the bill grows;
- operators (the admin token) are exempt; admin, trace and bulk-ingestion endpoints now require it.

## The design

```text
request ─▶ PrincipalMiddleware: Principal(ip, exempt=admin token?) in a context variable
             └─▶ get_viewer: principal.viewer = the web session / Telegram user
         ─▶ check_limits (chat, ask, answer, search, clips): rate per minute, then today's tokens → 429 + a sentence
         ─▶ model calls ─▶ UsageRecorder.record ─▶ llm_calls row  +  Redis: tokens for viewer, IP, and all
worker: processing a video acts for its owner, so captioning counts against the uploader
```

| Decision | Why |
|---|---|
| The allowance is in **tokens**, not requests | Costs differ ~10× between a rule-routed search and a question plus an upload |
| Tokens are counted where they are **already recorded** (the usage ledger) | One place sees every model call, API and worker alike; nothing to forget when adding a feature |
| Checked **before** a request, counted **after** | Token counts are only known afterwards. The request that crosses the line finishes; the next is refused |
| Counters in **Redis** with a TTL, per UTC day | Atomic increments shared by all API workers; they clean themselves up |
| **Redis down → no limits** (logged) | An outage of the counter shouldn't take the app down; the global budget is the backstop once Redis is back |
| Per **address** too, but not for the bot's users | The bot sends every user from one address; it vouches for each user instead (service token) |
| A limit is a **429 whose `detail` is a sentence** | The web page and the bot show it as it is: "You've used today's allowance of 50,000 tokens. It resets in 3 h 12 min" |
| Too-long uploads fail at the **probing** step | Before any model call: a refused video costs no tokens |
| Admin uploads join the **shared library** (no owner, no expiry) | How the library is built (e.g. the talk video used by the QA evaluation) |

## What was checked (live)

- One question turn counted **858 tokens**; the ledger rows for that request: answer 459 + route 399 = **858**.
- Uploading a 6-second video counted **216 tokens**; the ledger's caption rows for that video: **216**.
- 22 rapid messages: 20 answered, then "You're sending requests a bit fast. Please wait … seconds".
- 25 rapid messages with the admin token: all answered (evaluation scripts are exempt).

## Consequences

- **The Grafana "Open trace" link** opens `/api/v1/traces/{id}`, which now needs the admin token, so a browser gets
  403. Use `make trace ID=…` (reads Postgres directly).
- **A new cookie per request is possible for scripts**; the per-address limits bound them. Behind a proxy, the API
  must trust its `X-Forwarded-For` (uvicorn `--proxy-headers`), or every user shares the proxy's address.
- Limits are settings (`LIMIT_*`), so they can be tuned from the dashboards' real usage.

## Update (2026-09-30): with Claude in production

Claude Haiku 4.5 uses about **1,800 tokens per chat turn**, against ~400 for qwen2.5vl:3b: forcing a JSON reply
through a tool call adds a fixed few hundred tokens to every call (`eval/results/claude/conversations.json`). The cost
ceiling was kept rather than the number of requests:
- global budget **500k tokens/day** in production (≈ $0.70/day at most);
- **50k tokens per user** unchanged, which is now a few dozen chat turns rather than ~100; the greeting and the web
  page say "a few dozen requests".
