# ADR 0006: Private uploads without accounts: an anonymous owner, a filter in every query, deletion everywhere

- **Status:** accepted
- **Date:** 2026-09-29
- **Evidence:** `tests/unit/test_privacy.py`, `tests/api/test_privacy.py`, and a live two-browser run (below)

## In one minute

Before, every upload joined the shared library: anyone could find anyone's video. Now:
- each request has a **viewer**: a signed anonymous cookie on the web, `tg:<user id>` from the Telegram bot;
- an upload belongs to its viewer; **every search** returns the shared library plus the viewer's own videos only;
- uploads are **deleted after 7 days**, or when the user asks ("delete my video" → "yes"), from the search index,
  S3 and Postgres;
- the existing library (Pexels footage and earlier test uploads) has no owner: it stays shared and never expires.

## Options for identity

| Option | Verdict |
|---|---|
| **Accounts (email or OAuth)** | Not yet: sign-up is friction for a demo, and a password store is a liability. The owner column works the same once accounts exist |
| **Unsigned cookie / client-sent id** | Rejected: anyone could send someone else's id |
| **Signed anonymous cookie + a service token for the bot** | **Chosen:** no sign-up; a 128-bit random id, HMAC-signed, so it can't be guessed or forged |

## The design

```text
browser ── cookie vr_session = <random id>.<HMAC> ─┐
Telegram bot ── x-service-token + x-user-id: tg:42 ┴─▶ viewer ──▶ owner of new uploads
                                                              └─▶ search filter: no owner OR owner = viewer
```

| Decision | Why |
|---|---|
| The filter is inside the OpenSearch query, **kNN included** (`knn.filter`), not applied to results afterwards | Filtering after retrieval could return fewer than k results, or none, when others' videos rank first |
| Someone else's video is a **404**, never a 403 | A 403 would confirm the video exists |
| The answer cache is keyed by viewer **only if they own videos** | Everyone without uploads sees the same results, so they still share one cache |
| A conversation remembers its owner; another viewer's conversation id starts a **new** conversation | Its memory names the owner's videos, and could confirm their deletion |
| Deleting: the index first, then S3, then Postgres; then bump the index version | The video disappears from results even if a later step fails; the row stays until the end, so a failed run can be retried; no cached answer can point at a deleted video |
| Deleting from the chat is **rules only, with a yes/no confirmation**, never the model | It can't be undone, and the model is the part that can be wrong |
| A video being downloaded or processed is not deleted (409; retention retries next day) | A worker is still writing its files |
| Retention is a daily Airflow task calling `POST /admin/cleanup-uploads` | The same pattern as trace cleanup: the DAG only calls the API |
| Admin jobs (reindex, enrich) list **all** videos (`everything=True`) | The default is now the public library: without this a reindex would silently drop users' uploads |

## What was checked

- Search evaluation after reindexing to index v3 (with `owner_id`): **identical** to before (hybrid Recall@5 0.97,
  MRR 0.935), since the whole library is shared.
- Live, two browsers A and B, A uploads a video:
  - A sees it (`GET` 200, listed, in the chat's video list); B gets 404 on `GET`, `/clips` and `DELETE`, and doesn't see it
    in lists or the chat;
  - B polling A's conversation id gets a new conversation, not A's;
  - A: "delete my video" → asks to confirm → "yes" → deleted; 0 documents in OpenSearch, 0 files under `raw/`, `frames/`,
    `clips/`, 0 rows.

## Consequences

- **Clearing cookies loses access** to your uploads (they still expire after 7 days). Accounts would fix this.
- **The bot is trusted** to name its users: the service token must stay secret, like the bot token.
- `SESSION_SECRET` and `SERVICE_TOKEN` have development placeholders; production must set them, and `COOKIE_SECURE=true`.
- Presigned clip URLs work for anyone holding the link until they expire (1 hour), like any shared link.
