# ADR 0003: Clip delivery: real MP4s, re-encoded, cached, on a dedicated queue; template answers

- **Status:** accepted
- **Date:** 2026-09-27

## Context

`/ask` must return **the clip itself**, not a list of search results. That raises five questions:
- what file is returned;
- how it's cut;
- where it starts and ends;
- who cuts it, without users waiting behind long jobs;
- what text comes with it.

## Decisions

| Question | Decision | Rejected, and why |
|---|---|---|
| What is returned | a **real MP4** in S3, `clips/{video_id}/{start_ms}-{end_ms}.mp4`, via a presigned URL | `#t=start,end` on the full video: not a downloadable, shareable clip |
| How it's cut | **re-encode** (libx264 veryfast, CRF 23, `+faststart`) | **stream copy:** it can only start at a compression keyframe. Measured on the talk video: asked for 8.73–11.61 s (2.88 s), a copy produced **3.11 s** with no full frame at its start (nearest one: 7.96 s) |
| Where it starts and ends | quote → matched **words ± 0.75 s**; visual/topic → the shot, at most 15 s, centred on the keyframe | fixed windows: cut through words |
| Repeat requests | **cached** by that key (times rounded to 10 ms, so identical requests share a key) | re-cutting every time |
| Who cuts | a **`clip-worker`** that serves only the `clips` queue (2 at a time, ffmpeg + S3, ~270 MB) | the processing worker (1 job at a time, jobs take minutes): a user would wait behind video processing |
| The text | a **fixed template** from search results, e.g. *At 0:09–0:11 in "host_talk": "AI is changing everything."* | LLM-written prose: adds seconds on CPU, and could state things not in the video |
| Nothing to show | `needs_subject` → ask what to show; `no_match` → say so, **no clip** | returning the nearest thing anyway |

## Evidence

- **Accuracy:** the cut quote clip was **transcribed back with Whisper**. It contains "AI is changing everything", with
  "AI" at **9.49 s** in video time, where search placed it at 9.48 s.
- **Speed:**
  - a fresh short cut takes 0.1–0.9 s;
  - a cache hit takes ~1–8 ms;
  - `/ask` end to end is ~1–1.7 s, most of it the LLM (ADR 0002).
- **Isolation:** while the processing worker was busy with a 46 s video (27 s still to go), a new clip was delivered
  in **0.55 s**.
- **`+faststart`:** the MP4 index (`moov`) sits at byte 36 instead of at the end (byte 28,431 of 31,674 without it),
  so browsers start playing before the download finishes.

## Consequences

- **One more container,** small: it never loads a model.
- **If the clip worker is down, clip jobs wait in their queue** and `/ask` times out (504). `/health` reports
  `clip_worker` separately. A fallback would be the processing worker also listening to `clips`, at the cost of
  isolation.
- **The API waits in a thread** for each cut (up to 60 s): fine for short clips. Long clips would need a job id to check later.
- **The worker downloads the whole source for each cut:** fine for short stock clips, wasteful for hour-long talks.
  ffmpeg could read just the needed range from a signed S3 URL.
- **Timestamps in answers always come from search results,** never from a model.
- **Cached clips accumulate under `clips/`.** They're derived data, so an S3 lifecycle rule can expire them.
