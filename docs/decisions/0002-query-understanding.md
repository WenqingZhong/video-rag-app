# ADR 0002: Request understanding: a local LLM, checked by guards, with rules as a safety net

- **Status:** accepted
- **Date:** 2026-09-27
- **Evidence:** [`eval/results/intents.json`](../../eval/results/intents.json) (`make eval-intents`),
  [`eval/results/search.json`](../../eval/results/search.json) (`make eval`)

## In one minute

`/ask` turns a plain request into an **intent**:
- a type: **quote** (words someone says), **topic** (something discussed) or **visual** (something shown);
- the text to search for;
- what to **exclude**.

**The model:** `qwen2.5vl:3b`, reused from captioning, called through Ollama's JSON-schema mode so its reply always
parses.

**The model alone isn't reliable,** so plain-code **guards** check its answer against the user's own words, and a
**rule parser** is the fallback.

**On search quality** (the `understood` column), compared with Week 4's hybrid search:
- **MRR 0.940 → 0.954**;
- the negation query fixed (0.50 → 1.00);
- **false answers 7 → 2 of 12** (requests that should return nothing).

## Context

Week 4 searched the raw sentence. It couldn't handle quotes without quote marks, "where do they talk about…", or
negation: "a beach with **no** people" returned the beach *with* people. It also **answered almost anything**: "hi",
"an elephant" and "a spaceship landing on mars" each returned a clip, because vector search always returns the
nearest keyframes.

## Options

1. **Rules only:** extended regex.
2. **Local LLM, reusing the caption model** (chosen). Ollama keeps one model in memory, so a second model would be
   reloaded every time captioning and `/ask` alternate.
3. **A separate local text model:** not tested; rejected for the reloading cost.
4. **A hosted LLM:** cost per call, and requests would leave the machine.

## The design

```text
request ─▶ LLM (JSON schema) ─▶ guards (plain code, checked against the request) ──▶ intent
              │ error / invalid       1. type:      quote/topic need a speech word or quote marks, else → visual
              ▼                       2. fields:    keep only the field for the type
            rules ◀───────────────── 3. grounding: the search text must use the user's words, else → rules
                                      4. exclusions: LLM's ∪ rules', only words the user said, overlaps merged
no subject ("exclude people", "hi") → /ask asks what to show instead of searching
search: quote/topic found nothing → retry as visual · hybrid: vector-only results need similarity ≥ 0.20
```

## What was measured, in order

| Finding | Evidence | Response |
|---|---|---|
| The LLM invents details and fills irrelevant fields | "a dog" → "a dog **sitting on a couch**" | a stricter prompt; guards 2 and 3 |
| The LLM drops negations | "a beach with no people" → `exclude: []` | guard 4: add the rule parser's exclusions |
| The LLM copies prompt examples when a request has no subject | "exclude people" → **"a horse running"** | a prompt rule + "no subject" example: **4 of 5** subject-less requests fixed at the source. Removing the copied example made it copy *another* ("a street"), so it was kept; guard 3 catches the rest |
| The LLM routes bare nouns to speech search | "canine", "animals" → *topic* → searches speech only → nothing (search MRR fell to 0.72) | guard 1 (type), plus the visual fallback |
| **Vector search answers everything** | 8 of 12 no-answer requests got a clip | **Fix A:** chit-chat counts as "no subject". **Fix B:** a sweep of the vector-only cut-off (table below) |

**Fix B sweep** (`understood` mode):

| vector-only cut-off | Recall@5 | MRR | false answers |
|---|---|---|---|
| 0.15 | 0.989 | 0.954 | 5 / 12 |
| **0.20 (chosen)** | 0.970 | 0.954 | **2 / 12** |
| 0.22 | 0.906 | 0.926 | 2 / 12 |
| 0.25 | 0.860 | 0.870 | 2 / 12 |

0.20 removes three false answers ("an elephant", "a spaceship…", "a snowboarder…") and costs one relevant video on
one query ("automobiles"). Stricter values only lose recall.

## Results

**Request understanding** ([`eval/intents.json`](../../eval/intents.json), 58 requests; "all correct" = type, text and
exclusions):

| category | n | rules only | LLM + guards + rules |
|---|---|---|---|
| phrasings the rules were designed for (5 categories) | 30 | 1.00 | 0.90 |
| free phrasing | 12 | 0.17 | **0.58** |
| held out (labels fixed before any run) | 8 | 0.75 | 0.62 → **0.75** after one fix* |
| no subject (exclusion-only, placeholders, greetings) | 8 | 0.88 | 0.88 |
| **overall** | 58 | 0.78 | **0.81** (p50 latency 1.2 s) |

\* The held-out run exposed a gap in guard 1's word list: "tells" was missing, so "the clip where she **tells** viewers
see you next week" lost its quote. It was fixed; the 0.75 is measured **after** seeing the held-out set, so it is no
longer an unseen score.

**Search quality** (36 queries from ADR 0001, plus the 12 no-answer requests):

| mode | Recall@5 | MRR | negation | false answers |
|---|---|---|---|---|
| hybrid, raw text (Week 4 + Fix B) | 0.970 | 0.940 | 0.50 | 7 / 12 |
| **understood** (Week 5) | 0.970 | **0.954** | **1.00** | **2 / 12** |

## Consequences

- `/ask` spends **~1–1.3 s** on the LLM (after the first request loads the model; a cold start took 7 s).
- **If Ollama is down or slow, `/ask` still works:** the rule parser takes over, and every response reports how the
  request was understood (`source`, guard `notes`).
- **Known misses:**
  - the LLM drops descriptive words: "city traffic **at night**" → "city traffic";
  - the type guard misroutes spoken requests without a speech word: "which part **covers** the future of work",
    "the goodbye at the end, see you next week". Accepted: bare-noun visual requests are more common here;
  - **2 false answers remain**, from other causes: "someone playing the piano" matches common words in keyword
    search, and "the stock market crashed" finds no quote, falls back to visual, and gets a weak match there.
- **Tuning stopped deliberately.** Each set was used for at most one round of changes, to avoid overfitting. Further
  changes need **new** requests.

## When to revisit

- **Real user requests become available:** re-measure on those.
- **The library gains many talks:** re-check the balance of the type guard.
- **A stronger small model is available** at similar memory.
