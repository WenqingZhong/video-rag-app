# ADR 0001: Don't add caption embeddings as a third retriever

- **Status:** accepted
- **Date:** 2026-09-27
- **Evidence:** [`eval/results/caption_embeddings.json`](../../eval/results/caption_embeddings.json), reproduced with `make experiment-captions`

## In one minute

Every keyframe already gets a caption from a vision model (for keyword search and for the LLM), so it seemed natural to
**also embed the captions** and search them by meaning. We tested it as a third retriever next to keyword (BM25) and
CLIP, on 36 labelled queries (16 basic + 20 designed to be hard).

**It changed exactly 2 of 36 queries: one better, one worse.** Overall MRR moved 0.940 → 0.945 (noise), and MRR on the
hard queries *dropped* 0.917 → 0.902. The reason is that keyword and CLIP already **cover each other's weaknesses**:
- CLIP handles camera/style and synonyms, which captions can't express;
- keyword search over captions handles small details that CLIP misses.

Caption embeddings overlap with both, and they inherit every captioning mistake. Not worth a second model, a schema
migration, a new index version, a backfill, and an extra vector search per query.

## Context

Visual segments are searchable three ways today:

| Retriever | Looks at | Good at |
|---|---|---|
| **Keyword** (BM25) | transcript + **caption** words + title | exact words, small details named in the caption |
| **CLIP** (kNN) | the keyframe **pixels** | meaning, synonyms, camera/style, anything visible |
| *candidate:* **caption embeddings** | caption **meaning** (text-embedding model) | paraphrases of what the caption says |

Results are fused with Reciprocal Rank Fusion (RRF, k = 60).

## Options considered

1. **Keep keyword + CLIP** (current hybrid).
2. **Add caption embeddings** with a dedicated text model (`BAAI/bge-small-en-v1.5`), fused as a third retriever.
3. **Caption embeddings with CLIP's own text encoder** (no new model, text-to-text in CLIP space).
4. **Replace CLIP with caption embeddings** ("caption, then embed").

## How it was tested

- **Library:** 23 Pexels videos, 54 captioned keyframes.
  - CLIP: `ViT-B-32/laion2b_s34b_b79k`;
  - captions: `qwen2.5vl:3b`;
  - text embeddings: `bge-small-en-v1.5`.
- **Queries** ([`eval/queries.json`](../../eval/queries.json)): 36 queries, relevance labelled **per video** by Pexels
  id, set by looking at the actual keyframes (not just titles or captions). The 20 hard queries target where the
  retrievers were *expected* to differ:

  | Category | n | Designed to test | Example |
  |---|---|---|---|
  | basic | 16 | the Week 4 baseline | "give me a clip of a dog" |
  | style | 3 | camera/look: in the pixels, **never in captions** | "drone footage above the coast" |
  | detail | 6 | small objects/actions **named in captions** | "a bird eating from a metal bowl" |
  | synonym | 4 | words in **no** title or caption | "canine", "feline", "seashore" |
  | count | 4 | numbers (CLIP is known to be weak at counting) | "three people talking" |
  | color | 2 | colour words differing from the caption's | "turquoise water" |
  | negation | 1 | "no"/"without" | "a beach with no people" |

- **Method** (`scripts/experiment_caption_embeddings.py`):
  - keyword and CLIP rankings come from the **live API**; caption rankings are computed offline;
  - each retriever returns its top 30 segments, and they're fused with the **same RRF** as production;
  - rankings are reduced to one entry per video;
  - **metrics:** Recall@5 (share of relevant videos in the top 5, out of min(5, #relevant)) and MRR (1 / rank of the
    first relevant video).

## Results

### Overall

| System | all Recall@5 | all MRR | hard MRR |
|---|---|---|---|
| keyword | 0.717 | 0.722 | 0.625 |
| CLIP (images) | 0.989 | 0.926 | 0.900 |
| caption embeddings (bge) | 0.933 | 0.892 | 0.831 |
| caption embeddings (CLIP text encoder) | 0.793 | 0.799 | 0.716 |
| **keyword + CLIP (current)** | **0.989** | 0.940 | **0.917** |
| keyword + CLIP + caption embeddings | 0.989 | 0.945 | 0.902 |

### MRR per category: the part that explains *why*

| Category | keyword | CLIP | caption emb. | **keyword + CLIP** | + caption emb. |
|---|---|---|---|---|---|
| basic | 0.84 | 0.96 | 0.97 | 0.97 | 1.00 |
| style | 0.33 | **1.00** | 0.49 | 0.83 | 0.73 |
| detail | **1.00** | 0.83 | **1.00** | **1.00** | 1.00 |
| synonym | 0.00 | **1.00** | 0.88 | **1.00** | 1.00 |
| count | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| color | 0.50 | 0.75 | 0.58 | 0.67 | 0.67 |
| negation | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

### The only two queries the third retriever changed

| Query | keyword + CLIP | + caption emb. | Why |
|---|---|---|---|
| "people having a meeting" | #2 | **#1** | captions say "engaged in a conversation … professional setting", a close paraphrase |
| "drone footage above the coast" | **#2** | #5 | the aerial beach's caption says "a sandy beach…" and never "aerial"; caption embeddings rank it 8th and drag it down |

## Analysis

1. **Keyword and CLIP are complementary; caption embeddings are not.**
   - CLIP wins where the answer is only in the pixels (style 1.00, synonyms 1.00).
   - Keyword search over captions wins where the answer is a named detail (detail 1.00 vs CLIP's 0.83; CLIP put "a dog
     under a metal structure" and "a bird eating from a metal bowl" at #2).
   - Caption embeddings are good at what keyword already covers (details) and bad at what CLIP covers (style 0.49), so
     fusing them adds little and can pull good results down.

2. **A caption-only system inherits caption errors.** One keyframe of "Siamese cat pupils blue eyes" shows the cat
   **from behind**, and the vision model captioned it *"A light brown dog is lying on a gray couch"*. Searching captions
   by meaning then ranks that **cat video #1 for "a dog"**: the embedding model did its job, the caption was wrong. In
   the fused system, keyword and CLIP don't rank it first, so the error is outvoted. That's also why option 4 (replace
   CLIP with caption embeddings) is rejected: it removes the retriever that looks at the pixels.

3. **The model matters.** Embedding captions with CLIP's text encoder (option 3) is clearly worse (MRR 0.799). CLIP is
   trained to match text to *images*, not text to text.

4. **Cost of option 2**, for a gain within noise:
   - a second model in the embedding service;
   - a schema migration (new column);
   - index v3 plus a blue/green migration;
   - backfilling every caption;
   - an extra embedding call and kNN search on **every** query.

## Decision

Keep **keyword + CLIP** hybrid search. Captions stay: they feed keyword search (the "detail" wins above) and they're
the text Week 5's LLM reads, but they are **not** embedded.

## Consequences

- No schema, index or model changes.
- The evaluation set now has categorised hard queries, and `make eval` reports MRR per category, so future changes
  are judged per weakness, not only by one average.
- **Negation is unsolved by every retriever** ("a beach with no people" returns the beach *with* people first). That's
  a query-understanding problem for Week 5's LLM, not a retrieval one.

## Limitations of this evidence

- **Small:** 36 queries over 23 videos. The overall averages are informative; categories with 1–3 queries (negation,
  colour, style) are suggestive, not conclusive.
- **Counting went untested:** all four count queries were easy for every system, so they didn't test counting.
- **Labels by one person.** The keyframes were checked visually, but relevance is still a judgement.

## When to revisit

Re-run `make experiment-captions` if any of these change:
- **the library grows substantially**, especially with more speech-free footage where captions are the only text;
- **the caption model or prompt changes**, e.g. captions that describe camera and style;
- **the evaluation set gains** real counting/relation queries that current systems fail;
- **a stronger text-embedding model** is available at similar cost.

The decision should flip if caption embeddings improve **hard-query MRR** without hurting the style category.
