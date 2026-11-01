# ADR 0005 — A very small local embedder for vector search

**Status:** Proposed · **Date:** 2026-11-01 · **Tracks:** [#125](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/125) (the bake-off) · **Unblocks, if accepted:** [#6](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/6) (CrewAI)

This is a draft. It records the question, the options and the bar a
decision must clear. The numbers come from the bake-off in #125; the
decision (section 4) is filled in after it, and this ADR is then
accepted, changed, or rejected with the numbers recorded.

## Context

What exists:

- `Embedder` (a function from texts to vectors), `EmbeddingRanker`
  (`ranking.py`), persisted window vectors (`window_vectors`, #77) and a
  probe text that notices a changed embedder.
- A memory uses a ranker you pass, else the engine's embedder (Graphiti,
  Mem0), else keyword ranking with stemming.

What does not exist:

- **A default embedder.** Without Graphiti or Mem0, or one you write, a
  memory ranks by shared words.
- **A measurement.** Keyword windows beat naive retrieval on LoCoMo (140 vs
  109, #74). Whether any embedder beats keyword windows is unknown: model
  hosts were unreachable when #66, #67 and #77 were built.
- **A CrewAI adapter** (#6, deferred): its `StorageBackend.search` receives
  an embedding vector. An adapter without our own embedder would be a thin
  vector store that bypasses consolidation and ranking.

The question is whether a very small model, running on CPU inside the
user's process with no model API, gives enough better recall to be worth a
dependency and a model download.

## Options

| | A. Keyword only (today) | B. Static embeddings (Model2Vec "potion" class) | C. MiniLM-class, int8 ONNX | D. bge-small / e5-small class |
|---|---|---|---|---|
| Size on disk | 0 | about 8 to 30 MB | about 25 MB | about 35 to 130 MB |
| Runtime dependency | none | numpy (already optional) | `onnxruntime` plus a tokenizer | `onnxruntime` plus a tokenizer |
| Speed (CPU, per text) | n/a | under 1 ms | a few to about 15 ms | about 10 to 40 ms |
| Reads word order | no | no | yes | yes |
| Expected quality | baseline | between keywords and C | the likely sweet spot | best of the small ones |

Sizes and speeds are from memory of the model cards and are to be measured
in the bake-off, not trusted. `torch` and `sentence-transformers` are
rejected up front: hundreds of megabytes for a library that installs in
under a megabyte today.

Hybrid ranking (keyword score plus vector score) is a ranker too and is
included in each of B to D.

## Decision criteria

1. **Measured gain, not plausibility.** On all ten LoCoMo conversations
   (450 questions), an embedder is a default only if it adds at least 5
   points of evidence recall at 8 windows and 11 answers over keyword
   ranking, the bar used in #66 and #71. A smaller gain makes it opt-in.
   No gain, and it is not built.
2. **No weight in the base install.** Any embedder is the `commonmemo[embed]`
   extra, imported lazily; `pip install commonmemo` stays dependency-free.
3. **Cost.** Install size, cold start, ms per text, ms per search with
   persisted vectors; the 50 ms search with numpy (#77) must not get worse.
4. **Reproducible and safe to fetch.** The model name, revision and checksum
   are pinned; a download that does not match fails; offline use fails with
   a message that says how to get the weights, never silently falls back.
5. **License** allows redistribution of a download link (the weights are not
   vendored into the wheel).

## Decision

*Left blank until the bake-off reports (#125).* The outcomes it can lead to:

- **Ship as default:** `commonmemo[embed]` provides the chosen model;
  `Memory` uses it when installed and no other embedder is set. Vectors for
  facts, recaps and procedures follow in their own issue; CrewAI is
  revisited.
- **Ship opt-in:** the extra exists and is documented, but a memory without
  an explicit choice still ranks by keywords.
- **Do not build:** the numbers are recorded in `docs/live-runs.md` and the
  roadmap, and the "real vector index" and CrewAI items stay as they are.

## Risks

- **English first.** The small models named here are mostly English; a
  user writing in another language may do worse than with keywords. The
  bake-off is English (LoCoMo), so that gap is documented, not measured.
- **A model download at first use** is a network call from a library that
  makes none today. It happens only when the extra is installed and asked
  for, never on import.
- **A changed model** invalidates stored vectors. #77's probe already drops
  them; a pinned revision makes it a deliberate upgrade.
- **Extraction stays off by default;** this ADR does not change it.

## Consequences

- If accepted as default or opt-in, CrewAI's objection (#6) is gone for the
  search path; the adapter still has to keep writes going through the Log.
- Windows keep ids, not text, so a vector is always made from redacted text
  and removed on `forget_user` and `prune`, as in #77.
- A real vector index (sqlite-vec, pgvector) stays a separate question,
  worth asking once a user passes about 100,000 messages.
