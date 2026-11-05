# Related work: open-source agent memory projects

*Written 2026-11-05 for [#148](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/148). Research only: no code changed.*

## How this was made

Every claim below was read from the project's own repository at the commit
shown (shallow clone, files read directly). Links point at that commit. Three
limits to keep in mind:

- **Source and README only.** The projects' documentation sites were not
  reachable from the build environment, so behaviour that is documented only
  there is not covered.
- **Nothing was run.** No project was installed or benchmarked; statements
  about what code does come from reading it.
- **Star counts are left out.** The earlier blog-based figures disagreed with
  each other and cannot be cited from a repository checkout.

| Project | Commit read | Version | License |
|---|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0/tree/ed1f10fc87080f0b67f99be370817504888462ab) | `ed1f10f`, 2026-11-04 | 2.2.1 (`pyproject.toml`) | Apache-2.0 |
| [Graphiti](https://github.com/getzep/graphiti/tree/fb60e59d4a208c99e7d0186ee6335d2a257f17f2) (Zep) | `fb60e59`, 2026-11-03 | 0.31.2 | Apache-2.0 |
| [Cognee](https://github.com/topoteretes/cognee/tree/525b1d1179c550cd5b5661bcbf8f00f57a676928) | `525b1d1`, 2026-10-29 | 1.6.6 | Apache-2.0 |
| [LangMem](https://github.com/langchain-ai/langmem/tree/8620b986af5b0c7593e6b7fbece2998690d22d15) | `8620b98`, 2026-11-03 | 0.0.30 | MIT |
| [Letta](https://github.com/letta-ai/letta/tree/56ba9c25552605eec89de8ed3dc6394b625c1993) | `56ba9c2`, 2026-08-13 (`archive` branch) | see note | Apache-2.0 |

Letta's `main` branch is now a pointer: its README says current source lives in
[`letta-ai/letta-code`](https://github.com/letta-ai/letta-code) and the retired
V1 server is on the `archive` branch. The Letta rows describe that archived
server; the current product was not read. Supermemory, MemOS and MemPalace
(named in #148) were **not** read and are not covered.

## Per project

### Mem0

- **What is stored.** Short memory strings with an id, hash, timestamps and a
  lemmatised copy for keyword matching, written into a pluggable *vector
  store* (`mem0/memory/main.py`, `_create_memory`). 27 backends sit under
  `mem0/vector_stores/` (Qdrant, pgvector, Redis, Pinecone and others). A
  separate SQLite `history` table records each change (`memory_id`,
  `old_memory`, `new_memory`, `event`, `actor_id`, `role`; `mem0/memory/storage.py`).
- **How it is written.** `add(..., infer=True)` has a model extract facts from
  the messages; `infer=False` stores the text as given. The README says the
  April 2026 algorithm is "single-pass ADD-only extraction ... no UPDATE/DELETE.
  Memories accumulate; nothing is overwritten", with entity linking and
  semantic + BM25 + entity retrieval fused.
- **Scopes.** `user_id`, `agent_id` and `run_id` are filters on every call
  (`main.py`, session-scoping helper). There is no org or project level in the
  library; a self-hosted server and cloud platform add auth and a dashboard.
- **Deletion.** `delete(memory_id)`, and `delete_all(user_id=|agent_id=|run_id=)`,
  which requires at least one filter and deletes in batches of 1,000.
  `history(memory_id)` shows changes.
- **Telemetry.** On by default (`MEM0_TELEMETRY` defaults to `True`,
  `mem0/memory/telemetry.py`).
- **Claims.** The README table reports LoCoMo 92.5, LongMemEval 94.4 and BEAM
  64.1 (1M) for the new algorithm, and says the scores "reflect Mem0's managed
  platform, which includes proprietary optimizations not available in the
  open-source SDK". It links an evaluation framework
  ([`mem0ai/memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks),
  not read). These are vendor-reported on their own platform.

### Graphiti (Zep)

- **What is stored.** A temporal graph in a graph database: entities (nodes),
  facts (edges with validity windows) and *episodes*, the raw inputs. "Every
  derived fact traces back" to an episode (README). Drivers under
  `graphiti_core/driver/`: Neo4j, FalkorDB, Kuzu, Neptune.
- **Supersession.** "When information changes, old facts are invalidated, not
  deleted", so you can ask what was true at a point in time (README).
- **Scopes.** `group_id` partitions the graph. The reference REST and MCP
  servers "do not authenticate callers. Any caller can read, change, or delete
  data in any `group_id`" and have "no prompt-injection protection"; the README
  calls these omissions intentional and out of scope.
- **Deletion.** `remove_episode(uuid)` deletes the episode, edges only it
  supported and nodes only it mentioned (`graphiti.py`). This is a real
  delete, not a retraction.
- **Cost model.** Ingestion is model-driven (entity and edge extraction per
  episode); the README tunes it with `SEMAPHORE_LIMIT` for provider rate limits.
- **Telemetry.** Anonymous usage statistics at start-up, documented in the
  README with a disable switch.
- **Claims.** The README's comparison with GraphRAG is qualitative; it cites an
  arXiv paper (2501.13956, not read) and gives no benchmark table.
- **Ours.** CommonMemo already has Graphiti as an optional engine (#2); #3
  (a live run) is blocked on a model API key.

### Cognee

- **What is stored.** Documents, code and conversations turned into a
  knowledge graph plus vector entries, grouped in *datasets*. API verbs in the
  README: `remember`, `recall`, `forget`, `improve`, with an older
  `add` / `cognify` / `search` pipeline under `cognee/api/v1/`.
- **Local mode.** The README says it runs with no LLM key, using a local GLiNER
  extractor and local embeddings, and that this extractor "is a demo ... for a
  production-ready version ... reach out to us". Retrieval then returns source
  text rather than a generated answer.
- **Documents.** The closest match to our org scope: ingestion modules
  discover directory trees as datasets (`modules/ingestion/discover_directory_datasets.py`)
  and the README pitches a shared "Company Brain" over documentation, tickets
  and code.
- **Access.** Multi-tenant by default: with `ENABLE_BACKEND_ACCESS_CONTROL`
  unset, every API call needs an authenticated user; the README sets it to
  `false` for the single-user local posture. A permissions router exists
  (`api/v1/permissions`).
- **Deletion.** `forget` (`api/v1/forget/forget.py`) removes one item, a
  dataset, or everything the user owns; `memory_only=True` drops the graph and
  embeddings but keeps raw files so a dataset can be re-processed. The
  docstring notes an empty search afterwards "does not mean the data is gone".
- **Telemetry.** A `TELEMETRY_DISABLED` environment variable exists
  (`cognee/shared/utils.py`); the default was not verified.
- **Claims.** BEAM: 0.79 at 100K tokens, "four evaluation rounds over 20
  questions from one held-out conversation", with "benchmark-specific data
  formatting, prompts, and retrieval configuration" (README). The authors'
  own caveats are in the table; the sample is small.

### LangMem

- **What is stored.** Whatever the host's LangGraph `BaseStore` holds, under a
  *namespace* tuple such as `("memories", "{user_id}")`, filled in at run time
  from the config (`src/langmem/knowledge/tools.py`). LangMem is a library of
  tools and managers, not a store.
- **How it is written.** Two modes: tools the agent calls "in the hot path"
  (`create_manage_memory_tool`, `create_search_memory_tool`) and a background
  manager that extracts, consolidates and updates. The README: "the agent gets
  to decide what and when to store".
- **Deletion.** The memory manager takes `enable_deletes` (default `False`;
  `knowledge/extraction.py`), so removal is something the model is allowed to
  do only when switched on.
- **Claims.** None in the README. (CommonMemo ships a LangGraph `BaseStore`
  adapter, #111, so the two can sit side by side.)

### Letta (archived server, V1)

- **What is stored.** *Blocks* (labelled, character-limited text such as
  `human` and `persona` that sit in the context window, with `read_only` and a
  `limit`; `letta/schemas/block.py`), message history, and *passages* (text
  chunks with embeddings and tags; `SourcePassage` records the `file_name` it
  came from; `letta/orm/passage.py`). A `source` table and per-file passages
  are the document path.
- **Edits are reversible.** `block_history` stores each prior state of a block
  "for undo/redo functionality" (`letta/orm/block_history.py`).
- **Scopes.** Rows carry an organisation (`OrganizationMixin`); there are
  identities, groups and shared blocks (`groups_blocks`, `identities_blocks`).
- **Agent-managed.** The model edits its own blocks (the MemGPT idea); no
  per-write human approval was found in the files read.
- **Claims.** No benchmark table in the README.

## What this means for CommonMemo

What the comparison supports, as observed facts:

1. **Append-only history with real retraction is uncommon.** Mem0 keeps a
   history table but deletes the vector row; Graphiti's `remove_episode` and
   Cognee's `forget` delete. Letta's `block_history` is the nearest to our
   revertible compensating events, for blocks only.
2. **No project reads as having an approval step for agent-written facts.**
   Letta lets the agent rewrite its blocks; LangMem gates only deletion;
   Mem0's extraction writes straight to the store. Not found is not the same
   as absent: this covers the files read, not their hosted products.
3. **Org-wide scope exists as organisation rows (Letta) or datasets and
   tenants (Cognee), not as a precedence chain.** Nothing read merges or
   ranks user over project over org the way ADR 0006 proposes.
4. **Document memory is a graph or vector index built by a model.** Cognee
   and Letta both ingest documents; Cognee builds a graph with an extractor.
   ADR 0006's passages (text kept in the Log, searched by the existing
   ranker, no model at ingest) are cheaper and differ mainly in having no
   extraction step to trust.
5. **Benchmark claims are not comparable.** Four different benchmarks, two of
   them (Mem0, Cognee) stated by the maintainers on their own harness, one on
   a managed platform with undisclosed optimisations, one on 20 questions.
   Our own LoCoMo numbers (windows 276 vs extraction 265 of 450, Haiku) use a
   different judge and are not comparable either.
6. **Telemetry defaults differ.** Mem0 on by default, Graphiti collects
   anonymous stats at start-up; CommonMemo sends nothing.

## Ideas worth testing

Each has a follow-up issue:

- **Graphiti's validity windows for world facts** ([#149](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/149)): our supersession keeps
  history but a fact has no "true from / until" a reader can query. Candidate
  to compare against `as_of` (#27).
- **Cognee's `memory_only` forget** ([#150](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/150)) (drop derived data, keep the source):
  maps onto `forget_source` for passages in ADR 0006.
- **Mem0's fused semantic + keyword + entity retrieval** ([#152](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/152)): ours is keyword or
  keyword + vector (#125); entity matching is the part #146 tests.
- **Letta's character-limited labelled blocks** ([#151](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/151)) as a pinned, bounded context
  area for org rules, versus our pinned world facts.

## Not covered

Supermemory, MemOS and MemPalace; the current Letta product; every
documentation site; any hosted offering's behaviour; any measured cost or
quality. Re-run the read when the doc hosts are reachable.
