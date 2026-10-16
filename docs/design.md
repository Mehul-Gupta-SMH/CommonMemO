# CommonMemo — Design

## Goal

One memory system that any agent framework or harness can use: an
application installs one package, marks its events with a decorator or a
subclass, and gets four tiers of memory without wiring any of them by hand.

First target: **CommonADK**, which already normalizes six agent SDKs into one
event stream. Later targets: the frameworks' own memory interfaces (Google
ADK, OpenAI Agents, LangGraph, CrewAI, AutoGen, Claude Agent SDK) and coding
harnesses (Claude Code, Codex, OpenCode).

## Decisions (settled)

| Decision | Choice |
|---|---|
| Build or wrap the memory engine | **Wrap.** Extraction, retrieval, contradiction handling and their optimizations belong to dedicated engines; CommonMemo owns the tier model, the event log and the integrations |
| Language | **Python only** |
| First integration | **CommonADK**, via its observe-only `HookRegistry` |
| Developer surface | **Install + `@mem.track` decorator + `MemoryEvent` subclass**; everything else automatic |
| Core dependencies | **None** (stdlib `sqlite3` + `dataclasses`); engines and integrations are extras, imported lazily |
| Default storage | **SQLite** — one path is the whole setup |

## The four tiers

The tiers differ in **scope** (who owns the memory), **lifetime** and **how
it is written**:

| Tier | Holds | Scope | Lifetime | Written by | Read by | Owner |
|---|---|---|---|---|---|---|
| Event | immutable records: messages, tool calls, framework events, domain events | session / user | retention policy (`prune`) | automatic capture | consolidation, replay, debugging | CommonMemo `Store` |
| Session | working state of one conversation or run | `session_id` | until closed or expired (TTL); resumable | the application, live | every turn (`Session.context`) | CommonMemo `Store` |
| Episodic | one consolidated summary per session | user / agent | long-term | consolidation at session close | retrieval by relevance | wrapped `Engine` |
| User | durable facts about a user | `user_id`, across sessions and agents | long-term; superseded, never silently lost | explicit `UserFact` events + engine extraction | every turn | wrapped `Engine` |

**Data flow:** events → session (live) → on session close, consolidation →
episodes and user facts.

### Why this split of ownership

The event and session tiers are simple, hot (touched on every turn) and must
never be lost, so they live in a store CommonMemo controls; routing them
through an engine would add latency and couple the source of truth to an
engine's data model. The episodic and user tiers are where the hard,
optimizable work is (summarization, extraction, retrieval, conflict
resolution) — exactly what engines specialize in, so they are delegated.

### Rules every engine keeps

1. **Provenance.** Every episode and fact carries `source_event_ids`, so any
   memory can be traced back to what actually happened.
2. **Explicit facts win.** A `UserFact` event is stored as-is; the latest
   value per key supersedes the previous one, whose history is kept
   (the "invalidate, don't discard" rule Graphiti uses). LLM-extracted facts
   are added alongside explicit ones, never in place of them.
3. **Idempotent ingest.** Re-consolidating a session (after it is resumed
   and closed again) replaces what the previous ingest produced.

### Privacy and safety

- `Memory.forget_user(user_id)` deletes a user from every tier, including
  events recorded inside their sessions that did not carry a `user_id`.
- A session belonging to one user can never be resumed as another
  (`PermissionError`) — that would leak working memory across users.
- Recording is an observer, never a gate: a failure to record never fails
  the tracked call or the agent run.

## Developer surface

```python
mem = Memory("sqlite:///mem.db", engine="local")

@mem.track                         # sync or async; bare or configured
def tool(...): ...

class OrderPlaced(MemoryEvent):    # becomes a frozen kw-only dataclass, kind "order_placed"
    order_id: str

async with mem.session(user_id="u1", agent="a") as s:   # or plain `with` outside an event loop
    ctx = await s.context(query)   # MemoryContext: facts, episodes, state, recent events
    s.record(OrderPlaced(order_id="A1"))
    s.message("user", "...")
    s.state["k"] = v               # persisted on write
    s.remember("key", "value")     # explicit user fact
```

`@mem.track` records into the **current session** (a context variable set by
entering a session), so tools need no session argument.

Built-in event types: `Message`, `ToolInvoked`, `UserFact`, `ExternalEvent`
(events from another system, kept verbatim). An unknown `kind` read back from
the log becomes an `ExternalEvent` rather than an error, so a stored log is
always readable.

## Engines

`Engine` is four async methods: `ingest(session, events)`,
`search_episodes(query, user_id, agent, limit)`, `user_facts(user_id)`,
`forget_user(user_id)`.

| Engine | Status | Notes |
|---|---|---|
| `local` | shipped | offline and deterministic: rule-based episode digests, keyword retrieval, explicit facts only. For zero-setup use and reproducible tests |
| `graphiti` | shipped | temporal knowledge graph; see "The Graphiti engine" below |
| `mem0` | planned | fact extraction and retrieval for the user tier |

A planned engine raises `NotImplementedError` naming it; an unknown name
raises `ValueError`. `Engine.bind(store)` lets an engine keep local state in
the store's SQLite file.

### The Graphiti engine

| CommonMemo | Graphiti |
|---|---|
| one closed session | one episode, uuid derived from the session id (`uuid5`) |
| a user | one `group_id` (graph partition); ids outside `[A-Za-z0-9_-]` are sanitized and hash-suffixed |
| episodic retrieval | hybrid `search` → facts → the episodes they came from, best first, with the matching facts appended |
| extracted user facts | the group's currently valid entity edges |
| `forget_user` | delete the group |

What stays exact, in a local index next to the store: the session digest,
explicit `UserFact`s and provenance. Re-ingesting a session removes its graph
episode first, so nothing duplicates. Framework bookkeeping
(`ExternalEvent`s such as token counts) stays out of the episode text; it is
noise for extraction and remains in the event tier. Graphiti is reached
through a small `GraphBackend` interface, so the engine's logic is tested
without a graph database or an LLM.

Packaging note: graphiti-core 0.31 imports `httpx` without declaring it
(openai 3.x moved to `httpx2`), so the `graphiti` extra declares `httpx`.

## CommonADK integration

`commonmemo.integrations.commonadk` registers one catch-all hook on
CommonADK's `HookRegistry`, so all six CommonADK targets get the same memory:

- every CommonADK event is kept verbatim as `ExternalEvent(source="commonadk.<type>")`
  (token counts, cost, transfers, errors — nothing dropped);
- `run_started.prompt` and `run_finished.final_text` also become `Message`s;
- `tool_call` becomes `ToolInvoked`, so CommonADK tool calls and
  `@mem.track`-ed calls read the same downstream;
- ids derive from CommonADK's `run_id` + `seq`, so a double fire records once.

`run_with_memory(session, runner, project, agent, prompt)` does a whole turn:
fetch context, run with the hook attached, return CommonADK's `Trace`
unchanged.

**Gaps on the CommonADK side** (to be proposed as a CommonADK PR):

1. **Pre-run context.** CommonADK hooks are observe-only, so memory cannot be
   added to an agent's *instructions*. Until a pre-run context step exists,
   `run_with_memory` puts the rendered context in front of the prompt.
2. **User identity.** `RunStarted` / `RunSession` carry no `user_id`; today
   the user comes from the CommonMemo session. An optional field would let
   CommonADK traces carry it natively.

## Landscape (October 2026)

What exists, and what CommonMemo takes from it:

- **TencentDB Agent Memory** (MIT, Node): L0 raw → L1 facts → L2 scenes → L3
  persona; integrates mainly through an LLM proxy; team memory hub in v2.
  Its tiers are levels of *processing*; CommonMemo's are *scopes and
  lifetimes*, with a resumable session tier it has no equivalent of.
- **Engines:** Mem0 (extraction, personalization), Zep/Graphiti (temporal
  knowledge graph), Letta (self-editing memory blocks), LangMem (on
  LangGraph's store), Cognee (graph from documents), Hindsight, MemOS.
- **Framework-native interfaces** (verified against installed source):
  Google ADK `BaseMemoryService` + `SessionService`; OpenAI Agents `Session`
  (SQLite, Redis, SQLAlchemy, MongoDB, Dapr, encrypted, compaction); Claude
  Agent SDK `SessionStore` + per-agent `memory`; LangGraph `BaseStore` +
  checkpointer; CrewAI unified `Memory` + `StorageBackend`; AutoGen `Memory`.
  Every framework has a swappable interface; none shares memory with any
  other. CommonMemo's framework adapters implement those interfaces over one
  store.
- **Coding harnesses:** Claude Code (hooks, MCP, plugins), Codex (AGENTS.md,
  MCP, hooks), OpenCode (plugin events, MCP). Lesson: MCP alone is not
  enough — event capture must not depend on the model calling a tool, so each
  harness needs a small hook adapter.

## Roadmap

1. **v0.0.1 (this):** event and session tiers, `@track`, `MemoryEvent`,
   local engine, CommonADK integration, tests.
2. **Graphiti engine** (shipped), then **Mem0 engine**.
3. **CommonADK PR:** pre-run context step and optional `user_id`.
4. **Framework adapters:** ADK `BaseMemoryService`, OpenAI `Session`,
   LangGraph `BaseStore`, CrewAI `StorageBackend`, AutoGen `Memory`, Claude
   `SessionStore`.
5. **Harness adapters:** MCP server; Claude Code plugin, Codex hooks,
   OpenCode plugin — one session continuing across harnesses.
