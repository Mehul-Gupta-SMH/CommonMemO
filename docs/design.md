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
| How agents read memory | **Hybrid.** A small, bounded part is *pushed* every turn; agents *pull* the rest through memory tools |
| Can agents write memory | **Yes**, through memory tools; every agent write is recorded with `source="agent"`, so it can be told apart, reviewed and revoked |
| Structured kinds (tool, world, feedback, negatives, resume) | **Owned by CommonMemo**, as projections of the event log — exact by nature, so not delegated to an engine |

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
- `forget_user` also clears the user's feedback, negatives, tombstones,
  checkpoints and steps. Tool memory (shapes and scrubbed signatures only)
  and world facts are not about the user and stay.
- Agent writes are marked `source="agent"`; memory is rendered as data, not
  instructions (see "How agents use memory").

## Memory kinds: the second axis

The tiers say *who* a memory belongs to and *how long* it lives. Kinds say
*what it is about*. Kinds cut across tiers and need their own scopes —
tool memory, for one, is shared across users, which user facts never are.

| Kind | Example | Scope | Built from | Reaches the agent by |
|---|---|---|---|---|
| User facts | "prefers email" | user | `UserFact`, engine extraction | push (stable) + `memory_recall` |
| Episodes | "on 10/14 we debugged the charger" | user, agent | session close | push above a relevance threshold (turn) + `memory_recall` |
| **World facts** | "refund window is 30 days" | `global`, `project:<name>` | `WorldFact` | push (stable) + `memory_recall` |
| **Tool memory** | "`search_flights` fails on DD/MM dates; ISO works" | tool × agent, **across users** | `ToolInvoked` (incl. CommonADK `tool_call`) | the tool's own description |
| **Feedback** | ratings, comments, corrections | user × agent | `Feedback` | lessons, push (stable) |
| **Negatives** | "never suggest phone calls"; tombstones ("forget my old address") | user, agent, project, global | `Negative`, `Forgotten` | push first (stable), never ranked out; tombstones filter facts |
| **Resume points** | last checkpoint, completed and interrupted steps | session | `Checkpoint`, `@mem.step` | push when resuming (turn); steps replay |

Deliberately **not** a kind: raw model reasoning. It is not available
uniformly across providers (summaries, encrypted reasoning, or none),
it restates untrusted input, and it is not always faithful to what the model
did. What is worth keeping from it — decisions and their outcomes — belongs
in lessons and episodes.

## How agents use memory (hybrid)

| Mode | What | Where |
|---|---|---|
| **Push, stable** | negatives, user facts, world facts, lessons from feedback | the agent's instructions — changes rarely, so it keeps prompt caching intact |
| **Push, turn** | episodes above `min_episode_score`, session state, where a resumed process stands | in front of the turn's prompt |
| **In the tools** | tool memory's notes | each tool's own description, read exactly when the model decides how to call it |
| **Pull** | `memory_recall(query)` — every kind, wider limits, no threshold | a tool call |
| **Write** | `memory_remember`, `memory_note_fact`, `memory_avoid`, `memory_save_progress` | tool calls, recorded `source="agent"` |

`MemoryPolicy` sets the push: per-kind limits, the episode threshold, and
`recent` (events of this session to push — **0 by default**, because every
SDK with a session already resends the conversation, and pushing it again
doubles it).

**Memory is framed as data, not instructions.** It holds text that came from
users and tools; replayed with instruction authority, a stored "ignore
previous instructions" would act on every later turn. Rendered memory is
headed as reference data that yields to the user's current request.
Negatives are the one part phrased as standing constraints — they are the
user's own "never do X".

The memory tools are **synchronous** plain typed functions: not every
framework awaits an async tool (CommonADK's Claude adapter calls tool
functions directly). `memory_recall` runs the engine's async search on a
worker thread with its own event loop, and falls back to the exact local
index if an engine's client is bound to the caller's loop.

## Structured kinds

Each structured kind is an event type in the log plus a projection table
(`knowledge.py`) kept up to date as events are recorded and rebuildable from
the log at any time (`Memory.rebuild_knowledge`). A duplicate delivery of
the same event (same id) is ignored by the log *and* by the projections —
`Store.append` reports whether it inserted — so nothing is counted twice.

### Tool memory

Per `(agent, tool)`: calls, failures, durations; error **signatures**
(normalized, so "order 4411 not found" and "order 9 not found" are one
failure); and **fixes** — a failed call followed in the same session by a
successful call of the same tool records the change in argument *shape*
(`date: dmy_or_mdy_date -> iso_date`) as the fix. `notes()` turns that into
advice only once there is evidence (a fix, a failure seen twice, or a
failure rate of 20% over three or more calls).

Tool memory is shared across users, so it holds **no user data**: arguments
are stored as shapes (`iso_date`, `email`, `int`, `text`...), never values,
and error texts have emails, URLs, ids, numbers and quoted values replaced
before they are grouped. `forget_user` leaves it in place for that reason.

### World memory

Facts about the world or a project, not a user: `scope` is `global` or
`project:<name>` (`Memory(project=...)`). A new value for `(scope, key)`
supersedes the old one with history kept; `valid_until` (a TTL) marks facts
that go stale — a price, a status — which stop being pushed once past it.

### Feedback

Ratings in [-1, 1], comments, and corrections (`wrong` → `right`).
`feedback_summary` gives counts and approval; *lessons* — corrections, and
anything negative or commented — are pushed so the agent applies them next
time. Feedback also lands in the session's episode text, so engines learn
from it.

### Negatives and tombstones

A negative is a standing "never/avoid" rule, scoped to a user, an agent, a
project or everyone. Negatives are pushed first and never ranked out, and
can be revoked by id. A **tombstone** (`session.forget(key, pattern=...)`)
removes a user fact and keeps it from coming back: an engine re-extracting
the same thing from old conversations is filtered against tombstones. It
hides only what came before it, so a fact the user states again afterwards
shows again. Tombstones and step/checkpoint events are left out of what
engines extract from, so a forgotten fact is not re-learned from its own
tombstone.

### Resume points: restarting a stopped process

Memory that lets a process continue from where it stopped, with the details
it had at the time:

- `session.checkpoint(label, data)` records what is needed to carry on;
  `session.resume_point()` returns the latest checkpoint, the steps that
  completed / were interrupted / failed, and every event since the
  checkpoint.
- `@mem.step` makes a function's completed calls replayable: re-running the
  same code in a resumed session (same `session_id`, any process) returns
  each finished step's stored result instead of running it again, so the
  process picks up at the first step that did not finish — it "starts as
  is".
- A step that **started but never finished** is *in doubt*. A pure step
  simply runs again; a step declared `side_effects=True` (a payment, an
  email) raises `StepInDoubt`, because running it twice could do the thing
  twice. The application checks the external system and settles it with
  `resolve_step`.
- Step identity is the name plus the arguments (hashed), or `key=`. Results
  must be JSON-serializable to replay; others re-run rather than replay
  wrongly.
- Agents take part too: `memory_save_progress` records a checkpoint, and a
  resumed session's push says where the work stands.

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
    s.learn("refund_window", "30 days")          # world fact (ttl= for stale-able ones)
    s.feedback(-1, "too long"); s.correct("Pune", "Bangalore")
    s.never("suggest phone calls"); s.forget("old_address")
    s.checkpoint("drafted", {"draft_id": 7})     # restart point
    tools = mem.agent_tools(s)                   # memory_recall / _remember / _note_fact / _avoid / _save_progress

@mem.step(side_effects=True)       # resumable: replayed when already done
def send_email(to: str) -> str: ...
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
| `mem0` | shipped | LLM-extracted user memories; see "The Mem0 engine" below |

A planned engine raises `NotImplementedError` naming it; an unknown name
raises `ValueError`. `Engine.bind(store)` lets an engine keep local state in
the store's SQLite file.

### The Graphiti engine

| CommonMemo | Graphiti |
|---|---|
| one closed session | one episode; the uuid Graphiti assigns is recorded in the local index |
| a user | one `group_id` (graph partition); ids outside `[A-Za-z0-9_-]` are sanitized and hash-suffixed |
| episodic retrieval | hybrid `search` → facts → the episodes they came from, best first, with the matching facts appended |
| extracted user facts | the group's currently valid entity edges |
| `forget_user` | delete the group |

What stays exact, in a local index next to the store: the session digest,
explicit `UserFact`s and provenance. Re-ingesting a session removes its graph
episode (by the recorded uuid) first, so nothing duplicates. The engine never
passes `uuid=` to `add_episode`: in Graphiti that names an *existing* episode
to reprocess, and raises `NodeNotFoundError` for a new one. (The first
version did pass one; only the live run below caught it, which is why the
backend tests now bind every call against graphiti-core's real signatures.) Framework bookkeeping
(`ExternalEvent`s such as token counts) stays out of the episode text; it is
noise for extraction and remains in the event tier. Graphiti is reached
through a small `GraphBackend` interface, so the engine's logic is tested
without a graph database or an LLM.

Packaging note: graphiti-core 0.31 imports `httpx` without declaring it
(openai 3.x moved to `httpx2`), so the `graphiti` extra declares `httpx`.

Backend notes: on Neo4j and Kuzu a `group_id` is a property inside one
database (`GraphDriver.clone` returns the same driver); on FalkorDB each
group is its own graph. graphiti-core 0.31.1's `KuzuDriver` never sets
`_database`, so `add_episode` with a `group_id` raises `AttributeError` on
Kuzu; `scripts/live_engines.py` shows the one-line workaround. Kuzu is
deprecated upstream, so production use should be Neo4j or FalkorDB.

### Live runs

`scripts/live_engines.py` runs one scenario per engine against real
backends and an LLM (a user states a city and a preference, then moves; a
later session must see the new city and retrieve the move; `forget_user`
must leave nothing). It is not part of the test suite because it spends
model tokens. It needs `OPENAI_API_KEY` and network access to
`api.openai.com`; Graphiti uses Neo4j when `NEO4J_*` is set, else embedded Kuzu.

### The Mem0 engine

| CommonMemo | Mem0 |
|---|---|
| one closed session | one `add(messages, user_id, metadata={commonmemo_session_id})` |
| a user | Mem0 `user_id`; ids with whitespace are sanitized and hash-suffixed |
| extracted user facts | `get_all(filters={"user_id"})`, as `Fact(key="memory")` after the explicit ones |
| episodic retrieval | `search` → memories → the sessions that created them |
| re-ingest | delete the memories tagged with the session, then add again |
| `forget_user` | `delete_all(user_id=...)` |

Same local index as the Graphiti engine for the exact parts. No `run_id` is
passed, deliberately: Mem0 deduplicates within a scope, and per-session
scoping would make it re-learn the same preference every session instead of
updating one memory. Because Mem0 updates memories in place, a refined
memory's provenance points at the session that first created it. Anonymous
sessions (no `user_id`) stay local: Mem0 memory is per user. Both graph and
memory engines share `rank_by_hits` (`engines/_ranking.py`).

## CommonADK integration

`commonmemo.integrations.commonadk` registers one catch-all hook on
CommonADK's `HookRegistry`, so all six CommonADK targets get the same memory:

- every CommonADK event is kept verbatim as `ExternalEvent(source="commonadk.<type>")`
  (token counts, cost, transfers, errors — nothing dropped);
- `run_started.prompt` and `run_finished.final_text` also become `Message`s;
- `tool_call` becomes `ToolInvoked`, so CommonADK tool calls feed tool memory
  exactly like `@mem.track`-ed calls;
- ids derive from CommonADK's `run_id` + `seq`, so a double fire records once.

`run_with_memory(session, runner, project, agent, prompt)` does a whole
hybrid turn and returns CommonADK's `Trace` unchanged:

1. **push** — stable memory to CommonADK's `context=` (the agent's
   instructions), per-turn memory to `turn_context=` (in front of the prompt);
2. **pull** — `prepare_project` adds the memory tools to the agent and
   writes tool memory's notes into the agent's own tools' descriptions
   (both the `ToolSpec.description` the Claude adapter reads and the
   function docstring the other five read), on a copy of the project;
3. **capture** — the hook records the run.

Verified by building a real agent on all six targets from a prepared project
and checking that the note reaches each SDK's tool description.

A CommonADK release without `context=` / `turn_context=` (before CommonADK
PR #29) gets all of the memory in front of the prompt instead.

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

1. **v0.0.1 (shipped):** event and session tiers, `@track`, `MemoryEvent`,
   local engine, CommonADK integration, tests.
2. **Graphiti engine** and **Mem0 engine** (both shipped); a live run of
   each against a real backend and LLM.
3. **CommonADK PR #29:** `context=` / `turn_context=` and `RunSession.user_id`.
4. **Structured kinds** (this): tool memory, world memory, feedback,
   negatives and tombstones, resume points; hybrid push/pull with agent
   tools.
5. **Framework adapters:** ADK `BaseMemoryService`, OpenAI `Session`,
   LangGraph `BaseStore`, CrewAI `StorageBackend`, AutoGen `Memory`, Claude
   `SessionStore`.
6. **Harness adapters:** MCP server; Claude Code plugin, Codex hooks,
   OpenCode plugin — one session continuing across harnesses. The memory
   tools become the MCP server's tools.
7. **Later kinds:** procedural memory and lessons distilled by an LLM at
   consolidation (decision records: goal, plan, outcome), tasks / open loops
   for proactive follow-ups, and agent-relationship memory from CommonADK
   `Transfer` events.
