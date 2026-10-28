# CommonMemo — Design

## Goal

One memory system that any agent framework or harness can use: an
application installs one package, marks its events with a decorator or a
subclass, and gets four tiers of memory, plus the structured kinds (tool,
world, feedback, negatives, resume), without wiring any of them by hand.
Agents read it in hybrid mode (push plus pull) and can write it.

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
| Log | immutable records: messages, tool calls, framework events, domain events | session / user | retention policy (`prune`) | automatic capture | consolidation, replay, debugging | CommonMemo `Store` |
| Working | what one conversation or run has in hand | `session_id` | until closed or expired (TTL); resumable | the application, live | every turn (`Session.context`) | CommonMemo `Store` |
| Recaps | one consolidated summary per session | user (all their agents); agent when there is no user | long-term | consolidation at session close | retrieval by relevance | wrapped `Engine` |
| UserProfile | durable facts about one user | `user_id`, across sessions and agents | long-term; superseded, never silently lost | explicit `UserFact` events + engine extraction | every turn | wrapped `Engine` |

**Data flow:** the log → working memory (live) → on session close,
consolidation → a recap and UserProfile facts.

### Naming

Each name says what the tier holds. They replace event, session, episodic and
user, which clashed with the event types (`MemoryEvent`), with the `Session`
handle and every harness's own sessions, and with plain English ("episodic").
"Session" now means only one run. World facts are not part of UserProfile;
they stay a separate kind with their own scopes. The code uses the same names
([#19](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/19)): `Recap`, `MemoryContext.recaps`, `Engine.search_recaps`,
`Engine.user_profile`. Event types keep theirs (`UserFact` is an event, not a
tier), and so do Graphiti's own "episodes".

### Why this split of ownership

The log and working memory are simple, hot (touched on every turn) and must
never be lost, so they live in a store CommonMemo controls; routing them
through an engine would add latency and couple the source of truth to an
engine's data model. Recaps and the UserProfile are where the hard,
optimizable work is (summarization, extraction, retrieval, conflict
resolution) — exactly what engines specialize in, so they are delegated.

### Rules every engine keeps

1. **Provenance.** Every recap and fact carries `source_event_ids`, so any
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
| UserProfile facts | "prefers email" | user | `UserFact`, engine extraction | push (stable) + `memory_recall` |
| Recaps | "on 10/14 we debugged the charger" | user, agent | session close | push above a relevance threshold (turn) + `memory_recall` |
| **World facts** | "refund window is 30 days" | `global`, `project:<name>` | `WorldFact` | push (stable) + `memory_recall` |
| **Tool memory** | "`search_flights` fails on DD/MM dates; ISO works" | tool × agent, **across users** | `ToolInvoked` (incl. CommonADK `tool_call`) | the tool's own description |
| **Feedback** | ratings, comments, corrections | user × agent | `Feedback` | lessons, push (stable) |
| **Negatives** | "never suggest phone calls"; tombstones ("forget my old address") | user, agent, project, global | `Negative`, `Forgotten` | push first (stable), never ranked out; tombstones filter facts |
| **Resume points** | last checkpoint, completed and interrupted steps | session | `Checkpoint`, `@mem.step` | push when resuming (turn); steps replay |
| **Procedures** | the weekly report's template, rules and steps; accepted examples | task × (`global`, `project:<name>`, `user:<id>`) | `ProcedureEntry` (files, apps, agents) | push for the named or matched task (turn) + `memory_get_procedure` |

Deliberately **not** a kind: raw model reasoning. It is not available
uniformly across providers (summaries, encrypted reasoning, or none),
it restates untrusted input, and it is not always faithful to what the model
did. What is worth keeping from it — decisions and their outcomes — belongs
in lessons and recaps.

## How agents use memory (hybrid)

| Mode | What | Where |
|---|---|---|
| **Push, stable** | negatives, user facts, world facts, lessons from feedback | the agent's instructions — changes rarely, so it keeps prompt caching intact |
| **Push, turn** | recaps above `min_recap_score`, working state, where a resumed process stands | in front of the turn's prompt |
| **Push, mid-run** | recaps of earlier sessions a tool result brings up, not pushed yet this turn | after the tool call, before the next model call (`Session.refresh`) |
| **In the tools** | tool memory's notes | each tool's own description, read exactly when the model decides how to call it |
| **Pull** | `memory_recall(query)` — every kind, wider limits, no threshold | a tool call |
| **Write** | `memory_remember`, `memory_note_fact`, `memory_avoid`, `memory_save_progress` | tool calls, recorded `source="agent"` |

"Where" is where each part is *meant* to go. An integration delivers it as
far as its framework allows. Through released CommonADK, for one, all
pushed memory goes in front of the prompt until [CommonADK #30](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/30) lands (see
"CommonADK integration").

The push is bounded, so it leaves things out, and a model can't know
what it wasn't shown. When the push cuts something (facts, world facts,
negatives or lessons past their `max_*`, recaps past the limit that clear
half the threshold, entries cut for length) and the agent has
`memory_recall`, the turn text ends with one line saying so: "More in memory
than shown here: 5 facts about the user. Call memory_recall with what you
need." ([#23](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/23)). Nothing cut, or no recall tool: no line, so it never becomes noise.
It sits inside the reference-data framing, and the character cap never cuts
it. `build_agent_tools` tells the session its agent can recall; hooks need
`--recall-tool`, because they can't see a harness's MCP servers.

**Ranking** ([#21](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/21), `ranking.py`). One `Ranker` decides which profile facts
make the push when there are more than `max_facts`, which world facts a
request is about (with a key registry, or past `max_world`), and what
`memory_recall` returns, so the push and recall agree. `KeywordRanker`, the
default, scores the share of the request's terms in a text (stemmed, stop
words dropped). `EmbeddingRanker(embed)` scores cosine similarity from any
embedder, with embeddings cached; the Graphiti and Mem0 engines supply one
from their own embedder (`Engine.ranker()`). Ranking runs only when there is
more than fits, so a small profile costs no embedding call. Negatives are
never ranked out. Recaps are still ranked by the engine's own search, and
mid-run refresh keeps its shared-words rule (it runs after every tool call
and must stay cheap).

The turn's push is built from the request, so an agent that then runs many
steps used to work from that one snapshot. **Refresh** ([#22](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/22)) fills the gap:
after each tool call, `Session.refresh(result)` pushes the recaps the
result brings up that this turn hasn't pushed yet. A recap qualifies when
it shares at least `refresh_min_overlap` (3) distinct words with the
result; tool output is long and noisy, so the turn's threshold (a share of
the query's words) doesn't fit, and ranking by meaning is [#21](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/21). At most
`max_refreshes` (3) per turn. It searches the local recap index only, so a
refresh never costs a graph or LLM round trip, whatever the engine. Every push is recorded as a `ContextPushed`
event (reason, recap ids, characters), which bounds refreshes across hook
processes and measures what memory costs per run. The stable half never
changes mid-run, so prompt caching keeps working. The Claude Code and Codex
hooks return it from `PostToolUse`; OpenCode adds it to the system prompt
of the next model call; CommonADK needs [#7](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/7); the MCP server can't push.

A user's recaps come from every agent they worked with
(`MemoryPolicy.recaps_across_agents`, on by default, [#34](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/34)): with one agent per
harness, Claude Code sees what the user did in Codex, and a recap from
another agent says whose it was. Without a user, recaps stay per agent.
Tool notes and lessons stay per agent; they describe one agent's tools.

`MemoryPolicy` sets the push: per-kind limits, the recap threshold, and
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
(`knowledge.py`), rebuildable from the log at any time
(`Memory.rebuild_knowledge`). What memory reads from them is the `Projections`
interface (`projections.py`, [#97](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/97)), by operation and not by SQL;
`Knowledge` implements it, and `Memory(projections=...)` takes another.

The projections are a **consumer of the log**
([#15](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/15)). They store the last log position they applied, and
`Knowledge.catch_up` applies everything after it, writing the new position
in the same transaction as the changes. Catch-up runs after every write and
whenever a `Memory` is opened. So:

- a crash between logging an event and applying it loses nothing: the next
  write or the next process applies it;
- a duplicate delivery (same event id) is not logged twice, and a stretch of
  log is never applied twice, so nothing is counted twice;
- several handles or processes on one file share one position, and take
  turns applying through the database's write lock.

Positions come from the log in commit order (SQLite has one writer at a
time). A store with concurrent writers, such as Postgres, needs to hand out
positions in commit order too, or keep a small window of in-flight
positions; that is part of [#11](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/11).

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
`retract_fact` ends a fact with nothing replacing it.

A world fact is pushed to every user in its scope, so who may change one
matters ([ADR 0002](adr/0002-world-fact-trust.md)):

- **Agent writes are proposals** until an operator approves them
  (`confirm_fact`, `reject_fact`, `pending_world_facts`). A proposal is shown
  only to the user who caused it, marked as unconfirmed.
  `Memory(world_fact_approval=False)` turns approval off.
- **Source decides before recency.** `extracted` < `agent` < `tool` <
  application code. A less trusted write never supersedes a more trusted
  fact; it is kept as *disputed* for review.
- **The agent sees age and source**: `refund_window: 30 days (as of
  2026-10-14, operator)`.

World facts are a small set of curated operational facts, managed like
configuration, not a document store: prose that needs retrieval belongs in
the app's RAG. An **optional key registry** ([#24](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/24), `facts.py`) says which keys
exist and how they behave: `FactKey(key, description, allowed=, lifetime=,
pinned=)`, given to `Memory(fact_keys=...)`, `register_fact_key`, or a
JSON/YAML file through `load_facts`. Without one, nothing changes. With one:

- **Agents reuse keys.** `memory_note_fact`'s description lists them. An
  agent's write to an unregistered key, a near-duplicate of a registered one,
  or a value outside `allowed` stays a proposal with a note saying why, even
  with approval off; an operator's disallowed value is a `ValueError`.
- **Lifetimes.** A registered key's `lifetime` sets `valid_until` when the
  write doesn't: facts go stale by default, not by opt-in.
- **Pinned facts** go in the stable half; the rest are ranked by the request
  (by shared words with the key, value and description; by meaning with
  [#21](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/21)) and pushed with the turn, or left to `memory_recall`. The stable half stays
  small, and cached, however many facts exist.
- **Bulk import** is idempotent: a fact whose current value matches isn't
  written again (`source="import"`, an operator's rank).

**Environment facts** ([#57](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/57), `environment.py`) are world facts the
machine states: `env.python`, `env.os`, `env.python_packages`, `env.node`,
`env.node_packages`. `detect_environment(path)` reads them from the project's
virtualenv, `.python-version`, `pyproject.toml`, `requirements*.txt`,
`uv.lock` / `poetry.lock`, `package.json` and `package-lock.json`. They
cover direct dependencies at the locked version, with the installed version
where it differs. The hooks run it at SessionStart, with
`this_interpreter=False`, since commonmemo's own Python may not be the
project's. They are written with `source="tool"`, last 7 days, and are
rewritten only when they change or go stale; a key no longer detected is
retracted. `env.*` keys are always pinned, registry or not. Environment
variables are never read, and requirement lines with URLs are skipped.

Conditional facts ("30 days in the EU") come later ([#38](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/38)); until then,
per-region differences use project scopes.

### Feedback

Ratings in [-1, 1], comments, and corrections (`wrong` → `right`).
`feedback_summary` gives counts and approval; *lessons* — corrections, and
anything negative or commented — are pushed so the agent applies them next
time. Feedback also lands in the session's recap text, so engines learn
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

### Procedures: how a task is done

[ADR 0003](adr/0003-procedural-memory.md), [#26](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/26). Entries are keyed by task type and kind
(template, rule, procedure, exemplar) and name; a changed body is a new
version, so `procedure_outcomes` can compare versions by the feedback their
runs got and the steps runs skipped. The more specific scope wins (user
over project over global), and a user's own entry ("my reports skip the
risks section") overrides the project's by name.

A request finds its task in order: named by the app (`session(task=...)`),
matched against `use_when` lines by the memory's ranker (the share of the
task's description found in the request, so a long request isn't diluted)
above `MemoryPolicy.min_procedure_score`, or asked for by the agent
(`memory_get_procedure`, whose description lists the tasks). The push
carries rules and templates in full (each clipped) and a procedure as its
step names; exemplars are pull only. Each pushed or pulled entry is in the
turn trace, and at close the session records `ProcedureFollowed` for it,
with the steps of a step-backed procedure (`@mem.step` names) it never
finished.

Trust follows ADR 0002. Files and application code are current at once; an
agent's entry for everyone, or a learned one, is a proposal
(`confirm_procedure` / `reject_procedure`), and a less trusted source never
replaces a more trusted entry (it is kept as disputed). An agent may write
an entry in its own user's scope without review, as it may write that
user's facts. Exemplars are the user's data: written in the user's scope,
shared only through `confirm_procedure(..., scrubbed=True)` (the user's
known profile values are redacted on the way), removed by `forget_user`;
the approved copy is the operator's and stays.

**Learning from runs** (`learning.py`, phase 2). Off by default:
`Memory(learner=...)` takes any `Learner`, and `ModelLearner(complete)` wraps
any prompt-to-text function, sync or async, so no provider is built in. At
close (or `Memory.distill(session_id)`), the session's messages, tool calls,
feedback, step results and followed procedures go to the learner, with
forgotten text redacted. Raw reasoning is never an input. It returns:

- decision records (`DecisionRecorded`), returned by `memory_recall`;
- lessons for the session's user (`LessonLearned`); one the ranker scores at
  least `MemoryPolicy.lesson_merge_score` against an earlier lesson is merged
  into it (decided at write time and recorded as `same_as`), and lessons are
  pushed in the stable half, the most often seen first. Sessions without a
  user learn no lessons: one would reach every user of the agent unreviewed;
- proposed templates, rules and procedures, and exemplar candidates in the
  user's scope (`source="extracted"`), which wait for review.

What a learner returns is parsed defensively (counts and lengths capped,
malformed items dropped). Learning runs once per session, tagged
`run_id="learn:<session>"`, so `revert(run_id=...)` undoes what one session
taught; a failing learner is logged and never fails the close.

**Facts from what was said** (`extraction.py`, [#60](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/60)). The local engine's recap is a
one-line digest and `memory_recall` doesn't search the Log's messages, so
what a user said in a session was out of reach unless written down. Off by
default: `Memory(extractor=...)` takes any `Extractor`, and
`ModelExtractor(complete)` wraps a prompt-to-text function. At close (before
the engine ingests the session, so its profile and recap include the facts;
or `Memory.extract(session_id)`), the session's messages and tool calls, with
forgotten text redacted, go to the extractor in chunks. It returns keyed
facts, written as `UserFact(source="extracted")` in the session's user's
profile, never as world facts (one user's session can't tell others
anything). The model names the keys: stable for what changes (a newer value
supersedes), its own for an event; the user's current keys are shown so it
reuses one only to update that fact, and keeps what is still true. A value
that matches isn't rewritten; a fact quoting forgotten text is dropped.
Once per session (`run_id="extract:<session>"`), at most 25 facts; a
failing extractor is logged and never fails the close. A claim by a model
about what a user said, written at once to that user's own profile: it is not
covered by `revert`, and a correction is `forget` or stating the fact again.
A bulk first ingestion can trip the default velocity alert (`UserFact` is
watched); that is a warning in the log, not a block.

**What was said** (`said.py`, [#62](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/62)). Recaps, facts and lessons are
what memory made of earlier sessions; this searches the messages themselves.
`MemoryPolicy.messages` windows are pushed with the turn (the request is the
query) and `recall_messages` are returned by `memory_recall`, both 8 by
default ([#70](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/70); 0 turns them off): it puts the user's own earlier words into the prompt. A window is up
to 6 consecutive messages of one of the user's other sessions (lines clipped
to 400 characters), ranked by the memory's ranker against the query; those
past the limit are counted ("left out: earlier exchanges") and pointed at
recall. Scope: the user's own messages from their other sessions, never the
current one or another user's; a session without a user has none (anonymous
sessions share nothing); `recaps_across_agents=False` limits it to this
agent's sessions. Windows carry their event ids (trace item `said`, found by
`why`), forgotten text is redacted, and `forget_user` removes the messages.
`message_dates` ([#71](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/71)) makes a window say when it was said (`stamp`) and what
its relative dates mean (`resolve`, `dates.py`): derived from the Log's text and
timestamps when a window is read, never stored; the ranker scores the messages as
said, since ranking on the annotated text lost evidence. Off by default (no gain on LoCoMo).
**The index** ([#65](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/65)): windows are a projection of the Log. Each message joins
its session's open window as it is logged, and the window's stemmed terms go
into an inverted index (`message_windows`, `window_terms` in `Knowledge`). A
window stores the ids of its events, not their text, so the text stays in the
Log: `prune` drops the windows and terms of what it removed, `forget_user` deletes
them, redaction of forgotten text happens at read time, and a rebuild from the
Log gives the same index. A search asks the index for the 300 windows sharing
the most terms with the query (exactly the keyword ranker's score, so it returns
what a scan would), reads only those events and ranks them. The scratch copies
that `as_of` and reverts replay into keep no windows. An embedding ranker scores
meaning, not shared words, so it scores those windows and the user's 2,000 newest by cosine
over stored vectors ([#77](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/77)): `window_vectors` holds a float32 vector per window, made once from the
redacted text (at most 500 a search, newest first); a probe text embedded with every request detects
another embedder (its vector for the probe changes, and the stored vectors are dropped); a vector made
before a value was forgotten is made again, one per set of forgotten values; `forget_user` and `prune` delete
them. A cache the Log can always rebuild, so it needs no migration. Measured with `scripts/bench_said.py` on one
user: a search takes about 0.3 to 0.6 ms from 10,000 to 500,000 messages (the
old scan: 69 to 93 ms, and only the latest 4,000 messages); writing a message
costs about 0.1 ms more, and the index adds about 0.2 KB a message to the
database ([#76](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/76): a term row is a user number, the term and a window number, in a
`WITHOUT ROWID` table on SQLite, which Postgres's DDL drops; it was 0.57 KB with text ids). Not done: a
term dictionary would shrink it further. On
ten LoCoMo conversations this took memory from 16% to 61% correct and, with
extraction, 65%, against 50% for naive retrieval (`docs/live-runs.md`, [#74](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/74)).

`revert` covers procedures too: an entry changed since the target point is
restored as its next version, one added since is retired, and proposals
that exist only because of what is reverted are rejected.

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
- **Side-effecting steps run at most once, across workers**
  ([#16](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/16)). Before running, a worker *claims* the step: one atomic
  store write (`Store.claim`) that records `StepStarted` only if nobody holds
  the step. Of several workers resuming one session at once, exactly one
  wins. The others get `StepInProgress` (a kind of `StepInDoubt`) while the
  winner's lease (`@mem.step(lease=300)`) is live, and `StepInDoubt` once it
  expires. An expired lease never means "run it again", since the holder may
  have died after the effect. `StepFinished` with a stored result settles the
  claim for good. `StepFailed` (the function raised, or
  `resolve_step(failed=True)`) releases it, in the same write
  (`Store.settle`). Pure steps are not claimed: running one twice is
  harmless by definition.

## Memory history

Memory is event-sourced, so how it changes is already recorded: the Log is
append-only and ordered, and every projection is rebuilt from it
([ADR 0004](adr/0004-history-analytics-and-reversion.md), [#27](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/27)). Phase 1 is read-only:

- **`run_id`** on every event's envelope says what produced it: a deploy,
  an import, a consolidation pass, a framework run. It's set by
  `with mem.run("deploy-42"):`, `mem.session(run_id=...)`, or automatically
  by the CommonADK integration (`commonadk:<run id>`). A session's own run
  wins over an enclosing one; an event that already has a `run_id` keeps it.
- **Metrics** (`mem.history()`):
  - `velocity` — events per bucket by kind, source and scope;
  - `source_mix`;
  - `churn` — keys that keep changing, including A → B → A flip-flops;
  - `review_flow` — world-fact proposals, approvals, rejections, time to review;
  - `staleness`, `growth`;
  - `by_run(run_id)`.

  Counts come from the store (`Store.event_counts`). `prune` writes daily
  rollups of what it deletes first, so history outlives the events.
- **Point in time:** `mem.snapshot(label)` names a log position (nothing is
  copied). `mem.as_of(point)` replays the Log up to a position, time or
  snapshot into a scratch projection. `mem.diff(a, b)` lists what was added,
  changed and removed in between. These cover the kinds CommonMemo owns;
  Recaps and engine-held profile facts aren't replayed. Since every view
  replays today's Log, data forgotten since is gone from past views too.

Phase 2 adds the reins (`reins.py`):

- **`mem.revert(to=point)`** brings world facts and negatives back to a log
  position, time or snapshot. **`mem.revert(run_id=…, source=…,
  session_id=…)`** undoes a cause: the Log is replayed without those
  events, so changes others made later survive.
  - A revert appends compensating events (old value back, a retraction, a
    negative re-added or revoked, leftover proposals rejected), all tagged
    `run_id="revert:<id>"`, plus a `Reverted` record. It never deletes,
    so a revert can be reverted and rebuilds agree. `dry_run=True` only
    plans.
  - It covers world facts and negatives. Tool memory and feedback are
    observations, and Recaps and the UserProfile live in the engine
    (decision: owned kinds only). Forgotten data never comes back.
- **`mem.freeze(scope=…, source=…)`** turns direct world-fact writes there
  into proposals until `unfreeze`. **Budgets**
  (`Memory(budgets={"agent": 100})`, per hour by default) do the same past
  a cap. Nothing is dropped; it waits for review.
- **Alerts are on by default** (decision): after writes, at most once a
  minute, memory checks for velocity spikes, churn (flip-flops) and a
  review backlog. Alerts are logged to `commonmemo.alerts` and sent to
  `AlertPolicy.on_alert`; `mem.alerts()` checks by hand.
  `AlertPolicy(auto_freeze=True)` freezes the source of a spike; it is
  opt-in.

Phase 3 would re-consolidate engine memory after a revert; it isn't planned
for now.

## Storage

[ADR 0001](adr/0001-storage.md) (accepted). Everything CommonMemo keeps itself (the Log,
sessions and claims, the projections in `knowledge.py`, and the local
engine's recaps and facts) is plain SQL in one dialect that SQLite and
Postgres both accept, run through `sqldb.py`. Behind that sit three interfaces
that name the operations and not the SQL ([#52](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/52)): `Store` (the Log and
Working memory; `MemoryStore` is a plain-Python reference, and one suite,
`tests/test_store_contract.py`, holds every store to the contract), `Projections`
(what memory reads from the Log; the suite also runs entirely behind it with
`COMMONMEMO_TEST_INTERFACE=1`) and `Engine` (Recaps and the UserProfile):

- **SQLite** (default): `sqlite:///mem.db`, one file, standard library only.
- **Postgres** (`commonmemo[postgres]`): `postgresql://user@host/db`, with
  `?schema=name` to give one memory its own schema. `REAL` columns become
  `DOUBLE PRECISION` (Postgres' `REAL` is too coarse for timestamps), and
  numeric aggregates come back as numbers.

**Ordering.** The projections catch up by log position. That requires that
an event becomes visible only once every earlier position is visible. On
SQLite one writer at a time gives that. On Postgres, appends, claims and
settles take one transaction-scoped advisory lock per memory, so positions
commit in order across processes. This serializes the log's writes: about
400 events a second on Postgres on a 4-CPU machine, one commit per event
(`docs/benchmarks.md`, #17). Committing a turn's events together comes before
partitioning the log by user or session.

**Setup.** `CREATE ... IF NOT EXISTS` isn't safe when several processes
start at once on Postgres (two can both find a table missing), so creating
schemas and tables holds one database-wide advisory lock.

**Tests.** `COMMONMEMO_TEST_POSTGRES=postgresql://... pytest` runs the whole
suite on Postgres. `tests/conftest.py` makes a throwaway database, and
`tests/pg/sitecustomize.py` maps every SQLite target, in the test process
and in the CLI, MCP and hook subprocesses, to a schema in it. CI runs it on
Postgres 16. Running it this way found three Postgres bugs before release:
the setup race, ambiguous upsert columns, and the test mode itself silently
not applying.

Still to come ([#52](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/52)): the DynamoDB backend and a projection storage
interface defined by operations, not SQL; tool counters without a hot row;
log partitioning; archiving old events to object storage.

## Developer surface

```python
mem = Memory("sqlite:///mem.db", engine="local")   # or "postgresql://user@host/db?schema=x"

@mem.track                         # sync or async; bare or configured
def tool(...): ...

class OrderPlaced(MemoryEvent):    # becomes a frozen kw-only dataclass, kind "order_placed"
    order_id: str

async with mem.session(user_id="u1", agent="a") as s:   # or plain `with` outside an event loop
    ctx = await s.context(query)   # MemoryContext: facts, recaps, state, recent events
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
`search_recaps(query, user_id, agent, limit)`, `user_profile(user_id)`,
`forget_user(user_id)`.

| Engine | Status | Notes |
|---|---|---|
| `local` | shipped | offline and deterministic: rule-based recap digests, keyword retrieval, explicit facts only. For zero-setup use and reproducible tests |
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
| recap retrieval | hybrid `search` → facts → the recaps they came from, best first, with the matching facts appended |
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
noise for extraction and remains in the log. Graphiti is reached
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
| recap retrieval | `search` → memories → the sessions that created them |
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

1. **push** — today, all of it in front of the prompt (stable part first),
   since released CommonADK has no way to add to an agent's instructions;
   if a runner accepts `context=` / `turn_context=`, stable memory goes to
   the instructions and per-turn memory to the prompt instead (see below);
2. **pull** — `prepare_project` adds the memory tools to the agent and
   writes tool memory's notes into the agent's own tools' descriptions
   (both the `ToolSpec.description` the Claude adapter reads and the
   function docstring the other five read), on a copy of the project;
3. **capture** — the hook records the run.

Verified by building a real agent on all six targets from a prepared project
and checking that the note reaches each SDK's tool description.

**Asked of CommonADK, not changed in it.** CommonMemo makes no changes to
CommonADK. The split delivery above needs CommonADK to accept per-run
context; that is proposed in [CommonADK issue
#30](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/30) (`context=` for
the instructions, `turn_context=` for the prompt, and a `RunSession.user_id`).
`run_with_memory` checks the runner's signature and uses the split only when
it is there, so nothing breaks either way.

## Landscape (October 2026)

What exists, and what CommonMemo takes from it:

- **TencentDB Agent Memory** (MIT, Node): L0 raw → L1 facts → L2 scenes → L3
  persona; integrates mainly through an LLM proxy; team memory hub in v2.
  Its tiers are levels of *processing*; CommonMemo's are *scopes and
  lifetimes*, with a resumable Working tier it has no equivalent of.
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

## Seeing what memory did

Every push is a turn trace in the Log (`ContextPushed`): references to what
was pushed, with sources and scores, and what was left out; text only on
request, because it repeats user data. Traces are events like any other, so
`forget_user` and `prune` cover them, and every view reads the same thing:
`Memory.why`, the CLI (`commonmemo show`, `commonmemo trace`), the read-only
inspector (`commonmemo inspect`) and the OpenTelemetry exporter
(`commonmemo.otel`). Decisions ([#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25)): references by default, a read-only
inspector first, OpenTelemetry as the first exporter. See
[seeing-what-memory-did.md](seeing-what-memory-did.md).

## Decision records

- [ADR 0001 — storage](adr/0001-storage.md) (proposed): why the memory CommonMemo owns sits
  in a relational, ACID-per-partition store (SQLite by default, Postgres shared, distributed
  SQL to scale, DynamoDB as the non-SQL backend), and two correctness gaps it found
  ([#15](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/15), [#16](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/16), both fixed in [#2](https://github.com/Mehul-Gupta-SMH/CommonMemO/pull/2)).
- [ADR 0002 — who may change a world fact](adr/0002-world-fact-trust.md) (accepted): agent-written
  world facts are proposals until an operator approves them; a less trusted source never
  supersedes a more trusted one ([#24](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/24)).
- [ADR 0003 — procedural memory](adr/0003-procedural-memory.md) (accepted): templates, rules,
  procedures and exemplars keyed by task type, written and learned ones in one place; the app
  names the task, matching by description is the fallback; learned and agent-written entries
  are proposals; full exemplars stay with their user until approved and scrubbed ([#26](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/26)).
- [ADR 0004 — memory history](adr/0004-history-analytics-and-reversion.md) (proposed): metrics from the
  Log, point-in-time reads, and reverting by appending compensating events, never deleting;
  freeze and budgets reuse proposals; `forget_user` always wins ([#27](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/27)).

## Roadmap

Tracked in GitHub issues, with the [roadmap issue (#14)](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/14) as
the index. Nothing gets coded without an issue.

**Shipped:** the Log and Working tiers, `@mem.track`, `MemoryEvent`, the local
engine and the CommonADK integration ([#1](https://github.com/Mehul-Gupta-SMH/CommonMemO/pull/1)); the Graphiti and Mem0 engines, the
structured kinds and hybrid push/pull with agent tools, and the fixes for
the two correctness bugs ([#15](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/15), [#16](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/16)) ([#2](https://github.com/Mehul-Gupta-SMH/CommonMemO/pull/2)).

**Next**

- [#3](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/3) — live run of the Graphiti and Mem0 engines against real backends and
  an LLM (blocked on the environment: `OPENAI_API_KEY` and access to
  `api.openai.com`)
- [#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18) — evaluation suite. **Deterministic mode shipped:** `evals/`, nineteen
  scenarios with and without memory against a scripted agent, run in CI; results
  in [docs/evals.md](evals.md). Still to come: a live mode with a real model, and
  public long-memory benchmarks (after #3). New memory kinds (#8, #10, #26) are
  measured against it
- [#12](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/12) — release v0.0.1 to PyPI
- [#4](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/4) — MCP server exposing the memory tools. **Shipped:** `commonmemo mcp`, stdio,
  stdlib-only; tools bound to one session, stable memory in the server instructions,
  `memory://context` resource
- [#5](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/5) — harness hook adapters (Claude Code, Codex, OpenCode), for automatic
  capture and push, and one session continuing across harnesses. **Shipped:**
  `commonmemo hook` / `commonmemo setup`; one handler for all three (OpenCode
  through a JS shim); tested against recorded payloads and Codex's own schemas
- [#26](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/26) — procedural memory (ADR 0003). **Phase 1 shipped:** templates, rules,
  procedures and exemplars per task type, loaded from files or written by apps and agents;
  versions, trust and review, push for the named or matched task, `memory_get_procedure`,
  per-version outcomes. **Phase 2 shipped:** learning from runs with any model (decision
  records, merged lessons, proposals), and procedures in `revert` (absorbs #9)
- [#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25) — seeing what memory did. **Shipped:** turn traces (`Memory.traces`,
  `Memory.why`), `commonmemo show` / `trace`, the read-only `commonmemo inspect`,
  and OpenTelemetry spans (`commonmemo.otel`). Still to come: provenance for
  supersede history and the evaluation dashboard. **Review screen shipped**
  ([#45](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/45)): `commonmemo inspect --review` approves and rejects world facts and
  procedure entries

**Planned**

- [#6](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/6) — framework-native adapters (ADK `BaseMemoryService`, OpenAI
  `Session`, LangGraph `BaseStore`, CrewAI `StorageBackend`, AutoGen
  `Memory`, Claude `SessionStore`)
- [#8](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/8) — task memory: open loops and commitments, for proactive follow-ups
- [#27](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/27) — memory analytics and reins: velocity, delta, churn; point-in-time
  reads, diff and snapshots; revert by time or cause; freeze and budgets; see ADR 0004
- [#10](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/10) — agent-relationship memory from multi-agent handoffs
- [#17](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/17) — benchmark the memory workload
- [#11](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/11) — storage interface for projections, Postgres, DynamoDB (see ADR 0001)
- [#13](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/13) — hardening follow-ups (tool-result TTLs, tool preconditions, implicit
  feedback, per-engine recap thresholds, step identity, bulk review of agent
  writes, pruning projections)

**Waiting on CommonADK:** [#7](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/7), split context delivery once [CommonADK #30](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/30)
lands. CommonMemo makes no changes to CommonADK; anything it needs from
CommonADK is raised as a CommonADK issue.
