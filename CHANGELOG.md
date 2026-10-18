# Changelog

## 0.0.1 — unreleased

Memory history, phase 2: reins ([#27](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/27), ADR 0004 accepted):

- **`mem.revert(to=…)`** and **`mem.revert(run_id=…, source=…, session_id=…)`**
  for world facts and negatives. Compensating events tagged
  `run_id="revert:<id>"` and a `Reverted` record; never deletes; a revert can
  be reverted; `dry_run=True` plans only. Forgotten data never comes back.
- **`mem.freeze` / `unfreeze` / `freezes`**, and per-source **budgets**
  (`Memory(budgets=...)`): both turn direct world-fact writes into
  proposals.
- **Alerts, on by default:** velocity spikes, churn and review backlog,
  logged to `commonmemo.alerts` and sent to `AlertPolicy.on_alert`;
  `mem.alerts()` checks by hand; `AlertPolicy(auto_freeze=True)` is opt-in.
- `mem.diff` now identifies a negative by scope and statement, so a rule
  that was re-added doesn't show as removed and added.

Memory history, phase 1 ([#27](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/27), [ADR 0004](docs/adr/0004-history-analytics-and-reversion.md)), read-only:

- **`run_id`** on every event's envelope: `with mem.run(...)`,
  `mem.session(run_id=...)`, and `commonadk:<run id>` from the CommonADK
  integration. Store files from before it get the column on open.
- **`mem.history()`:** `velocity`, `source_mix`, `churn` (with
  flip-flops), `review_flow`, `staleness`, `growth`, `by_run`.
  `Store.event_counts`; `prune` keeps daily rollups of what it deletes.
- **`mem.snapshot(label)`, `mem.as_of(point)`, `mem.diff(a, b)`** for the
  kinds CommonMemo owns, replayed into a scratch projection; forgotten data
  stays gone from past views. `Store.position`, and a `run_id` filter on
  `Store.events`.

World facts: who may change them ([#24](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/24), [ADR 0002](docs/adr/0002-world-fact-trust.md)):

- **Fix: one user's conversation could change the world facts every user
  was told.** An agent's `memory_note_fact` wrote at project scope, and the
  newest value won. Agent-written world facts are now **proposals**: shown
  only to the user who caused them, marked as unconfirmed, until an operator
  approves them. New `Memory.pending_world_facts`, `confirm_fact`
  (optionally correcting the value), `reject_fact`;
  `Memory(world_fact_approval=False)` turns approval off.
- **Source decides before recency:** `extracted` < `agent` < `tool` <
  application code. A less trusted write never supersedes a more trusted
  fact; it is kept as *disputed* for review.
- `Memory.retract_fact` ends a fact with nothing replacing it.
- The push shows each world fact's age and source (`as of 2026-10-14,
  operator`).
- Approvals carry a copy of the fact, so they outlive `forget_user` of the
  proposer; unapproved proposals are deleted with their user. Proposals are
  not fed to engines as things the user said.
- Projection tables carry a schema version; a file from an older release
  rebuilds them from the log on open.
- Fix: `describe()` checked `Feedback.kind` after the field became
  `feedback_kind`, so engines got corrections as plain negative ratings.

Correctness fixes:

- **Projections can no longer fall behind the log**
  ([#15](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/15)). They store the last log position applied and
  catch up after every write and on open, in the same transaction as the
  changes, so a crash between logging and applying loses nothing and nothing
  is counted twice. New `Store.events_after`.
- **Side-effecting steps run at most once across concurrent workers**
  ([#16](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/16)). The step is claimed with an atomic conditional
  write before it runs; losers get `StepInProgress` (a `StepInDoubt`) while
  the winner's lease is live, and `StepInDoubt` after. New `Store.claim`,
  `Store.get_claim`, `Store.settle`, and `@mem.step(lease=...)`.
- `Feedback`'s `kind` field is renamed `feedback_kind`. As `kind` it replaced
  the event type, so feedback came back from the log as another kind and
  `rebuild_knowledge` dropped corrections and comments. Feedback logged under
  the old name still reads back. A field named `kind` on any event is now
  refused.

Tier names: the four tiers are now **Log**, **Working**, **Recaps** and
**UserProfile** (before: event, session, episodic, user), in the docs and in
code ([#19](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/19)). Renamed, with no aliases (nothing is released yet):

- `Episode` → `Recap`; `MemoryContext.episodes` and `Session.episodes` →
  `.recaps`
- `Engine.search_episodes` → `Engine.search_recaps`; `Engine.user_facts` →
  `Engine.user_profile`
- `MemoryPolicy.episodes` → `.recaps`, `min_episode_score` →
  `min_recap_score`; `Session.context(episodes=)` and
  `Memory.context(episodes=)` → `recaps=`

Unchanged: event types (`UserFact`, ...), `Fact`, Graphiti's own episode
API, the SQLite table names (renaming them would hide existing data), and
the headings the agent sees, which stay in plain English.

Structured memory kinds and hybrid reads:

- **Tool memory** from every `ToolInvoked` (and CommonADK `tool_call`): calls,
  failures, durations, scrubbed error signatures, and the argument-shape
  change that fixed a failure. Holds shapes, never values, so it is safe to
  share across users. `tool_profile`, `tool_notes`.
- **World memory**: `WorldFact` with `global` / `project:<name>` scopes,
  superseding history and `valid_until` expiry. `learn`, `world_facts`.
- **Feedback**: ratings, comments and corrections; summaries and lessons.
- **Negatives** ("never X", scoped, revocable) and **tombstones** (`forget`)
  that keep forgotten facts from being re-learned.
- **Resume points**: `checkpoint` / `resume_point`, and `@mem.step` for
  replayable steps; interrupted side-effecting steps raise `StepInDoubt`.
- **Hybrid reads**: `MemoryPolicy`; context split into `stable_text()` and
  `turn_text()`, framed as data; episodes pushed only above a relevance
  threshold; recent events off by default.
- **Agent tools**: `memory_recall`, `memory_remember`, `memory_note_fact`,
  `memory_avoid`, `memory_save_progress` (sync, typed, documented; writes
  marked `source="agent"`).
- **CommonADK**: `prepare_project` adds the memory tools and writes tool
  notes into the tools' own descriptions; `run_with_memory` puts the memory
  in front of the prompt, which is what released CommonADK supports. If a
  runner accepts `context=` / `turn_context=` (proposed in CommonADK issue
  #30), stable and per-turn memory go there instead, automatically.
- Projections are kept per event, ignore duplicate deliveries, and can be
  rebuilt from the log (`rebuild_knowledge`).

Mem0 engine:

- `Mem0Engine`: each closed session is one Mem0 `add`, tagged with its
  session id; extracted memories join explicit facts (explicit first) with
  provenance through the origin session; retrieval ranks past sessions by
  matching memories; re-ingest replaces a session's memories;
  `forget_user` reaches Mem0. `pip install "commonmemo[mem0]"`.
- Episode ranking shared between the Graphiti and Mem0 engines
  (`engines/_ranking.py`).
- `scripts/live_engines.py`: a live run of both engines against real
  backends and an LLM.

### Fixed

- **The Graphiti engine could not create any episode.** It passed a
  session-derived `uuid=` to `Graphiti.add_episode`, which in Graphiti means
  "reprocess this existing episode" and raises `NodeNotFoundError` when
  there is none -- so every real ingest failed. The fake graph in the tests
  accepted the argument, so only the first live run exposed it. The engine
  now lets Graphiti assign the uuid and records it in the local index
  (`LocalEngine.set_external_id`); the backend test binds the call against
  graphiti-core's real signature.

Graphiti engine:

- `GraphitiEngine`: one Graphiti episode per closed session in the user's
  graph partition; retrieval ranked by matching graph facts; extracted facts
  alongside explicit ones, with provenance; re-ingest replaces; `forget_user`
  deletes the partition. `pip install "commonmemo[graphiti]"`.
- `Engine.bind(store)` so an engine can keep local state in the store's file.
- The `graphiti` extra declares `httpx`, which graphiti-core imports without
  declaring.

First skeleton:

- Event tier: `MemoryEvent` (subclass to define events, no decorator needed),
  built-ins `Message`, `ToolInvoked`, `UserFact`, `ExternalEvent`; `@mem.track`
  for sync and async functions.
- Session tier: resumable sessions with persisted state, ownership check on
  resume, TTL.
- `Engine` contract for episodic and user memory, with the offline `local`
  engine (rule-based episodes, keyword retrieval, superseding facts with history).
- `forget_user` across every tier; `prune` for event retention.
- CommonADK integration: one hook for all six CommonADK targets, plus
  `run_with_memory`.
