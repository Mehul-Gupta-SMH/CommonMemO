# Changelog

## 0.0.1 — unreleased

Ranking by meaning ([#21](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/21)):

- New `commonmemo.ranking`: a `Ranker` decides which profile facts make the
  push (when there are more than `max_facts`: related ones first, not the
  first 20), which world facts a request is about, and what `memory_recall`
  returns. `KeywordRanker` (default, no dependencies: stemmed terms, stop
  words dropped) or `EmbeddingRanker(embed)` (cosine similarity from any sync
  or async embedder, cached). `Memory(ranker=...)`; without one, the
  engine's (`Engine.ranker()`: Graphiti's embedder, Mem0's embedding model),
  else keywords. `MemoryPolicy.min_recall_score` sets recall's bar.
- `memory_recall` no longer filters by raw shared words; it uses the ranker,
  best first.
- New eval scenario **recall by meaning** (14 of 14): four questions phrased
  unlike the stored keys; 4/4 recalled with an embedding ranker, 1/4 by
  shared words. The eval uses a small offline concept embedder
  (`evals/embedder.py`), not a model; a live comparison comes with #18's live
  mode.

World facts as configuration ([#24](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/24), decisions recorded there):

- **Optional key registry** (`commonmemo.facts`): `FactKey(key, description,
  allowed=, lifetime=, pinned=)` via `Memory(fact_keys=...)`,
  `Memory.register_fact_key` or `Memory.load_facts`. `memory_note_fact`
  lists registered keys (and so does its MCP schema). An agent's write to an
  unregistered key, a near-duplicate of a registered one, or a disallowed
  value stays a proposal with a note (`WorldFact.note`), even with approval
  off; an operator's disallowed value raises `ValueError`.
- **Per-key lifetimes** set `valid_until` when a write doesn't.
- **Pinned facts** stay in the stable half; with a registry, the rest are
  ranked by the request (key, value and description) into the turn half
  (`MemoryContext.world_turn`, "Facts for this request"), or left to recall
  and counted in the "more available" line.
- **Bulk import** from JSON or YAML, idempotent (`source="import"`, shown as
  "config").
- Eval scenarios **staleness** and **precedence**; 13 of 13 pass.
- Conditional facts are later: [#38](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/38).

Seeing what memory did ([#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25)):

- **Turn traces.** `ContextPushed` now records each pushed item by
  reference (kind, key or id, source, score, source events) and what was left
  out; the pushed text only with `MemoryPolicy(trace_text=True)`. New
  `Session.push(query, reason=, part=)` (`push_turn` is `push(reason="turn")`);
  the hooks record session starts, and CommonADK runs record each turn.
  `Memory.traces(...)` and `Memory.why(key, session_id=...)`, which answers
  with the events an item came from. `forget_user` removes a user's traces.
  New `Store.get_events(ids)`.
- **CLI:** `commonmemo show user|session ID`, `commonmemo trace SESSION
  [--why KEY]`.
- **`commonmemo inspect`:** a local, read-only web inspector (stdlib, 127.0.0.1,
  GET only, no scripts, strict CSP, every value escaped): overview with world
  facts and proposals, user page, session timeline with turn traces and
  "why" links; JSON under `/api/`.
- **OpenTelemetry:** `commonmemo.otel.instrument(mem)` emits a
  `commonmemo.push` span per push under the current span; `export(...)` for
  stored traces. `otel` extra; the user's id and pushed text only on request.
  New `Memory.listeners`; a failing listener never breaks a write.
- Docs: [seeing what memory did](docs/seeing-what-memory-did.md).

More in memory than pushed ([#23](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/23)):

- When the push leaves things out and the agent has `memory_recall`, the
  turn text ends with one line: "More in memory than shown here: 5 facts
  about the user. Call memory_recall with what you need."
  (`MemoryContext.more_line`). It counts facts, world facts, negatives and
  lessons past their `max_*`, recaps past the limit that clear half the
  relevance threshold, and entries cut for length (`MemoryContext.more`). No
  line when nothing was cut or there's no recall tool; the character cap
  never cuts it.
- `build_agent_tools` marks the session (`Session.recall_tool`); the hooks
  take `--recall-tool` / `COMMONMEMO_RECALL_TOOL=1`, also passed on by
  `commonmemo setup`.
- **Fix:** facts stated in the current session were merged into a profile
  already cut to `max_facts`, so which 20 facts were pushed depended on the
  engine's order. They are now merged first and cut once.
- New eval scenario, **more available**: with the line the agent makes one
  recall call and answers correctly; without it, none, and it can't. 11 of
  11 pass.

Mid-run refresh ([#22](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/22)):

- **`Session.refresh(text)`**: memory to push mid-run after a tool result or
  the agent's own message: recaps of earlier sessions that share at least
  `MemoryPolicy.refresh_min_overlap` (3) distinct words with it and weren't
  pushed yet this turn. At most `max_refreshes` (3) per turn;
  `MemoryPolicy(refresh=False)` turns it off. Forgotten values stay
  redacted; recaps follow `recaps_across_agents`. It searches the local
  recap index only, so it never costs a Graphiti or Mem0 round trip.
- **`Session.push_turn(query)`**: `context()` for the start of a turn,
  recorded as pushed. Every push is a new `ContextPushed` event (reason,
  recap ids, characters): bookkeeping that bounds refreshes across hook
  processes and records what memory costs per run.
- The hooks use them: prompts are `push_turn`, and `PostToolUse` /
  `PostToolUseFailure` return refreshed memory as `additionalContext`
  (Claude Code and Codex; checked against Codex's output schema). The
  OpenCode plugin adds it to the next model call's system prompt.
- New eval scenario, **mid-run refresh**: with refresh the agent recovers
  the device the way an older session found works; with refresh off it
  doesn't. The run reports the characters pushed with and without refresh.
  10 of 10 pass.

Recaps shared across agents ([#34](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/34)):

- **Behaviour change:** a user's recaps now come from all their agents,
  not just the current one, so with one agent per harness, Claude Code sees
  the user's Codex sessions. New `MemoryPolicy.recaps_across_agents`
  (default `True`); `False` restores per-agent recaps, for the push and for
  `memory_recall` alike. Sessions without a user stay per agent.
- A recap from another agent is labelled `(in <agent>)` unless its summary
  already names the agent (`MemoryContext.recap_line`). Tool notes and
  lessons stay per agent.

Harness hooks: automatic capture and push ([#5](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/5)):

- **`commonmemo hook --harness claude-code|codex|opencode`**: one handler for
  the harnesses' hook events, JSON on stdin. Session start and each prompt
  push memory back (`additionalContext`); prompts, tool calls (into tool
  memory), replies, compaction (a checkpoint) and session end (close and
  consolidate) are recorded. Never blocks the harness: errors go to stderr
  and it exits 0. `--read-only` pushes without recording. Long tool input and
  output are cut to 4000 characters; the memory tools' own calls are skipped.
- **`commonmemo setup <harness>`** prints the config: the hooks block for
  Claude Code (`.claude/settings.json`) and Codex (`.codex/hooks.json`), or
  the OpenCode plugin (`opencode_plugin.js`), a thin JS shim that translates
  OpenCode's events into the same payload.
- Tested against recorded payloads: Claude Code 2.1.302's hook inputs, and
  Codex payloads checked against the hook schemas built into Codex 0.163.0
  (vendored in `tests/fixtures/hooks/codex-schema`). The OpenCode plugin
  runs under Node in the tests.
- **One session across harnesses:** the same `--session` continues a
  session from Claude Code in Codex. **Behaviour change:** a resumed
  session's handle now reports, and stamps new events with, the agent that
  resumed it; the stored session record still keeps its first agent.
- `python -m commonmemo` runs the `commonmemo` command.

MCP server ([#4](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/4)):

- **`commonmemo mcp`** (also `python -m commonmemo.mcp`): the memory tools
  for MCP clients over stdio, bound to one session, stdlib-only. Tools from
  `build_agent_tools` (writes recorded with `source="agent"`; `--read-only`
  gives `memory_recall` only); stable memory in the `initialize`
  instructions; the full context as resource `memory://context`.
- Configured by flags or `COMMONMEMO_*` variables (store, engine, project,
  user, agent, session, read-only). Resuming another user's session is
  refused and the server exits with an error; stdin closing closes and
  consolidates the session.
- Speaks the `initialize` handshake (2024-11-05 to 2025-11-25) and answers
  `server/discover` with "method not found", so 2026-revision clients fall
  back to it. Tested over real stdio and with the official MCP client
  (`mcp` extra, new CI leg).
- New console script `commonmemo`.

Evaluation suite, deterministic mode ([#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)):

- `evals/`: nine scenarios run with and without memory against a scripted
  agent that decides only from the prompt and tool descriptions, so they
  measure what memory actually delivers. `python -m evals [--md PATH]`;
  `tests/test_evals.py` runs them in CI. Baseline in `docs/evals.md`: 9 of 9
  pass with memory, and memory turns a failure into a success in all 5
  scenarios that have a without-memory baseline.
- **Fix, found by the suite: a forgotten fact came back through recaps.** A
  recap quotes the session it summarizes, including profile facts stated
  in it, so after `forget("home_address")` the address still reached the
  agent in the push and through `memory_recall`. Recaps are now redacted
  at read time for every engine: forgotten values and tombstone patterns
  become `[forgotten]`. A value stated again after forgetting shows again.

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
