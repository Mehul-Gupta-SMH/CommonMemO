# Changelog

## 0.0.1 — unreleased

Facts from what was said ([#60](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/60)):

- **`Memory(extractor=ModelExtractor(complete))`** (off by default): when a
  session closes, a model lists its durable facts and they are written to the
  user's profile as `UserFact(source="extracted")`, before the engine ingests
  the session. They are pushed and recalled like any profile fact and removed
  by `forget_user`. Forgotten text is redacted from the model's input and
  facts quoting it are dropped; long sessions go in chunks; once per session
  (`mem.extract(session_id)` to run it yourself); a failing model never fails
  the close.
- **Eval scenario "said once, recalled later"** (18 of 18).
- **LoCoMo:** the benchmark's `extracted` condition now uses this feature:
  38/48 with the first prompt and 35/48 with the shipped one (the same within
  noise; 14/48 with no memory, 23/48 with memory as shipped), at about 0.03
  dollars a question against 0.11 for the whole conversation.

LoCoMo live benchmark ([#59](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/59), part of [#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)):

- **`scripts/locomo_benchmark.py`** runs a LoCoMo conversation with Claude Code
  as the model in four conditions (no memory; memory as shipped; memory with
  model-extracted facts; the whole conversation), judged by a model. The data
  is downloaded at run time (CC BY-NC 4.0) and disguised against
  contamination; answers are cached per question and a budget caps the spend.
- **First result** (48 questions, one conversation; `docs/live-runs.md`): 14/48
  with no memory, 23/48 with memory as shipped, 38/48 with extracted facts,
  39/48 with the whole conversation at 3.6 times the cost per question. The
  local engine's one-line recaps are why memory as shipped stays close to the
  floor: [#60](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/60) (model-backed extraction at session end).

Environment facts ([#57](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/57)):

- **`mem.detect_environment(path)`** records the Python version, OS and the
  versions the project pins as `env.*` world facts. The pins are read from
  `pyproject.toml` and `requirements*.txt`, locked in `uv.lock` /
  `poetry.lock`, with the installed version where it differs; `package.json`
  and the Node version are read the same way.
  - They are trusted (`source="tool"`), always pinned in the stable half, and
    last 7 days.
  - Only changes and stale values are written; a key that is gone is
    retracted.
  - Environment variables are never read.
- **The hooks run it at SessionStart** (`--no-environment` or
  `COMMONMEMO_NO_ENVIRONMENT` turns it off). They use only the project's
  files and virtualenv, not commonmemo's own interpreter.
- **Eval scenario "pinned library version":** with memory, the agent calls
  pydantic 1's `.dict()`; without it, the v2 `.model_dump()`.

Two findings of the live run fixed ([#51](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/51)):

- **Procedure matching scores the task's description, not the request.** A
  request now matches a task when it contains enough of the task's name and
  `use_when` words (half, by default: `min_procedure_score` 0.5). A long
  request ("...We shipped the login page; the payments API is late...") no
  longer dilutes the match, and a short one sharing two common words ("the
  team meeting this week") no longer triggers it. With an embedding ranker
  it is cosine similarity, as before.
- **Recaps name the facts a session learned, without their values.** The
  local engine's recap said "learned: key=value; ...", so a recap could carry
  facts the push had just left out ("More in memory than shown here").
  Values live in the UserProfile, where the push ranks and limits them; the
  recap now says "learned about the user: key, key, ..." (ten names at most).

Traces tell the truth (bugs found by the live run):

- [#49](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/49): a push's trace lists only the items in the text it delivered.
  A turn push (`push_turn`, the hooks' `UserPromptSubmit`) no longer claims
  the stable half's facts, constraints and lessons; `trace_items(ctx, part)`.
- [#50](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/50): `memory_recall` records what it returned as a `recall` trace
  (references, like any push; the query only with `trace_text`), so
  `commonmemo trace`, `why` and the inspector show what the agent pulled.
  Read-only tools still record nothing.

First live run, with Claude Code as the model ([#48](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/48)):

- `scripts/live_claude_code.py` runs five scenarios through real headless
  `claude -p` sessions, wired with CommonMemo's hooks and MCP server as a
  user would set them up: preference across sessions, recall on purpose,
  procedure by description, learning (`ModelLearner` over `claude -p`) and
  injection. Results, verbatim, are in `docs/live-runs.md`: 5 of 6 checks
  passed for $0.08.
- It found two bugs, filed as [#49](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/49) (a turn push's trace lists stable
  items it didn't deliver) and [#50](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/50) (`memory_recall` leaves no record),
  plus two weaknesses in [#51](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/51).

Postgres ([#11](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/11), [ADR 0001](docs/adr/0001-storage.md), now accepted; the rest of the ADR is [#52](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/52)):

- `postgresql://user@host/db` store URLs (`commonmemo[postgres]`), for the
  Log, sessions and claims, the projections and the local engine's recaps;
  `?schema=name` gives a memory its own schema. `PostgresStore`, and
  `SqlStore` underneath both stores; `SQLiteStore` is unchanged.
- New `commonmemo.sqldb`: one SQL dialect for both databases. SQLite-only
  SQL (`INSERT OR IGNORE` / `OR REPLACE`) is now `ON CONFLICT` upserts,
  which SQLite runs too.
- Log positions commit in order on Postgres (a transaction-scoped advisory
  lock around appends, claims and settles), so catching up never skips an
  event committed by another process. Setup DDL is serialized, because
  `CREATE ... IF NOT EXISTS` races when several processes start at once.
  `forget_user` and `prune` are now single transactions on both databases.
- The whole test suite runs on Postgres with `COMMONMEMO_TEST_POSTGRES`,
  and CI runs it on Postgres 16. New Postgres-only tests cover four writer
  processes against one reader, exactly-once appends from six threads,
  schema isolation, and numeric results.

README brought to the level of agent-gauntlet and CommonADK ([#46](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/46)):

- Banner, pitch, CI badge, real inspector screenshots, an offline
  quickstart, How it works (a diagram), Status, Honest caveats, Related and
  repo metadata. Stale parts corrected: the MCP tool list, what `revert`
  covers, the eval list, the inspector's review mode.
- `scripts/demo_memory.py` builds the small memory the screenshots come
  from (offline, no keys).
- The inspector and `commonmemo show` / `trace` now describe procedure,
  lesson and decision events in words instead of raw payloads; a learned
  lesson in a trace is labelled with its text; the timeline's time column no
  longer wraps; `--why` no longer prints "in session None".

Review screen in the inspector ([#45](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/45)):

- `/review` lists proposed and disputed world facts and procedure entries,
  each with what it would replace (`views.review_queue`, `/api/review`).
- `commonmemo inspect --review [--reviewer NAME]` adds Approve (a world
  fact optionally corrected), Reject (with a reason) and, for an exemplar
  candidate, Share with everyone ("personal data checked" required, known
  profile values redacted). Without `--review` the inspector stays read-only.
- Actions are plain form posts with a per-start token; a request whose Host
  or Origin isn't the inspector's own, or marked cross-site, is refused.
  Reads now refuse a foreign Host too (DNS rebinding).
- Every action runs under `run_id="review:<reviewer>"`, so
  `revert(run_id=...)` undoes a review session.

Learning from runs, procedural memory phase 2 ([#26](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/26), ADR 0003 section 6):

- `Memory(learner=...)`, off by default. `ModelLearner(complete)` works with
  any model: `complete(prompt) -> text`, sync or async. `Learner` is the
  protocol for writing your own. `Memory.distill(session_id)` runs it on
  demand; at close it runs by itself, once per session.
- It writes decision records (`DecisionRecorded`, `Memory.decisions`, and
  `[decision]` lines in `memory_recall`), lessons for the session's user
  (`LessonLearned`, merged with an earlier lesson above
  `MemoryPolicy.lesson_merge_score`, pushed as "Lessons from past
  sessions"), and proposed templates, rules, procedures and exemplar
  candidates (`source="extracted"`, waiting for review).
- The learner sees messages, tool calls, feedback, step results and followed
  procedures, with forgotten text redacted; never raw reasoning. Its reply is
  parsed defensively; a failing learner never fails the close. Sessions
  without a user learn no lessons.
- `forget_user` removes the user's decisions and lessons.
- `revert` now covers procedures (`REVERTIBLE` gains `"procedure"`):
  changed entries are restored as their next version, added ones retired,
  and proposals from what is reverted rejected. Everything one session's
  learning wrote carries `run_id="learn:<session>"`, so it reverts by cause.
- New eval scenario **learning from runs** (16 of 16): a correction in one
  session becomes a lesson that changes the same user's next answer, and no
  other user's.
- Projection schema version 5.

Procedural memory, phase 1 ([#26](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/26), [ADR 0003](docs/adr/0003-procedural-memory.md), accepted):

- New kind: **templates, rules, procedures and exemplars per task type**
  (`commonmemo.procedures`, events `ProcedureEntry`, `ProcedureConfirmed`,
  `ProcedureRejected`, `ProcedureRetired`, `ProcedureFollowed`). Scopes
  `global`, `project:<name>`, `user:<id>`; the most specific wins.
- `Memory.load_procedures(dir)`: Markdown files with a front matter block
  (task from the directory, name from the file), or JSON/YAML; current at
  once, idempotent, a changed body is a new version.
- `Memory.session(task=...)` / `context(task=...)` names the task; otherwise
  the request is matched against `use_when` lines by the memory's ranker
  (`MemoryPolicy.min_procedure_score`, 0.5 since #51). The turn half gets rules and
  templates (clipped at `max_procedure_chars`) and a procedure's step names.
- Agent tools: `memory_get_procedure(task)` (only when tasks exist; its
  description lists them) and `memory_save_procedure` (the user's own way at
  once, or for everyone as a proposal). Also over MCP and the hooks.
- Review as for world facts: `pending_procedures`, `confirm_procedure`,
  `reject_procedure`, `retire_procedure`; a less trusted source is kept as
  disputed instead of replacing a more trusted entry.
- Exemplars (`Session.save_exemplar`) stay in the user's scope; sharing one
  needs `confirm_procedure(id, scrubbed=True)`, which also redacts the
  user's known profile values. `forget_user` removes the user's entries,
  exemplars and proposals; approved copies stay.
- Runs: each pushed or pulled entry is in the turn trace, and at close the
  session records `ProcedureFollowed` with the steps of a step-backed
  procedure it never finished (`Session.follow` to state it yourself).
  `Memory.procedure_outcomes(task)`: runs, ratings, corrections and skipped
  steps per entry version.
- Keyword ranking: single letters are no longer terms ("week's" no longer
  adds "s"), and a final "e" is dropped from longer stems, so "write" and
  "writing" match.
- New eval scenario **procedure by description** (15 of 15): two report
  requests that don't name the task get the house template; an unrelated
  request gets none.
- Projection schema version 4: existing projection files rebuild from the Log.

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
