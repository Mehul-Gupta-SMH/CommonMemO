# ADR 0004 — Memory history: analytics, point-in-time reads and reversion

**Status:** Accepted; phases 1 and 2 implemented · **Date:** 2026-10-17 · **Tracks:** [#27](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/27)

## Context

Memory changes all the time: users state facts, agents propose world facts,
engines extract, consolidation learns procedures. Nothing shows how it is
evolving, and nothing can undo a bad stretch of it. A poisoning attempt, a
runaway agent loop, an unstable extraction run or a bad procedure version is
visible only once an agent says something wrong.

CommonMemo is event-sourced. The Log is append-only and ordered (`seq`);
every event carries a timestamp, a source and its envelope (session, user,
agent); every projection is rebuilt from the Log (`catch_up`, `rebuild`).
History is therefore already recorded. This ADR decides how it is read,
measured and replayed, and what reverting means.

## Decision (proposed)

### 1. History is read from the Log; metrics are rollups

- Metrics are computed from the Log's `(seq, ts, kind, source, envelope)`.
  They are: velocity, delta, churn, source mix, review flow, staleness and
  growth (see #27).
- A **rollup projection** keeps daily counts per kind, scope and source. It
  is written **before `prune`** removes old events, so metrics outlive the
  events they summarize.
- Read-side metrics (what was pushed, recalled, used) come from turn traces
  (#25), and outcome impact from the evaluation suite (#18). This ADR does
  not define them.

### 2. Point-in-time reads replay the Log into a scratch projection

`mem.as_of(t)` rebuilds the projections from events up to a time, a log
position or a named snapshot, into a separate scratch projection that is
never the live one. `mem.diff(a, b, kind=, scope=)` compares two such views.
`mem.snapshot(label)` records a `Snapshot(label, seq)` event; snapshots are
names for log positions, not copies.

A full replay is fine at today's sizes. At scale, `as_of` starts from the
nearest stored checkpoint of the projections instead of from position 0
(#11, #17).

### 3. Reverting appends compensating events; it never deletes from the Log

A revert computes the difference between now and the target, and appends
events that undo it:

- a fact that changed gets its old value back;
- a fact that was added is retracted;
- a fact that was retracted is restored.

Every compensating event points at the revert that caused it
(`reverts=<revert id>`).

- **Auditable:** the Log says who reverted what, when and why.
- **Reversible:** a revert can itself be reverted.
- **Consistent:** rebuilds replay the compensating events like any other, so
  live and rebuilt projections agree.

Rewinding by deleting events was rejected: it destroys the audit trail,
can't be undone, and breaks every other process's log position.

### 4. Revert by time, or by cause

- **By time:** "kind K in scope S back to time T / snapshot X".
- **By cause:** "undo what came from source `extracted`", "…from session X",
  "…from run R", "…since snapshot D".

  Cause needs a tag on every write, so the event envelope gains an optional
  **`run_id`**. Integrations, consolidation runs and bulk imports set it.
  Events from before the field existed can only be reverted by time.

### 5. Authority and limits

- A revert that touches a shared scope (`global`, `project:*`) has operator
  rank and is audited like `confirm_fact` (ADR 0002). Agents get no revert tool.
- **`forget_user` always wins.** A revert, `as_of` or snapshot never brings
  back data a user asked to delete; tombstones and deletions are applied after
  any replay.
- **`prune` sets the horizon.** Reverts and `as_of` can't reach past the
  oldest kept event; they fail loudly rather than silently returning a
  partial past.

### 6. Freeze and budgets reuse the proposal mechanism

- **Freeze** marks a scope or source frozen. While frozen, direct writes
  there are recorded as proposals (ADR 0002) instead of applied. A freeze is
  an event, so it is replayed and audited too.
- **Budgets** cap writes per source per window. Writes over the cap become
  proposals too, so nothing is dropped and a runaway loop can't rewrite
  memory faster than anyone can review it.
- **Alerts** fire on velocity or churn anomalies. Automatic freeze on an
  alert is opt-in.

### 7. Engine-held memory

Recaps and UserProfile facts held by Graphiti or Mem0 aren't projections
CommonMemo can replay. Reverting them means re-consolidating the affected
sessions from the Log after the compensating events are written. That is
correct, but it costs LLM calls, so it is phase 3 and explicit (`engines=True`).
Explicit `UserFact` events are in the Log, so the profile's explicit facts are
revertible in phase 2; extracted ones need re-consolidation.

## Consequences

- One new envelope field (`run_id`), one rollup projection, three new event
  types: `Snapshot`, `Reverted` and `Freeze`/`Unfreeze`. Compensating events
  reuse the existing kinds with a `reverts` reference.
- `prune` gains a step: write rollups first.
- The inspector (#25) gets the timeline, diff and revert screens; the
  evaluation suite (#18) gets a "detect, freeze, revert, recover" scenario.

## Open questions

- ~~**Do reverts cover engine memory?**~~ **Decided (owner, 2026-10-17):
  owned kinds only.** Reverts cover world facts and negatives; Recaps and
  the UserProfile (explicit profile facts included, since engines hold
  them) are not reverted. Engine re-consolidation is not planned.
- ~~**Automatic freeze on by default?**~~ **Decided (owner, 2026-10-17):
  alerts on by default, automatic freeze opt-in.** Shipped in phase 2.
- ~~**`run_id` now?**~~ **Decided (owner, 2026-10-17): yes**, shipped with
  phase 1. Without it, revert by cause would be impossible for everything
  written before it landed.

## Alternatives considered

- **Rewind by deleting events after T.** Rejected: no audit trail,
  irreversible, and other processes' log positions would point at events that
  no longer exist.
- **Periodic full copies of the projections.** Simple, but coarse (only the
  times copies were taken), and they duplicate user data that `forget_user`
  would then have to chase. Snapshots are positions in the Log instead.
- **External analytics only** (export the Log to a warehouse). Useful and
  still possible, but reins need to act on memory itself, which a warehouse
  can't.
