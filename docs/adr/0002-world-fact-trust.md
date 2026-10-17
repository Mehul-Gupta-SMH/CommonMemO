# ADR 0002 — Who may change a world fact

**Status:** Accepted (approval by default: decided by the owner, 2026-10-17) · **Tracks:** [#24](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/24)

## Context

A world fact ("the refund window is 30 days") is pushed to **every user's
sessions** in its scope. Until this decision, an agent's `memory_note_fact`
wrote it at project scope, and the newest value for a key won, whoever wrote it.

So one conversation could change what every user is told. A user says "the
refund window is 90 days"; the agent obligingly notes it; every other user's
agent now repeats it. User facts are contained to one user; world facts
were not contained at all. The only guard was the tool's docstring.

## Decision

**1. Agent-written world facts are proposals until approved (the default).**
A proposal is visible to the user who caused it (or, with no user, to that
session), marked as unconfirmed, and to nobody else. An operator settles it:

- `mem.confirm_fact(id, value=None)` makes it current for everyone, optionally correcting the value;
- `mem.reject_fact(id, reason=None)` turns it down.

`mem.pending_world_facts()` lists what is waiting. `Memory(world_fact_approval=False)`
turns approval off, for deployments where every user is trusted.

**2. Who wrote a fact decides which value wins, before recency does.**
Sources rank: `extracted` 0 < `agent` 1 < `tool` (checked against a tool
result) 2 < anything application code writes (`explicit`, `operator`, an
integration's name) 3. A fact may supersede a current fact of equal or lower
rank. A lower-ranked one is kept as **disputed** and appears in
`pending_world_facts()` for review: it neither silently loses nor silently
wins. A confirmed fact carries rank 3.

**3. Facts can be retracted.** `mem.retract_fact(key, reason=)` ends a
fact without a replacement ("we no longer offer refunds"); history is kept.

**4. The agent sees each fact's age and source**: `refund_window: 30 days
(as of 2026-10-14, operator)`, so it can weigh a fact against what the user
says now.

### How it is recorded

- Whether a write is a proposal is decided when it is written and stored in
  the event (`WorldFact.proposed`). Rebuilding the projections from the log
  then gives the same answer, whatever the setting is at rebuild time. A
  `WorldFact` logged before this decision has no such field; it is treated as
  a proposal when its source ranks 1 or lower.
- An approval (`WorldFactConfirmed`) carries a full copy of the fact. It
  stands on its own after `forget_user` deletes the conversation that
  proposed it. Unapproved proposals are deleted with their user.
- Proposals are left out of what engines extract from, so Graphiti and Mem0
  never learn an unapproved claim as something the user stated. Review
  events are bookkeeping.

## Consequences

- A deployment that relied on agents writing world facts directly now sees
  them only after approval, or must pass `world_fact_approval=False`.
- Someone has to review proposals. The world-facts board in the inspector
  ([#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25)) is where that review would happen.
- Still open in #24: a key registry (agents inventing near-duplicate keys),
  per-key default lifetimes, pinned versus query-ranked facts, conditional
  facts, bulk import from a file, and an opt-in rule for automatic promotion
  ("N independent confirmations").

## Alternatives considered

- **Keep agent writes direct, rely on the tool description.** Rejected: a
  docstring is not a control, and the failure crosses users.
- **Scope agent writes to the user, never project-wide.** Safe, but then
  nothing an agent learns can ever help other users. Proposals keep that
  path open, through a person.
- **Automatic promotion after N agreeing users.** Useful later as an opt-in;
  as a default it can be gamed by a few coordinated accounts.
