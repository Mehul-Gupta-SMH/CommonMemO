# ADR 0003 — Procedural memory: a task-type axis, and how a request finds its procedure

**Status:** Accepted · **Date:** 2026-10-17 (accepted 2026-10-20) · **Tracks:** [#26](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/26) (supersedes #9)

## Context

The tiers hold who a user is (UserProfile), what happened (Log, Recaps) and
what a run has in hand (Working). The structured kinds hold what is true
(world facts), what to avoid (negatives) and how tools behave (tool memory).
Nothing holds **how a type of task is done**:

- **templates** — the shape of a good result;
- **rules** — constraints the result must follow;
- **procedures** — the steps or flow that produce it;
- **exemplars** — results that were accepted, as references.

Today this lives in prompts. Prompts can't learn from runs, aren't versioned
against outcomes, and nothing can say which version of a flow worked better.

## Decision

### 1. A task-type axis

Procedural entries are keyed by **task type** (`task:<name>`), combined with
an existing scope:

- `project` — how this project does the task (the default);
- `user:<id>` — one user's own way of doing it ("my reports skip the risks section");
- `global`.

More specific wins when combined: user over project over global. A session
may name its task: `mem.session(task="weekly_report")`.

### 2. Four kinds, one shape

Each entry: `task`, `kind` (template | rule | procedure | exemplar), `name`,
`use_when` (one line, used for matching), `body`, `version`, `source`,
`status` (current | proposed | disputed | retired). Entries are events in the
Log with a projection, like every other kind, so they rebuild from the Log.

### 3. Three sources, one trust rule

| Source | Rank | Status on write |
|---|---|---|
| Files in the app's repo (`mem.load_procedures("procedures/")`) | operator (3) | current |
| Learned at consolidation from successful runs | extracted (0) | proposed |
| Written by the agent (`memory_save_procedure`) | agent (1) | proposed |

This is ADR 0002's rule applied to procedures. An entry reaches every user
doing the task, so anything not written by people is a proposal until an
operator approves it. A less trusted source never replaces a more trusted
entry. Loading a file whose body changed records a new version.

### 4. How a request finds its procedure, in order

1. **Named by the app:** `session(task=...)`. Exact, so the default.
2. **Matched by description:** the request is matched against `use_when`
   lines with the shared ranking from #21. A match above a threshold is
   pushed; one below it is left for the agent to pull.
3. **Asked for by the agent:** `memory_get_procedure(task)`, whose
   description lists the available task names.

### 5. Use

- **Push:** the template and rules for the matched task go into the
  per-turn half; a procedure is pushed as a summary (its step names), with
  the full body on pull.
- **Run:** a procedure's steps may be backed by `@mem.step`, so a stopped run
  resumes at the step where it stopped.
- **Track:** every run records `ProcedureFollowed(task, name, version)` and
  the steps it skipped. With feedback, that gives per-version outcomes.

### 6. Learning (carried over from #9)

An optional consolidation step at session close: pluggable, off by default,
behind an extra; the local engine stays deterministic. It produces:

- **decision records** (goal → plan → chosen approach → rejected alternatives → outcome);
- **lessons** merged and deduplicated across sessions;
- **proposed** procedures, rules and exemplar candidates.

Inputs are messages, tool calls and their outcomes, feedback and step
results. **Raw reasoning is never stored**: it's unevenly available across
providers, restates untrusted input, and isn't always faithful.

### 7. Privacy

Exemplars and user-scoped entries are the user's data: `forget_user` removes
them. An exemplar is shared beyond its user only once an operator has
approved it and personal data has been scrubbed from it.

## Resolved questions

Decided on 2026-10-20:

- **One home.** CommonMemo holds written procedures too, loaded from files
  (`mem.load_procedures`), next to learned ones: one source of truth for how
  a task is done, with per-version outcomes for both.
- **Explicit task type first.** `session(task=...)` when the app knows it;
  matching the request against `use_when` lines is the fallback (section 4).
- **Full outputs in exemplars.** Kept, but user-scoped until an operator
  approves the exemplar and personal data is scrubbed from it (section 7).

## Alternatives considered

- **Keep procedures in prompts or skills only.** No learning from runs, no
  per-version outcomes, no resume: the reasons this belongs in memory at all.
- **Store them as world facts.** World facts are short key/value truths;
  procedures are versioned bodies that need matching by description and
  outcome tracking.
- **Index everything for retrieval and let the model pick.** Loses the exact
  path (the app naming the task) and the trust rule; matching stays a
  fallback, not the only way in.
