# ADR 0008 — Dream mode: batch extraction, one reversible run

**Status:** Draft (v1 built; open questions below) · **Date:** 2026-11-15 · **Tracks:** [#159](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/159)

## Context

Facts are extracted at close only when the memory has an extractor (#60). Harness hooks build memory
without one, so sessions closed there were never read. The owner's decisions on #159: the aim is both
cross-session facts and lessons or procedures, measured separately; version 1 only extracts.

## Decision

`mem.dream()` / `commonmemo dream` run the existing extractor over sessions closed since the last pass.

- **No new extraction logic.** `extraction.extract` gained `run_id=`, `write=` and `not_before=`; redaction of
  forgotten values and the facts dropped for quoting them are unchanged.
- **Dry run by default; opt-in; no scheduler.** The dry run prices the prompts it would send (4 characters a
  token) and sends nothing.
- **Once per session** through a `Checkpoint(label="dream")` per processed session, bound to its user (so
  `forget_user` removes it) and to no session (so no session's resume state changes). Sessions with an
  `extract:<session>` run are skipped. A done-set, not a bare high-water position, because a long session can
  close after a later one; `--since` is a position filter on top.
- **One run id**, `dream:<timestamp>`. `revert` is extended: for a `dream:` run id it also plans compensating
  user facts (the value before, else a `Forgotten` tombstone), only where the dream's fact is still the current
  one, and re-ingests the touched sessions because the local engine builds profiles at ingest. A reverted run's
  sessions stay done; a tombstone for a bad value keeps it from being extracted again.
- **Spend cap.** Checked before each session against what was spent plus the next estimate; the report names the
  position and the session it stopped before. A model function that carries a running `.cost` (the CLI's does,
  from `claude -p`) is trusted over the estimate.
- **Stale-overwrite guard.** A fact for a key whose current value was stated after the session closed is dropped.

## Not decided

- Engines other than local (they extract for themselves) are refused.
- A session whose close-time extraction found nothing leaves no `extract:` event, so dream reads it again.
- No review queue in v1: facts go straight into the profile like close-time extraction.
- The learner half (lessons, procedures through review) and its measurement bar.
