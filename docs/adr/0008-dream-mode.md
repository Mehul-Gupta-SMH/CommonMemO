# ADR 0008 — Dream mode: batch extraction, one reversible run

**Status:** Draft (v1 built; what is still open is listed last) · **Date:** 2026-11-15 · **Tracks:** [#159](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/159)

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
- **Stale-overwrite guard.** A fact for a key whose current value was stated after the session closed ## Choices made in v1

- **A done-set per session, not a high-water position.** One stored position would be cheaper to read, but
  sessions do not close in log order: a long session can close after a shorter one that started later, and a pass
  the budget stopped has done some sessions and not others. A position would skip the first or repeat the second.
  The mark is per session, on the Log, and `--since` only narrows the walk.
- **A session is done once it was read, whatever it yielded.** A pass marks a session the model found nothing in
  the same way as one it wrote facts from, so a re-run does not pay for it again (tested).
- **A close-time extraction that found nothing is now on the Log.** It used to leave no trace (no fact, so no
  `extract:<session>` event), and dream read the session again. `extract()` now appends a
  `Checkpoint(label="extract")` under `run_id="extract:<session>"`, bound to the user and to no session, when it
  wrote nothing; dream skips those sessions, and a second `mem.extract(session)` no longer asks the model again.
  **What cannot be detected:** a session whose close-time extraction found nothing *before* this change. Nothing
  in the Log separates it from a session that was never read, and no cheap test does either (the transcript is the
  same; only a second model call would tell). Such a session is read once by the next pass, marked, and not read
  after that. The cost is one extra read per such session, not a repeated one. Not measured: how many exist in
  any real log.
- **Engines other than the local one are refused.** Graphiti and Mem0 extract facts themselves at ingest; a pass
  over them would write the same session twice through two extractors, and `dream` rebuilds the profile by
  re-ingesting, which only the local engine does. `dream` raises rather than running with half the effect.
- **A reverted run's sessions stay done.** Reverting takes the facts back; it does not ask for them again. A
  re-read would cost the same money and most likely write the same values. To read a session again, call
  `mem.extract(session, force=True)`.
- **No review queue in v1.** A pass writes a user's own extracted facts into that user's profile under the rules
  the close-time extractor already follows: `source="extracted"`, never a world fact, forgotten values redacted
  and dropped, at most 25 per session. This is what #159 asks of version 1. Nothing a pass writes is visible to
  another user. Facts for everyone (world facts) and procedures, when a later version adds them, go in as
  **pending proposals** through the existing review path (ADR 0002, ADR 0003), never straight into memory.
- **A pass does not raise the velocity alert.** The alert exists to catch a writer nobody was watching. A pass is
  started by a person, so its facts (extracted user facts under a `dream:` run id, and only those) are left out of
  the per-source velocity count, both the recent window and the baseline. In its place the report prints one
  line: the number of facts written and the run id that reverts them. Every other writer, including the
  close-time extractor and any other source using a `dream:` run id, is counted as before (tested). Churn and
  review-backlog alerts are unchanged.

## Still open

- The learner half (lessons, procedures through review) and its measurement bar.
- The facts-against-message-windows bar on a fresh sample. The v1 measurement (70 LoCoMo sessions) gives cost
  and volume; whether the facts are correct was not measured.
