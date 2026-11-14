# ADR 0007 — Journal mode for the SQLite store

**Status:** Draft, recommendation only (no default is changed) · **Date:** 2026-11-13 · **Tracks:** [#177](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/177)

## Context

The SQLite store opens its file with Python's defaults: rollback journal (`journal_mode=delete`)
and a 5 second busy timeout. Several harnesses can share one Log (a hook in one, the MCP server
in another), and `commonmemo doctor` already warns when the file is not in WAL mode. The question:
should CommonMemo switch a file to WAL when it opens it?

## Measurement

`scripts/bench_wal.py`: N processes each open their own `Memory` on one file and record 40 sessions
of 10 messages (about 250 characters each), concurrently, with the journal mode set on the file
beforehand. Three repetitions per mode. SQLite 3.45.1, Python 3.13, 4 cores, local disk, Linux.

| Writers | Mode | Messages per second (median) | Longest single `message()` call | Errors |
|---|---|---|---|---|
| 2 (30 sessions each) | delete (current) | 217 | 0.03 s | 0 |
| 2 | wal | 594 | 0.43 s | 0 |
| 4 (40 sessions each) | delete (current) | 229 | 3.15 s | 0 |
| 4 | wal | 662 | 0.94 s | 0 |

Nothing failed in either mode, because the 5 second timeout was never reached. WAL records about
2.7 to 2.9 times as many messages per second, and with four writers the worst wait fell from about
3 s to under 1 s; in `delete` mode the worst wait grows with the number of writers and would pass the
5 s timeout with more of them or slower disks. In WAL mode the longest single call is occasionally
longer with two writers (a checkpoint), which this test does not separate from noise. Not measured:
a reader running during the writes, network file systems, Windows and macOS, and power loss.

## Options

1. **Leave the default.** No new files; the cost is the contention above.
2. **WAL when CommonMemo opens a local file path** (`PRAGMA journal_mode=WAL`, `synchronous=NORMAL`,
   an explicit `busy_timeout`). Persistent in the file, so every later opener, including the CLI, gets it.
3. **WAL as an opt-in** (`COMMONMEMO_JOURNAL=wal` or a `Memory(...)` argument).

## Recommendation

Option 2, behind the checks below, once Windows and macOS legs have been measured in the weekly drift
workflow (#168). Reasons: a much shorter worst wait with several writers, and no change to what is stored.

Costs to accept or guard against:

- Two sidecar files, `-wal` and `-shm`, next to the database. Copying the `.db` alone while a writer is
  active can miss recent events; `commonmemo export` (JSON lines) and the retention code are unaffected.
- WAL does not work on network file systems. Detect a failed switch (the pragma returns another mode) and
  keep the rollback journal, as `doctor` would report.
- `synchronous=NORMAL` can lose the last few transactions on power loss (not corrupt the file); the Log
  stays consistent and projections rebuild from it. Keep `FULL` if that is not acceptable.
- The Postgres store is unaffected.

Until accepted, `commonmemo doctor` keeps its warning and a user can switch a file with
`sqlite3 memory.db 'PRAGMA journal_mode=wal'`, which sticks.
