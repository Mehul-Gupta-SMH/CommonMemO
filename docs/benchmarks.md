# Benchmarks: the memory workload ([#17](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/17))

ADR 0001 estimated the workload (about 20 events a turn at 1 to 2 KB, one Postgres primary up
to about a million turns a day, storage as the first wall) and chose a backend from those
estimates. These are the measurements. They replace the estimate table in
[ADR 0001](adr/0001-storage.md) §3.

## How

`scripts/bench_workload.py` records a synthetic agent workload exactly as the Claude Code hooks
record a real turn (`hooks.py`):
- the prompt as a user `Message`;
- the turn's push (`Session.push_turn`), which also searches the user's past messages and records a trace;
- 8 tool calls, each a `ToolInvoked` with a result of about 1,200 characters, followed by `Session.refresh`;
- the reply as an assistant `Message`.

A session closes every 20 turns. Tools come from 40, drawn by popularity with a Zipf-like skew: `0` is uniform, `1.2` is a realistic long tail, and `3` sends most calls to a few tools. Each worker is a separate process with its own `Memory` and its own five users, so one user's turns are never concurrent, as with a real agent.

`--only-tools` records tool calls back to back and nothing else. It isolates the tool-counter row, and with skew `100` every call hits one row. `--cold N` times N hook invocations the way Claude Code makes them, a new `commonmemo hook` process each.

```
python scripts/bench_workload.py --store sqlite --workers 1 4 --skew 0 1.2 3 --turns 300 --cold 20
python scripts/bench_workload.py --store postgresql://postgres@127.0.0.1:5434/postgres --workers 1 4 8 --skew 0 1.2 3 --turns 300 --cold 20
python scripts/bench_workload.py --store ... --only-tools --workers 1 4 8 --skew 0 100 --turns 150
```

**Machine:**
- 4 CPUs, Linux 6.18 (x86_64), a cloud container with a virtual disk;
- Python 3.13.16, SQLite 3.45.1, PostgreSQL 16.15;
- the Postgres server runs on the same 4 CPUs as the workers.

These are one machine's numbers, so the ratios matter more than the absolute values. A real Postgres on its own host would do better on everything that is server-bound.

## The full workload

11 events a turn (2 messages, 8 tool calls, 1 trace).

| Store | Workers | Tool skew | Turns | Events/s | Turns/s | Append p50 / p99 | Context p50 / p99 | Close p50 / p99 | Bytes/turn | GB per 1M turns | Hook process p50 / p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SQLite | 1 | 0 | 300 | 92 | 8.4 | 4.8 / 10.5 ms | 14.9 / 42.4 ms | 16 / 31 ms | 32,591 | 32.6 | 301 / 465 ms |
| SQLite | 1 | 1.2 | 300 | 96 | 8.8 | 4.7 / 10.1 ms | 14.6 / 38.1 ms | 15 / 27 ms | 32,536 | 32.5 | 281 / 377 ms |
| SQLite | 1 | 3 | 300 | 113 | 10.3 | 4.4 / 9.7 ms | 13.1 / 39.7 ms | 16 / 34 ms | 32,481 | 32.5 | 276 / 398 ms |
| SQLite | 4 | 0 | 1,200 | 104 | 9.4 | 6.6 / 134.2 ms | 37.7 / 240.3 ms | 27 / 196 ms | 33,782 | 33.8 | 271 / 375 ms |
| SQLite | 4 | 1.2 | 1,200 | 103 | 9.3 | 6.7 / 135.4 ms | 38.9 / 246.6 ms | 31 / 142 ms | 33,655 | 33.7 | 310 / 383 ms |
| SQLite | 4 | 3 | 1,200 | 104 | 9.5 | 6.0 / 235.1 ms | 30.6 / 366.5 ms | 27 / 167 ms | 33,618 | 33.6 | 291 / 369 ms |
| Postgres | 1 | 0 | 300 | 24 | 2.2 | 6.0 / 13.1 ms | 57.7 / 89.5 ms | 22 / 32 ms | 41,697 | 41.7 | 550 / 733 ms |
| Postgres | 1 | 1.2 | 300 | 25 | 2.3 | 5.8 / 12.4 ms | 54.4 / 92.0 ms | 18 / 30 ms | 41,588 | 41.6 | 529 / 751 ms |
| Postgres | 1 | 3 | 300 | 47 | 4.3 | 5.6 / 12.6 ms | 32.5 / 58.2 ms | 21 / 33 ms | 41,315 | 41.3 | 496 / 662 ms |
| Postgres | 4 | 0 | 1,200 | 102 | 9.3 | 5.2 / 15.5 ms | 50.3 / 104.7 ms | 18 / 41 ms | 40,366 | 40.4 | 506 / 641 ms |
| Postgres | 4 | 1.2 | 1,200 | 99 | 9.0 | 5.7 / 16.9 ms | 50.9 / 111.2 ms | 18 / 41 ms | 40,387 | 40.4 | 531 / 588 ms |
| Postgres | 4 | 3 | 1,200 | 134 | 12.2 | 6.9 / 18.6 ms | 39.0 / 88.2 ms | 19 / 37 ms | 40,271 | 40.3 | 528 / 582 ms |
| Postgres | 8 | 0 | 2,400 | 101 | 9.2 | 11.7 / 33.0 ms | 102.8 / 201.9 ms | 31 / 77 ms | 39,810 | 39.8 | 510 / 558 ms |
| Postgres | 8 | 1.2 | 2,400 | 107 | 9.7 | 10.6 / 29.7 ms | 97.9 / 192.8 ms | 29 / 77 ms | 39,823 | 39.8 | 478 / 540 ms |
| Postgres | 8 | 3 | 2,400 | 139 | 12.7 | 12.8 / 34.0 ms | 72.6 / 164.1 ms | 30 / 67 ms | 39,581 | 39.6 | 472 / 521 ms |

## Tool calls only: the hot-counter test

| Store | Workers | Tool skew | Events/s | Append p50 / p99 |
|---|---|---|---|---|
| SQLite | 1 | 0 (40 tools) | 278 | 3.2 / 6.0 ms |
| SQLite | 1 | 100 (one tool) | 256 | 3.4 / 7.9 ms |
| SQLite | 4 | 0 | 272 | 3.3 / 8.4 ms |
| SQLite | 4 | 100 | 272 | 3.3 / 9.7 ms |
| Postgres | 1 | 0 | 213 | 4.0 / 7.6 ms |
| Postgres | 1 | 100 | 220 | 3.8 / 8.7 ms |
| Postgres | 4 | 0 | 407 | 8.9 / 16.4 ms |
| Postgres | 4 | 100 | 404 | 9.0 / 16.4 ms |
| Postgres | 8 | 0 | 405 | 18.3 / 30.7 ms |
| Postgres | 8 | 100 | 380 | 19.0 / 33.1 ms |
| Postgres, `synchronous_commit = off` | 4 | 0 | 634 | 5.5 / 12.1 ms |

## What the numbers say

1. **The first wall is the read path, not the database.** A turn issues about 1,060 SQL statements (`cProfile`, 31,878 in 30 turns on Postgres). Most come from `tool_notes`, which `push_turn` and every `refresh` call: it reads the profile of every known tool one query at a time (12,796 `tool_profile` calls in 30 turns, about 50 per call). That accounts for 65% of a Postgres turn's time. It is why Postgres is 4 times slower than SQLite with one worker (each query is a round trip, not a function call), and why a more skewed workload is faster (fewer distinct tools). Fixing that query comes before any backend decision.
2. **The tool-counter row is not a hot spot at this scale.** Sending every call to one tool costs 0 to 8% of throughput against 40 tools, at up to 8 writers (8% with one SQLite writer, 6% with 8 Postgres writers, otherwise under 1%). ADR 0001's fix (append and aggregate, or sharded counters) isn't needed yet.
3. **The write ceiling is the log's single ordered append.** On Postgres, tool calls alone level off at about 400 events a second from 4 workers on. The p50 doubles each time the workers double, which is a queue. Each append is two transactions, one for the log (under the log lock) and one for the projections, and each commit waits for the disk: turning off `synchronous_commit` lifts the ceiling to 634 (+56%). The levers, in order:
   - write a turn's events in one transaction instead of one each (group commit);
   - apply projections without a second commit per event;
   - only then, log partitioning (#52 part 4).

   SQLite is one writer, at about 270 events a second whatever the number of processes. With 4 writing processes a session close can wait up to 0.5 to 5 s for the lock (close p99 in the tool-only run).
4. **Storage: 33 KB a turn on SQLite, 40 KB on Postgres, so 33 to 42 GB per million turns.** ADR 0001 guessed about 30. By table (SQLite, 1,220 turns):
   - the event log is 90%;
   - the turn traces (`context_pushed`, about 8.6 KB each, more than one per turn) are almost half of the log;
   - the window index is 3%.

   Traces store references, not text, but a reference list that names every tool note is long. Making traces smaller, or giving them a shorter retention than the events, saves more than any storage engine.
5. **A hook costs a process.** Claude Code starts `commonmemo hook` for every event. A `UserPromptSubmit` takes 270 to 310 ms with SQLite and 470 to 550 ms with Postgres before the model sees anything: starting Python, importing, connecting and opening the memory, then the push (not profiled further). In a long-running app it is the context latency above (15 ms SQLite, 55 ms Postgres p50). A resident process (a small local server the hook talks to) would remove most of it.
6. **What this means for scale.**
   - **Within reach:** one log on this 4-CPU machine sustains about 400 small events a second, about 36 turns a second at 11 events a turn, or 3 million turns a day if the load were flat. ADR 0001's "Product" row (1 million turns a day, about 230 writes a second on average) fits.
   - **Out of reach:** the "Product" peak of about 2,500 writes a second does not fit, nor does anything past it.
   - **What changes first:** fixes 1 and 3 move these numbers more than a new backend would. ADR 0001's ranking of backends stands, and its order of work changes.

## Follow-ups

- [#81](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/81): `tool_notes` reads every tool's profile one query at a time; one grouped query instead. This is the largest single cost.
- [#82](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/82): one transaction per turn for its events, and projection updates without a commit per event.
- [#83](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/83): smaller turn traces, or a shorter retention for them.
- A resident hook process (optional), so a hook doesn't pay for a process start and a connection.
- #52 revisited:
  - the hot-counter work (part 3) is not needed at measured scale;
  - log partitioning (part 4) comes after the two fixes above;
  - the DynamoDB backend (part 2) is not justified by these numbers yet, though the storage interface (part 1) still matters for any non-SQL backend.
