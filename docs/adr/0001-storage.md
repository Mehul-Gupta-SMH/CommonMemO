# ADR 0001 — Storage for the memory CommonMemo owns

**Status:** Proposed · **Date:** 2026-10-16 · **Tracks:** [#11](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/11), [#15](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/15), [#16](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/16), [#17](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/17)

## Question

Which kind of database should hold the memory CommonMemo owns itself: the
log, the Working tier, and the structured kinds (tool memory, world
facts, feedback, negatives, resume points)? Memory has to be *correct*
(nothing lost, nothing counted twice, nothing forgotten that the user asked to
keep, nothing kept that the user asked to forget). Agent workloads also
produce a lot of writes, which argues for NoSQL scale.

Out of scope: the engines' own stores. Graphiti brings a graph database and
Mem0 a vector store; those are the engines' choices.

## The false choice: "ACID or NoSQL"

ACID and "NoSQL" are not opposites:

- Many NoSQL databases are ACID **within a partition**. DynamoDB and MongoDB
  both offer conditional writes and transactions, and FoundationDB is fully
  ACID.
- Distributed SQL databases (CockroachDB, YugabyteDB, Spanner, AlloyDB,
  Aurora) give ACID **and** horizontal scale. Several speak the Postgres
  protocol.

So the useful questions are narrower:

1. Which writes need strong consistency, and over what scope (one session,
   one user, everything)?
2. Does the data partition cleanly?
3. What is the real write volume?
4. Which queries do we run?

## 1. What "correct memory" requires

| # | Invariant | Needs |
|---|---|---|
| I1 | An acknowledged event is never lost | durable write before ack |
| I2 | An event is stored once, however often it is delivered | idempotent insert by event id (conditional put) |
| I3 | Events of one session keep their order | per-session ordering key |
| I4 | Projections reflect the log exactly — every event applied once | atomic "append + apply", **or** idempotent, checkpointed apply from the log |
| I5 | Read-your-writes within a session (a fact remembered now is visible next turn) | strong reads on the session/user partition |
| I6 | A side-effecting step runs at most once, even with two workers resuming the same session | conditional write / lease on `(session, step)` |
| I7 | `forget_user` removes everything about the user, everywhere | deletable records, scoped by user |

Every invariant except I7's "everywhere" lives **inside one partition**: a
session, or a user. Nothing requires transactions *across* users. That single
observation decides most of this ADR: we need strong consistency **per
partition**, not globally — which every serious candidate offers.

### What today's code gets wrong (found while writing this)

- **I4 is violated.** The log (`stores/sqlite.py`) and the projections
  (`knowledge.py`) use separate connections and separate transactions. A
  crash after the event commits but before its projection applies leaves the
  projection missing that event. A retry then sees a duplicate and skips the
  apply, so the gap is permanent until `rebuild_knowledge`. → **[#15](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/15)**
- **I6 is violated.** `@mem.step` checks a step's status and then writes it
  ("check-then-act"). Two processes resuming one session can both see
  "not finished" and both run a side-effecting step. → **[#16](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/16)**

These are bugs in how the code uses the store, not choices of database, and
they have to be fixed on SQLite first. A different database fixes neither.

## 2. Access patterns

| Data | Writes | Reads | Partition key |
|---|---|---|---|
| Event log | append-only, every event | by session in order; by user (forget, consolidation); by kind | session (user secondary) |
| Working tier | state on every write; open/close | by session id, every turn | session |
| User-scoped projections (feedback, negatives, tombstones, steps, checkpoints) | small upserts | filtered reads per user/session, every turn | user / session |
| World facts | rare; supersede with history | by scope, every turn | scope |
| Tool memory | **a counter increment on every tool call** | group-by per tool, per agent | tool × agent — **shared across users** |

The read path builds context on every turn: 6–8 small reads (negatives,
facts, world, lessons, tool notes, episodes, state, resume), all within one
user's or session's partition, except tool notes (per tool) and world facts
(per scope). Latency matters more than throughput here.

## 3. Volume — an order of magnitude, to be measured ([#17](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/17))

A turn produces roughly 5–30 events: run start and finish, LLM calls, tool
calls, messages. Call it 20, at about 1–2 KB each.

| Scale | Turns/day | Events/day | Avg writes/s | Peak (×10) | Raw log/day |
|---|---|---|---|---|---|
| Team | 10k | 200k | ~2 | ~25 | ~0.3 GB |
| Product | 1M | 20M | ~230 | ~2.5k | ~30 GB |
| Platform | 100M | 2B | ~23k | ~250k | ~3 TB |

Read: a single well-run Postgres handles the first two rows on write
throughput (thousands of small inserts per second with group commit and
batching). What it needs at "Product" is **storage management**: time
partitioning plus archiving old events to object storage. "Platform" needs
horizontal write scale. These figures are estimates; [#17](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/17) replaces them with
measurements before anything is decided on them.

Two hot spots exist at any scale and in any database:

- **Tool counters:** every call of a popular tool increments the same row
  (`tool_stats`), a hot key in SQL and NoSQL alike. Fix it in the design,
  not the database: append tool outcomes, aggregate periodically or with
  sharded counters.
- **Log growth:** fix it with time partitioning and archiving, plus
  per-kind retention.

## 4. Candidates

| Option | I1–I3 | I4 | I5 | I6 | I7 | Queries (group-by, supersede) | Write scale | Ops | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **SQLite** | ✅ | ✅ in one transaction | ✅ | ✅ (single writer) | ✅ | ✅ | one process | none | **Default**: embedded, tests, single host |
| **Postgres** | ✅ | ✅ in one transaction | ✅ | ✅ `INSERT … ON CONFLICT` / row lock | ✅ | ✅ direct | one primary (+ partitioning, replicas) | everywhere | **Shared default** |
| **Distributed SQL** (CockroachDB, YugabyteDB, AlloyDB; Postgres protocol) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | horizontal | managed options | **The scale path, same SQL** |
| **DynamoDB** | ✅ conditional put | ✅ `TransactWriteItems` (≤100 items) or a stream consumer | ✅ strongly consistent reads per item | ✅ conditional write | ✅ by partition key | ⚠️ no group-by; precompute everything; design keys up front | horizontal, serverless | none (AWS only) | **Second backend** for serverless / AWS |
| **MongoDB** | ✅ unique index | ✅ multi-document transactions (replica set) | ✅ majority reads | ✅ conditional update | ✅ | ⚠️ aggregation pipeline works; supersede by convention | sharding | moderate | Alternative second backend |
| **Redis** | ⚠️ durability depends on AOF / persistence settings | ⚠️ | ✅ | ✅ `SET NX` | ✅ | ❌ | high | moderate | **Cache for the Working tier only**, never the source of truth |
| **Cassandra / ScyllaDB** | ✅ (idempotent by key) | ❌ no multi-row atomicity | ⚠️ tunable | ⚠️ lightweight transactions (Paxos), costly | ⚠️ tombstones | ❌ group-by | very high | heavy | Rejected: I4 and I6 are what memory needs most |
| **Kafka / Kinesis / EventStoreDB** | ✅ | ✅ projections as idempotent consumers | ⚠️ needs a read model elsewhere | ❌ needs another store | ❌ immutable log; deletion only via crypto-shredding | ❌ | very high | heavy | Rejected as primary; maybe an ingestion buffer at "Platform" scale |
| **Object storage (S3)** | ✅ | n/a | ❌ | ❌ | ✅ | ❌ | unlimited | none | **Archive tier** for old events |

## Decision (proposed)

**Relational and ACID, partitioned by user and session, with the scaling path
inside the Postgres protocol — plus one non-SQL backend to keep the storage
interface honest.**

1. **SQLite stays the default** (zero install, single host, tests).
2. **Postgres is the shared backend.**
   - The projections are relational (upserts, supersede with history,
     group-by), so they map directly.
   - Per-partition ACID covers I1–I6 in a single transaction.
   - The scale path doesn't leave the protocol: time-partitioned events,
     then read replicas, then distributed SQL (CockroachDB / YugabyteDB).
   - pgvector later gives the local engine semantic search without a
     second database.
3. **DynamoDB is the second backend**, for serverless and AWS deployments. It
   is also the proof that the storage interface isn't SQL-shaped. Every
   invariant has a native mechanism: conditional writes, item-level
   transactions, strongly consistent reads. MongoDB is the alternative if a
   deployment already runs it.
4. **Redis is only ever a cache** for the Working tier. Durable session state
   is what makes resuming work, so it can't live only in a cache.
5. **Old events go to object storage**, under per-kind retention. This
   addresses the volume problem whichever database is chosen.
6. **Rejected:** Cassandra (no atomic multi-row apply, expensive
   conditional writes), and Kafka as the primary store (needs a second
   store for reads, and conflicts with deletion for `forget_user`).

### Changes to the code this requires, whatever the database

- **Projections apply idempotently from the log with a checkpoint ([#15](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/15)).**
  Each projection records the last event sequence it applied, and on start
  it catches up from the log. In the same database this can be one
  transaction; across databases, or at scale, it becomes a consumer. This
  fixes I4 on every backend.
- **Leases on steps ([#16](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/16)).** A conditional write claims `(session, step)`
  before running it, which fixes I6.
- **Projections behind an interface defined by operations, not SQL ([#11](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/11)).**
  The operations: append-if-absent, increment, supersede-with-history,
  filtered read, aggregate.
- **Tool memory without a hot counter row:** append outcomes and aggregate
  ([#11](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/11)).

## What would change this decision

- [#17](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/17) measures one Postgres primary falling short of the target write rate
  even with batching and partitioning. Then distributed SQL, or DynamoDB, as
  the primary.
- Active-active multi-region writes become a requirement. Then distributed
  SQL or DynamoDB global tables.
- A deployment can't operate a database at all. Then DynamoDB, or a
  managed Postgres.
