"""Projections of the event log for the structured memory kinds.

The event log stays the source of truth. Each kind here -- tool memory,
world facts, feedback, negatives and tombstones, checkpoints and steps -- is
a table kept up to date from the log, and can always be rebuilt from it
(`rebuild`). They live in the store's SQLite file.

The projections are a **consumer of the log**: they record the last log
position they applied (`applied_seq`), and `catch_up` applies everything
after it, saving the new position in the same transaction as the changes.
So a process that dies between writing an event and applying it loses
nothing -- the next `catch_up`, from any process, applies the event -- and
applying the same stretch of log twice is impossible, so nothing is counted
twice.

These kinds are exact by nature (a rule, a step result, a counted failure),
so they are not handed to a wrapped engine the way Recaps and the UserProfile
are.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .events import (
    Checkpoint,
    Feedback,
    Forgotten,
    MemoryEvent,
    Negative,
    NegativeRevoked,
    StepFailed,
    StepFinished,
    StepStarted,
    ToolInvoked,
    WorldFact,
    WorldFactConfirmed,
    WorldFactRejected,
    WorldFactRetracted,
)
from .tool_memory import (
    ToolError,
    ToolProfile,
    arguments_shape,
    dumps_shape,
    error_signature,
    shape_change,
)

if TYPE_CHECKING:
    from .stores.base import Store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_stats (
    agent TEXT NOT NULL, tool TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,
    total_ms REAL NOT NULL DEFAULT 0, timed_calls INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent, tool)
);
CREATE TABLE IF NOT EXISTS tool_errors (
    agent TEXT NOT NULL, tool TEXT NOT NULL, signature TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0, last_seen REAL NOT NULL,
    fix TEXT,
    PRIMARY KEY (agent, tool, signature)
);
CREATE TABLE IF NOT EXISTS tool_last (
    session_id TEXT NOT NULL, agent TEXT NOT NULL, tool TEXT NOT NULL,
    ok INTEGER NOT NULL, signature TEXT, shape TEXT NOT NULL,
    PRIMARY KEY (session_id, agent, tool)
);
CREATE TABLE IF NOT EXISTS world_facts (
    event_id TEXT PRIMARY KEY, scope TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    valid_from REAL NOT NULL, valid_to REAL, valid_until REAL,
    source TEXT NOT NULL, rank INTEGER NOT NULL,
    status TEXT NOT NULL,  -- current | superseded | retracted | proposed | disputed | confirmed | rejected
    user_id TEXT, session_id TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS world_current ON world_facts(scope, key, status);
CREATE TABLE IF NOT EXISTS feedback (
    event_id TEXT PRIMARY KEY, ts REAL NOT NULL,
    user_id TEXT, agent TEXT, session_id TEXT,
    kind TEXT NOT NULL, rating REAL, comment TEXT, wrong TEXT, "right" TEXT,
    target_event_id TEXT
);
CREATE TABLE IF NOT EXISTS negatives (
    event_id TEXT PRIMARY KEY, ts REAL NOT NULL, scope TEXT NOT NULL,
    statement TEXT NOT NULL, reason TEXT, source TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1, user_id TEXT
);
CREATE TABLE IF NOT EXISTS tombstones (
    user_id TEXT NOT NULL, key TEXT NOT NULL, pattern TEXT, ts REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, ts REAL NOT NULL,
    label TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS checkpoints_session ON checkpoints(session_id, ts);
CREATE TABLE IF NOT EXISTS steps (
    session_id TEXT NOT NULL, step_id TEXT NOT NULL,
    status TEXT NOT NULL, result TEXT, replayable INTEGER NOT NULL DEFAULT 1,
    error TEXT, side_effects INTEGER NOT NULL DEFAULT 0,
    started_at REAL, finished_at REAL,
    PRIMARY KEY (session_id, step_id)
);
CREATE TABLE IF NOT EXISTS knowledge_meta (
    key TEXT PRIMARY KEY, value INTEGER NOT NULL
);
"""

SCHEMA_VERSION = 2
"""Bumped whenever a projection table changes shape. Projections are derived
from the log, so a file with another version drops them and rebuilds."""

_TABLES = ["tool_stats", "tool_errors", "tool_last", "world_facts", "feedback",
           "negatives", "tombstones", "checkpoints", "steps"]


@dataclass(frozen=True)
class WorldRecord:
    scope: str
    key: str
    value: str
    valid_from: float
    """When it became current: written, or confirmed. Shown as its age."""
    valid_until: Optional[float]
    source: str
    event_id: str
    status: str = "current"
    """`current`, or for a pending fact `proposed` / `disputed`."""
    user_id: Optional[str] = None
    """Who proposed it, for a pending fact."""
    note: Optional[str] = None

    @property
    def pending(self) -> bool:
        return self.status in PENDING


PENDING = ("proposed", "disputed")

_RANKS = {"extracted": 0, "agent": 1, "tool": 2}


def source_rank(source: str) -> int:
    """How much a world fact's source is trusted: `extracted` 0, `agent` 1,
    `tool` (checked against a tool result) 2, and anything else -- `explicit`,
    `operator`, an integration's name -- 3, since application code wrote it.
    A fact can only supersede a current fact of equal or lower rank."""
    return _RANKS.get(source, 3)


def _world_record(r: sqlite3.Row) -> WorldRecord:
    return WorldRecord(r["scope"], r["key"], r["value"], r["valid_from"], r["valid_until"], r["source"],
                       r["event_id"], r["status"], r["user_id"], r["note"])


@dataclass(frozen=True)
class FeedbackRecord:
    event_id: str
    ts: float
    user_id: Optional[str]
    agent: Optional[str]
    session_id: Optional[str]
    kind: str
    rating: Optional[float]
    comment: Optional[str]
    wrong: Optional[str]
    right: Optional[str]
    target_event_id: Optional[str]

    def line(self) -> str:
        if self.kind == "correction" and self.wrong and self.right:
            return f"Corrected: not '{self.wrong}' but '{self.right}'" + (f" ({self.comment})" if self.comment else "")
        mood = "" if self.rating is None else ("liked: " if self.rating > 0 else "disliked: " if self.rating < 0 else "")
        return f"{mood}{self.comment}" if self.comment else f"{mood}rating {self.rating}"


@dataclass(frozen=True)
class NegativeRecord:
    event_id: str
    ts: float
    scope: str
    statement: str
    reason: Optional[str]
    source: str


@dataclass(frozen=True)
class CheckpointRecord:
    event_id: str
    session_id: str
    ts: float
    label: str
    data: dict[str, Any]


@dataclass(frozen=True)
class StepRecord:
    session_id: str
    step_id: str
    status: str  # "started" | "finished" | "failed"
    result: Optional[str]
    replayable: bool
    error: Optional[str]
    side_effects: bool
    started_at: Optional[float]
    finished_at: Optional[float]


@dataclass
class FeedbackSummary:
    count: int = 0
    rated: int = 0
    positive: int = 0
    negative: int = 0
    corrections: int = 0
    recent: list[FeedbackRecord] = field(default_factory=list)

    @property
    def approval(self) -> Optional[float]:
        return self.positive / self.rated if self.rated else None


def _agent_key(agent: Optional[str]) -> str:
    return agent or ""


class Knowledge:
    """The structured-memory projections, in one SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS knowledge_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
            row = self._conn.execute("SELECT value FROM knowledge_meta WHERE key = 'schema'").fetchone()
            if row is None or row["value"] != SCHEMA_VERSION:
                for table in _TABLES:
                    self._conn.execute(f"DROP TABLE IF EXISTS {table}")
                # No position: the next catch_up recomputes everything from the log.
                self._conn.execute("DELETE FROM knowledge_meta WHERE key = 'applied_seq'")
                self._conn.execute(
                    "INSERT INTO knowledge_meta (key, value) VALUES ('schema', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (SCHEMA_VERSION,),
                )
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- applying events ----------------------------------------------------

    def catch_up(self, store: "Store") -> int:
        """Apply every event the log holds past `applied_seq`; return how
        many were applied. With no recorded position (a new file, or after
        `rebuild`), every projection is recomputed from the whole log."""
        while True:
            # Read the log first, holding no lock: reading it while holding
            # the write lock could deadlock with a writer on the store.
            seen = self.applied_seq
            batch = store.events_after(seen or 0)
            with self._lock:
                # Take the write lock, then re-read the position: another
                # process may have applied part of the batch meanwhile.
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    row = self._conn.execute(
                        "SELECT value FROM knowledge_meta WHERE key = 'applied_seq'"
                    ).fetchone()
                    if (row is None and seen is not None) or (row is not None and row["value"] < (seen or 0)):
                        self._conn.rollback()
                        continue  # rebuilt meanwhile: start over from the whole log
                    if row is None:
                        for table in _TABLES:
                            self._conn.execute(f"DELETE FROM {table}")
                    after = 0 if row is None else row["value"]
                    applied = 0
                    for seq, event in batch:
                        if seq <= after:
                            continue
                        handler = _HANDLERS.get(type(event))
                        if handler is not None:
                            handler(self, event)
                        after, applied = seq, applied + 1
                    self._conn.execute(
                        "INSERT INTO knowledge_meta (key, value) VALUES ('applied_seq', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (after,),
                    )
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
            return applied

    @property
    def applied_seq(self) -> Optional[int]:
        """The last log position applied, or None before the first catch-up."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM knowledge_meta WHERE key = 'applied_seq'").fetchone()
        return None if row is None else row["value"]

    def rebuild(self, store: "Store") -> None:
        """Recompute every projection from the whole log."""
        with self._lock:
            self._conn.execute("DELETE FROM knowledge_meta WHERE key = 'applied_seq'")
            self._conn.commit()
        self.catch_up(store)

    def _tool(self, e: ToolInvoked) -> None:
        agent, tool, c = _agent_key(e.agent), e.name, self._conn
        failed = e.error is not None
        shape = arguments_shape(e.arguments or {})
        c.execute("INSERT OR IGNORE INTO tool_stats (agent, tool) VALUES (?, ?)", (agent, tool))
        c.execute(
            "UPDATE tool_stats SET calls = calls + 1, failures = failures + ?, "
            "total_ms = total_ms + ?, timed_calls = timed_calls + ? WHERE agent = ? AND tool = ?",
            (int(failed), e.duration_ms or 0.0, int(e.duration_ms is not None), agent, tool),
        )
        signature = error_signature(e.error) if failed else None
        if failed:
            c.execute(
                "INSERT INTO tool_errors (agent, tool, signature, count, last_seen) VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(agent, tool, signature) DO UPDATE SET count = count + 1, last_seen = excluded.last_seen",
                (agent, tool, signature, e.ts),
            )
        if e.session_id is not None:
            prev = c.execute(
                "SELECT ok, signature, shape FROM tool_last WHERE session_id = ? AND agent = ? AND tool = ?",
                (e.session_id, agent, tool),
            ).fetchone()
            if prev is not None and not prev["ok"] and not failed:
                # The previous call of this tool in this session failed and
                # this one worked: the shape difference is the fix.
                change = shape_change(json.loads(prev["shape"]), shape) or "retrying with the same argument shapes"
                c.execute(
                    "UPDATE tool_errors SET fix = ? WHERE agent = ? AND tool = ? AND signature = ?",
                    (change, agent, tool, prev["signature"]),
                )
            c.execute(
                "INSERT OR REPLACE INTO tool_last VALUES (?, ?, ?, ?, ?, ?)",
                (e.session_id, agent, tool, int(not failed), signature, dumps_shape(shape)),
            )

    def _world_insert(self, e: MemoryEvent, scope: str, key: str, value: str, valid_until: Optional[float],
                      source: str, rank: int, status: str, note: Optional[str] = None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO world_facts VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
            (e.id, scope, key, value, e.ts, valid_until, source, rank, status, e.user_id, e.session_id, note),
        )

    def _world_current(self, scope: str, key: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM world_facts WHERE scope = ? AND key = ? AND status = 'current'", (scope, key)
        ).fetchone()

    def _world_end(self, row: sqlite3.Row, ts: float, status: str, note: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE world_facts SET status = ?, valid_to = ?, note = COALESCE(?, note) WHERE event_id = ?",
            (status, ts, note, row["event_id"]),
        )

    def _world(self, e: WorldFact) -> None:
        rank = source_rank(e.source)
        proposed = e.proposed if e.proposed is not None else rank <= 1
        if proposed:
            self._world_insert(e, e.scope, e.key, e.value, e.valid_until, e.source, rank, "proposed")
            return
        current = self._world_current(e.scope, e.key)
        if current is not None and current["rank"] > rank:
            # A less trusted source cannot overwrite a more trusted fact; it
            # waits for review instead of silently losing or silently winning.
            self._world_insert(e, e.scope, e.key, e.value, e.valid_until, e.source, rank, "disputed",
                               note=f"conflicts with a {current['source']} fact")
            return
        if current is not None:
            self._world_end(current, e.ts, "superseded")
        self._world_insert(e, e.scope, e.key, e.value, e.valid_until, e.source, rank, "current")

    def _world_confirmed(self, e: WorldFactConfirmed) -> None:
        self._conn.execute(
            "UPDATE world_facts SET status = 'confirmed' WHERE event_id = ? AND status IN ('proposed', 'disputed')",
            (e.fact_id,),
        )
        current = self._world_current(e.scope, e.key)
        if current is not None:
            self._world_end(current, e.ts, "superseded")
        # An operator's approval: the fact now carries an operator's rank.
        self._world_insert(e, e.scope, e.key, e.value, e.valid_until, "confirmed", 3, "current",
                           note=f"confirmed {e.fact_id}")

    def _world_rejected(self, e: WorldFactRejected) -> None:
        self._conn.execute(
            "UPDATE world_facts SET status = 'rejected', note = ? WHERE event_id = ? AND status IN ('proposed', 'disputed')",
            (e.reason, e.fact_id),
        )

    def _world_retracted(self, e: WorldFactRetracted) -> None:
        current = self._world_current(e.scope, e.key)
        if current is not None and source_rank(e.source) >= current["rank"]:
            self._world_end(current, e.ts, "retracted", note=e.reason)

    def _feedback(self, e: Feedback) -> None:
        self._conn.execute(
            'INSERT OR IGNORE INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (e.id, e.ts, e.user_id, e.agent, e.session_id, e.feedback_kind, e.rating, e.comment,
             e.wrong, e.right, e.target_event_id),
        )

    def _negative(self, e: Negative) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO negatives VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
            (e.id, e.ts, e.scope, e.statement, e.reason, e.source, e.user_id),
        )

    def _revoke(self, e: NegativeRevoked) -> None:
        self._conn.execute("UPDATE negatives SET active = 0 WHERE event_id = ?", (e.negative_id,))

    def _forgotten(self, e: Forgotten) -> None:
        if e.user_id is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO tombstones VALUES (?, ?, ?, ?)", (e.user_id, e.key, e.pattern, e.ts)
        )

    def _checkpoint(self, e: Checkpoint) -> None:
        if e.session_id is None:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO checkpoints VALUES (?, ?, ?, ?, ?)",
            (e.id, e.session_id, e.ts, e.label, json.dumps(e.data, default=str, sort_keys=True)),
        )

    def _step_started(self, e: StepStarted) -> None:
        if e.session_id is None:
            return
        self._conn.execute(
            "INSERT INTO steps (session_id, step_id, status, side_effects, started_at) VALUES (?, ?, 'started', ?, ?) "
            "ON CONFLICT(session_id, step_id) DO UPDATE SET status = 'started', error = NULL, "
            "side_effects = excluded.side_effects, started_at = excluded.started_at",
            (e.session_id, e.step_id, int(e.side_effects), e.ts),
        )

    def _step_finished(self, e: StepFinished) -> None:
        if e.session_id is None:
            return
        self._conn.execute(
            "UPDATE steps SET status = 'finished', result = ?, replayable = ?, finished_at = ? "
            "WHERE session_id = ? AND step_id = ?",
            (e.result, int(e.replayable), e.ts, e.session_id, e.step_id),
        )

    def _step_failed(self, e: StepFailed) -> None:
        if e.session_id is None:
            return
        self._conn.execute(
            "UPDATE steps SET status = 'failed', error = ?, finished_at = ? WHERE session_id = ? AND step_id = ?",
            (e.error, e.ts, e.session_id, e.step_id),
        )

    # -- tool memory --------------------------------------------------------

    def tool_profile(self, tool: str, agent: Optional[str] = None) -> ToolProfile:
        """The tool's profile for `agent`, or across all agents when None."""
        where, params = ("tool = ? AND agent = ?", (tool, _agent_key(agent))) if agent is not None else ("tool = ?", (tool,))
        with self._lock:
            s = self._conn.execute(
                f"SELECT SUM(calls) c, SUM(failures) f, SUM(total_ms) ms, SUM(timed_calls) t FROM tool_stats WHERE {where}",
                params,
            ).fetchone()
            rows = self._conn.execute(
                f"SELECT signature, SUM(count) n, MAX(last_seen) seen, MAX(fix) fix FROM tool_errors WHERE {where} "
                "GROUP BY signature ORDER BY n DESC",
                params,
            ).fetchall()
        return ToolProfile(
            tool=tool,
            agent=agent,
            calls=s["c"] or 0,
            failures=s["f"] or 0,
            total_ms=s["ms"] or 0.0,
            timed_calls=s["t"] or 0,
            errors=[ToolError(r["signature"], r["n"], r["seen"], r["fix"]) for r in rows],
        )

    def tools(self, agent: Optional[str] = None) -> list[str]:
        with self._lock:
            if agent is None:
                rows = self._conn.execute("SELECT DISTINCT tool FROM tool_stats ORDER BY tool").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT tool FROM tool_stats WHERE agent = ? ORDER BY tool", (_agent_key(agent),)
                ).fetchall()
        return [r["tool"] for r in rows]

    # -- world memory -------------------------------------------------------

    def world_facts(self, scopes: Sequence[str], *, now: Optional[float] = None,
                    include_stale: bool = False) -> list[WorldRecord]:
        """Current facts in `scopes` (later scopes win on equal keys).
        Facts past `valid_until` are left out unless `include_stale`.
        Pending facts are never included (see `pending_world_facts`)."""
        if not scopes:
            return []
        now = time.time() if now is None else now
        marks = ",".join("?" * len(scopes))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM world_facts WHERE status = 'current' AND scope IN ({marks})", tuple(scopes)
            ).fetchall()
        order = {s: i for i, s in enumerate(scopes)}
        best: dict[str, sqlite3.Row] = {}
        for r in sorted(rows, key=lambda r: order[r["scope"]]):
            if not include_stale and r["valid_until"] is not None and r["valid_until"] <= now:
                continue
            best[r["key"]] = r
        return [_world_record(r) for r in sorted(best.values(), key=lambda r: r["key"])]

    def pending_world_facts(self, scopes: Optional[Sequence[str]] = None, *, user_id: Optional[str] = None,
                            session_id: Optional[str] = None) -> list[WorldRecord]:
        """Proposed and disputed facts, oldest first: all of them, or only
        those proposed by `user_id` or in `session_id` when either is given."""
        where, params = [f"status IN ({','.join('?' * len(PENDING))})"], list(PENDING)
        if scopes is not None:
            if not scopes:
                return []
            where.append(f"scope IN ({','.join('?' * len(scopes))})")
            params.extend(scopes)
        who = []
        if user_id is not None:
            who.append("user_id = ?")
            params.append(user_id)
        if session_id is not None:
            who.append("session_id = ?")
            params.append(session_id)
        if who:
            where.append("(" + " OR ".join(who) + ")")
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM world_facts WHERE " + " AND ".join(where) + " ORDER BY valid_from", params
            ).fetchall()
        return [_world_record(r) for r in rows]

    def scopes_in_use(self) -> list[str]:
        """Every scope that holds a current world fact or an active negative."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope FROM world_facts WHERE status = 'current' "
                "UNION SELECT scope FROM negatives WHERE active = 1 ORDER BY 1"
            ).fetchall()
        return [r["scope"] for r in rows]

    def world_fact(self, event_id: str) -> Optional[WorldRecord]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM world_facts WHERE event_id = ?", (event_id,)).fetchone()
        return _world_record(r) if r else None

    def world_history(self, scope: str, key: str) -> list[tuple[str, float, Optional[float]]]:
        """`(value, valid_from, valid_to)` for every value that was current."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT value, valid_from, valid_to FROM world_facts WHERE scope = ? AND key = ? "
                "AND status IN ('current', 'superseded', 'retracted') ORDER BY valid_from",
                (scope, key),
            ).fetchall()
        return [(r["value"], r["valid_from"], r["valid_to"]) for r in rows]

    # -- feedback -----------------------------------------------------------

    def _feedback_rows(self, user_id: Optional[str], agent: Optional[str], limit: Optional[int]) -> list[FeedbackRecord]:
        where, params = [], []
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if agent is not None:
            where.append("agent = ?")
            params.append(agent)
        sql = "SELECT * FROM feedback" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY ts DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [FeedbackRecord(*[r[k] for k in r.keys()]) for r in rows]

    def feedback_summary(self, *, user_id: Optional[str] = None, agent: Optional[str] = None,
                         recent: int = 5) -> FeedbackSummary:
        rows = self._feedback_rows(user_id, agent, None)
        out = FeedbackSummary(count=len(rows), recent=rows[:recent])
        for r in rows:
            if r.rating is not None:
                out.rated += 1
                out.positive += r.rating > 0
                out.negative += r.rating < 0
            out.corrections += r.kind == "correction"
        return out

    def lessons(self, *, user_id: Optional[str], agent: Optional[str], limit: int = 5) -> list[FeedbackRecord]:
        """Feedback worth acting on next time: corrections, and anything
        negative or commented -- newest first."""
        rows = [r for r in self._feedback_rows(user_id, agent, None)
                if r.kind == "correction" or (r.rating is not None and r.rating < 0) or r.comment]
        return rows[:limit]

    # -- negatives and tombstones -------------------------------------------

    def negatives(self, scopes: Sequence[str]) -> list[NegativeRecord]:
        if not scopes:
            return []
        marks = ",".join("?" * len(scopes))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM negatives WHERE active = 1 AND scope IN ({marks}) ORDER BY ts", tuple(scopes)
            ).fetchall()
        return [NegativeRecord(r["event_id"], r["ts"], r["scope"], r["statement"], r["reason"], r["source"])
                for r in rows]

    def tombstones(self, user_id: str) -> list[tuple[str, Optional[str], float]]:
        """`(key, pattern, ts)` for each tombstone of `user_id`."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, pattern, ts FROM tombstones WHERE user_id = ?", (user_id,)
            ).fetchall()
        return [(r["key"], r["pattern"], r["ts"]) for r in rows]

    # -- checkpoints and steps ----------------------------------------------

    def latest_checkpoint(self, session_id: str) -> Optional[CheckpointRecord]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY ts DESC LIMIT 1", (session_id,)
            ).fetchone()
        if r is None:
            return None
        return CheckpointRecord(r["event_id"], r["session_id"], r["ts"], r["label"], json.loads(r["data"]))

    def step(self, session_id: str, step_id: str) -> Optional[StepRecord]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM steps WHERE session_id = ? AND step_id = ?", (session_id, step_id)
            ).fetchone()
        return _step(r) if r else None

    def steps(self, session_id: str) -> list[StepRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM steps WHERE session_id = ? ORDER BY started_at", (session_id,)
            ).fetchall()
        return [_step(r) for r in rows]

    # -- lifecycle ----------------------------------------------------------

    def forget_user(self, user_id: str, session_ids: Sequence[str]) -> int:
        """Remove everything user-specific. Tool memory holds only shapes and
        scrubbed signatures, and current world facts are not about the user,
        so both stay; world facts the user's conversations proposed and nobody
        approved go."""
        with self._lock:
            n = self._conn.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,)).rowcount
            n += self._conn.execute(
                "DELETE FROM world_facts WHERE user_id = ? AND status IN ('proposed', 'disputed', 'rejected')",
                (user_id,),
            ).rowcount
            n += self._conn.execute(
                "DELETE FROM negatives WHERE scope = ? OR user_id = ?", (f"user:{user_id}", user_id)
            ).rowcount
            n += self._conn.execute("DELETE FROM tombstones WHERE user_id = ?", (user_id,)).rowcount
            if session_ids:
                marks = ",".join("?" * len(session_ids))
                for table in ("checkpoints", "steps", "tool_last"):
                    n += self._conn.execute(
                        f"DELETE FROM {table} WHERE session_id IN ({marks})", tuple(session_ids)
                    ).rowcount
            self._conn.commit()
        return n

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _step(r: sqlite3.Row) -> StepRecord:
    return StepRecord(r["session_id"], r["step_id"], r["status"], r["result"], bool(r["replayable"]),
                      r["error"], bool(r["side_effects"]), r["started_at"], r["finished_at"])


_HANDLERS = {
    ToolInvoked: Knowledge._tool,
    WorldFact: Knowledge._world,
    WorldFactConfirmed: Knowledge._world_confirmed,
    WorldFactRejected: Knowledge._world_rejected,
    WorldFactRetracted: Knowledge._world_retracted,
    Feedback: Knowledge._feedback,
    Negative: Knowledge._negative,
    NegativeRevoked: Knowledge._revoke,
    Forgotten: Knowledge._forgotten,
    Checkpoint: Knowledge._checkpoint,
    StepStarted: Knowledge._step_started,
    StepFinished: Knowledge._step_finished,
    StepFailed: Knowledge._step_failed,
}

__all__ = [
    "Knowledge",
    "WorldRecord",
    "source_rank",
    "FeedbackRecord",
    "FeedbackSummary",
    "NegativeRecord",
    "CheckpointRecord",
    "StepRecord",
]
