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

import array
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from .events import (
    Checkpoint,
    DecisionRecorded,
    LessonLearned,
    Feedback,
    Forgotten,
    Freeze,
    MemoryEvent,
    Message,
    Negative,
    NegativeRevoked,
    ProcedureConfirmed,
    ProcedureEntry,
    ProcedureFollowed,
    ProcedureRejected,
    ProcedureRetired,
    StepFailed,
    StepFinished,
    StepStarted,
    ToolInvoked,
    Unfreeze,
    UsersLinked,
    UsersUnlinked,
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

from .ranking import terms
from .said import WINDOW, window_line
from .sqldb import connect

if TYPE_CHECKING:
    from .stores.base import Store

Row = Any
"""A result row: `sqlite3.Row` or a dict, both indexed by column name."""

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
CREATE TABLE IF NOT EXISTS user_links (
    event_id TEXT PRIMARY KEY, ts REAL NOT NULL, user_a TEXT NOT NULL, user_b TEXT NOT NULL, reason TEXT,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS user_links_a ON user_links (user_a);
CREATE INDEX IF NOT EXISTS user_links_b ON user_links (user_b);
CREATE TABLE IF NOT EXISTS freezes (
    event_id TEXT PRIMARY KEY, ts REAL NOT NULL, scope TEXT, source TEXT, reason TEXT,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS procedures (
    event_id TEXT PRIMARY KEY, scope TEXT NOT NULL, task TEXT NOT NULL, entry_kind TEXT NOT NULL,
    name TEXT NOT NULL, use_when TEXT NOT NULL, body TEXT NOT NULL, steps TEXT NOT NULL,
    version INTEGER NOT NULL, source TEXT NOT NULL, rank INTEGER NOT NULL,
    status TEXT NOT NULL,  -- current | superseded | retired | proposed | disputed | confirmed | rejected
    valid_from REAL NOT NULL, valid_to REAL, user_id TEXT, session_id TEXT, note TEXT
);
CREATE INDEX IF NOT EXISTS procedures_current ON procedures(task, status);
CREATE TABLE IF NOT EXISTS procedure_runs (
    event_id TEXT PRIMARY KEY, ts REAL NOT NULL, session_id TEXT, user_id TEXT,
    entry_id TEXT NOT NULL, task TEXT NOT NULL, entry_kind TEXT NOT NULL, name TEXT NOT NULL,
    version INTEGER NOT NULL, skipped TEXT NOT NULL, steps_tracked INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    event_id TEXT PRIMARY KEY, ts REAL NOT NULL, session_id TEXT, user_id TEXT, agent TEXT, task TEXT,
    goal TEXT NOT NULL, plan TEXT NOT NULL, chosen TEXT NOT NULL, rejected TEXT NOT NULL, outcome TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons_learned (
    event_id TEXT PRIMARY KEY, scope TEXT NOT NULL, text TEXT NOT NULL, task TEXT,
    sessions INTEGER NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL, user_id TEXT
);
CREATE TABLE IF NOT EXISTS message_windows (
    id TEXT PRIMARY KEY, wid INTEGER NOT NULL UNIQUE, session_id TEXT NOT NULL, window_no INTEGER NOT NULL,
    user_id TEXT NOT NULL, agent TEXT, first_ts REAL NOT NULL, n INTEGER NOT NULL, event_ids TEXT NOT NULL,
    UNIQUE (session_id, window_no)
);
CREATE INDEX IF NOT EXISTS message_windows_user ON message_windows (user_id, first_ts);
CREATE TABLE IF NOT EXISTS window_users (
    uid INTEGER PRIMARY KEY, user_id TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS window_terms (
    uid INTEGER NOT NULL, term TEXT NOT NULL, wid INTEGER NOT NULL,
    PRIMARY KEY (uid, term, wid)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS window_vectors (
    wid INTEGER PRIMARY KEY, redaction TEXT NOT NULL, vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS vector_probe (
    id INTEGER PRIMARY KEY, vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_meta (
    key TEXT PRIMARY KEY, value INTEGER NOT NULL
);
"""

SCHEMA_VERSION = 8
"""Bumped whenever a projection table changes shape. Projections are derived
from the log, so a file with another version drops them and rebuilds."""

_TABLES = ["tool_stats", "tool_errors", "tool_last", "world_facts", "feedback",
           "negatives", "tombstones", "checkpoints", "steps", "freezes", "user_links", "procedures", "procedure_runs",
           "decisions", "lessons_learned", "message_windows", "window_users", "window_terms",
           "window_vectors", "vector_probe"]


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


def _world_record(r: Row) -> WorldRecord:
    return WorldRecord(r["scope"], r["key"], r["value"], r["valid_from"], r["valid_until"], r["source"],
                       r["event_id"], r["status"], r["user_id"], r["note"])


@dataclass(frozen=True)
class ProcedureRecord:
    """One procedural entry (ADR 0003), as the projection holds it."""

    event_id: str
    scope: str
    task: str
    kind: str
    """`template`, `rule`, `procedure` or `exemplar`."""
    name: str
    use_when: str
    body: str
    steps: tuple[str, ...]
    version: int
    """1 for the first body that became current, then +1 per change (0 while pending)."""
    source: str
    status: str
    valid_from: float
    user_id: Optional[str] = None
    note: Optional[str] = None

    @property
    def pending(self) -> bool:
        return self.status in PENDING

    @property
    def key(self) -> str:
        """How traces name it: `task/kind/name`."""
        return f"{self.task}/{self.kind}/{self.name}"


def _procedure_record(r: Row) -> ProcedureRecord:
    return ProcedureRecord(r["event_id"], r["scope"], r["task"], r["entry_kind"], r["name"], r["use_when"],
                           r["body"], tuple(json.loads(r["steps"])), r["version"], r["source"], r["status"],
                           r["valid_from"], r["user_id"], r["note"])


@dataclass
class ProcedureOutcome:
    """How runs that used one version of an entry went (`Memory.procedure_outcomes`)."""

    task: str
    kind: str
    name: str
    version: int
    runs: int = 0
    rated: int = 0
    positive: int = 0
    negative: int = 0
    corrections: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    """Step name -> runs that never finished it (runs with tracked steps only)."""

    @property
    def approval(self) -> Optional[float]:
        return self.positive / self.rated if self.rated else None


@dataclass(frozen=True)
class DecisionRecord:
    event_id: str
    ts: float
    session_id: Optional[str]
    user_id: Optional[str]
    agent: Optional[str]
    task: Optional[str]
    goal: str
    plan: str
    chosen: str
    rejected: tuple[str, ...]
    outcome: str

    def line(self) -> str:
        out = f"{self.goal}: chose {self.chosen}"
        if self.rejected:
            out += f" over {', '.join(self.rejected)}"
        return out + (f"; {self.outcome}" if self.outcome else "")


@dataclass(frozen=True)
class LessonRecord:
    event_id: str
    scope: str
    text: str
    task: Optional[str]
    sessions: int
    """How many sessions taught it (merged repeats)."""
    first_seen: float
    last_seen: float

    def line(self) -> str:
        return self.text + (f" (seen in {self.sessions} sessions)" if self.sessions > 1 else "")


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
class FreezeRecord:
    event_id: str
    ts: float
    scope: Optional[str]
    source: Optional[str]
    reason: Optional[str]


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


def _encode(vec: Sequence[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _decode(blob: Any) -> list[float]:
    a = array.array("f")
    a.frombytes(bytes(blob))
    return a.tolist()


def _agent_key(agent: Optional[str]) -> str:
    return agent or ""


class Knowledge:
    """The structured-memory projections, in one SQLite file."""

    def __init__(self, path: str | Path = ":memory:", *, windows: bool = True) -> None:
        self.path = str(path)
        self._windows = windows
        """Index the messages of users' sessions in windows (`said.py`). Off for the scratch copies
        history and reverts replay the log into, which never search them."""
        self._conn = connect(self.path, lock="knowledge")
        self._lock = threading.Lock()
        # Several processes may start at once: one checks the schema at a time.
        with self._lock, self._conn.setup():
            self._conn.executescript(
                "CREATE TABLE IF NOT EXISTS knowledge_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
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
        c.execute("INSERT INTO tool_stats (agent, tool) VALUES (?, ?) ON CONFLICT DO NOTHING", (agent, tool))
        c.execute(
            "UPDATE tool_stats SET calls = calls + 1, failures = failures + ?, "
            "total_ms = total_ms + ?, timed_calls = timed_calls + ? WHERE agent = ? AND tool = ?",
            (int(failed), e.duration_ms or 0.0, int(e.duration_ms is not None), agent, tool),
        )
        signature = error_signature(e.error) if failed else None
        if failed:
            c.execute(
                "INSERT INTO tool_errors (agent, tool, signature, count, last_seen) VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(agent, tool, signature) DO UPDATE SET count = tool_errors.count + 1, last_seen = excluded.last_seen",
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
                "INSERT INTO tool_last VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (session_id, agent, tool) DO UPDATE SET "
                "ok = excluded.ok, signature = excluded.signature, shape = excluded.shape",
                (e.session_id, agent, tool, int(not failed), signature, dumps_shape(shape)),
            )

    def _message(self, e: Message) -> None:
        """Add a message to its session's open window (6 messages), indexing the window's terms. A
        window holds event ids, not text: the text stays in the Log, so `prune` and `forget_user`
        reach it (#65)."""
        if not self._windows or e.user_id is None or e.session_id is None:
            return
        c = self._conn
        line = window_line(e.role, e.content)
        uid = self._window_user(e.user_id, create=True)
        last = c.execute("SELECT id, wid, window_no, n, event_ids FROM message_windows WHERE session_id = ? "
                         "ORDER BY window_no DESC LIMIT 1", (e.session_id,)).fetchone()
        if last is None or last["n"] >= WINDOW:
            number = 0 if last is None else last["window_no"] + 1
            # Window ids are integers in the order windows are applied (one writer at a time), so
            # an index row is a few bytes, not the text of the session id.
            wid = c.execute("SELECT COALESCE(MAX(wid), 0) + 1 AS n FROM message_windows").fetchone()["n"]
            c.execute("INSERT INTO message_windows VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?) ON CONFLICT DO NOTHING",
                      (f"{e.session_id}#{number}", wid, e.session_id, number, e.user_id, e.agent, e.ts, json.dumps([e.id])))
        else:
            wid, ids = last["wid"], json.loads(last["event_ids"])
            if e.id in ids:
                return
            c.execute("UPDATE message_windows SET n = n + 1, event_ids = ? WHERE id = ?",
                      (json.dumps([*ids, e.id]), last["id"]))
        c.executemany("INSERT INTO window_terms VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                      [(uid, t, wid) for t in sorted(terms(line))])

    def _window_user(self, user_id: str, *, create: bool = False) -> Optional[int]:
        """The small integer a user's index rows carry in place of their id."""
        row = self._conn.execute("SELECT uid FROM window_users WHERE user_id = ?", (user_id,)).fetchone()
        if row is not None or not create:
            return None if row is None else row["uid"]
        uid = self._conn.execute("SELECT COALESCE(MAX(uid), 0) + 1 AS n FROM window_users").fetchone()["n"]
        self._conn.execute("INSERT INTO window_users VALUES (?, ?) ON CONFLICT DO NOTHING", (uid, user_id))
        return self._window_user(user_id)

    def window_candidates(self, user_id: Union[str, Sequence[str]], query_terms: Sequence[str], *, exclude_session: Optional[str] = None,
                          agent: Optional[str] = None, limit: int = 300) -> list[Row]:
        """The user's windows that share terms with a query, most shared terms first, then newest:
        `id, wid, session_id, agent, first_ts, event_ids, hits`."""
        if not query_terms:
            return []
        with self._lock:
            uids = [u for u in (self._window_user(i) for i in _ids(user_id)) if u is not None]
        if not uids:
            return []
        where, params = [f"t.term IN ({','.join('?' * len(query_terms))})"], list(query_terms)
        if exclude_session is not None:
            where.append("w.session_id != ?")
            params.append(exclude_session)
        if agent is not None:
            where.append("w.agent = ?")
            params.append(agent)
        with self._lock:
            return self._conn.execute(
                "SELECT w.id, w.wid, w.session_id, w.agent, w.first_ts, w.event_ids, COUNT(*) AS hits "
                "FROM window_terms t JOIN message_windows w ON w.wid = t.wid "
                f"WHERE t.uid IN ({','.join('?' * len(uids))}) AND {' AND '.join(where)} "
                "GROUP BY w.id, w.wid, w.session_id, w.agent, w.first_ts, w.event_ids "
                "ORDER BY hits DESC, w.first_ts DESC LIMIT ?", (*uids, *params, limit)).fetchall()

    def recent_windows(self, user_id: Union[str, Sequence[str]], *, exclude_session: Optional[str] = None, agent: Optional[str] = None,
                       limit: int = 400) -> list[Row]:
        """The user's newest windows, for rankers that don't work on shared terms (embeddings)."""
        ids = _ids(user_id)
        where, params = [f"user_id IN ({','.join('?' * len(ids))})"], list(ids)
        if exclude_session is not None:
            where.append("session_id != ?")
            params.append(exclude_session)
        if agent is not None:
            where.append("agent = ?")
            params.append(agent)
        with self._lock:
            return self._conn.execute(
                f"SELECT id, wid, session_id, agent, first_ts, event_ids FROM message_windows WHERE {' AND '.join(where)} "
                "ORDER BY first_ts DESC LIMIT ?", (*params, limit)).fetchall()

    # -- window vectors (#77): embeddings of redacted window text, a cache the Log can always rebuild

    def window_vectors(self, wids: Sequence[int], redaction: str) -> dict[int, list[float]]:
        """The stored vectors of these windows made from text redacted as `redaction` says."""
        out: dict[int, list[float]] = {}
        with self._lock:
            for i in range(0, len(wids), 500):
                chunk = list(wids[i: i + 500])
                for r in self._conn.execute(
                        f"SELECT wid, vec FROM window_vectors WHERE redaction = ? AND wid IN ({','.join('?' * len(chunk))})",
                        (redaction, *chunk)):
                    out[r["wid"]] = _decode(r["vec"])
        return out

    def store_window_vectors(self, rows: Sequence[tuple[int, str, Sequence[float]]]) -> None:
        with self._lock:
            for wid, redaction, vec in rows:
                self._conn.execute(
                    "INSERT INTO window_vectors VALUES (?, ?, ?) ON CONFLICT(wid) DO UPDATE SET "
                    "redaction = excluded.redaction, vec = excluded.vec", (wid, redaction, _encode(vec)))
            self._conn.commit()

    def vector_probe(self) -> Optional[list[float]]:
        """The embedding of the fixed probe text when the stored vectors were made: if an embedder now
        gives another, they are from another model."""
        with self._lock:
            row = self._conn.execute("SELECT vec FROM vector_probe WHERE id = 1").fetchone()
        return None if row is None else _decode(row["vec"])

    def reset_vectors(self, probe: Sequence[float]) -> None:
        """Drop every stored vector and remember `probe` as the embedder's answer for the probe text."""
        with self._lock:
            self._conn.execute("DELETE FROM window_vectors")
            self._conn.execute("DELETE FROM vector_probe")
            self._conn.execute("INSERT INTO vector_probe VALUES (1, ?)", (_encode(probe),))
            self._conn.commit()

    def prune_windows(self, store: "Store") -> int:
        """Drop windows whose messages `prune` has removed from the Log, so their terms don't outlive
        the text. Returns windows dropped."""
        dropped = 0
        with self._lock:
            rows = self._conn.execute("SELECT id, wid, user_id, event_ids FROM message_windows").fetchall()
        gone = []
        for i in range(0, len(rows), 500):
            chunk = rows[i: i + 500]
            alive = {e.id for e in store.get_events(
                [json.loads(r["event_ids"])[0] for r in chunk if json.loads(r["event_ids"])])}
            gone += [r for r in chunk if (json.loads(r["event_ids"]) or [""])[0] not in alive]
        with self._lock:
            for r in gone:
                self._conn.execute("DELETE FROM window_terms WHERE uid = ? AND wid = ?",
                                   (self._window_user(r["user_id"]), r["wid"]))
                self._conn.execute("DELETE FROM window_vectors WHERE wid = ?", (r["wid"],))
                dropped += self._conn.execute("DELETE FROM message_windows WHERE id = ?", (r["id"],)).rowcount
            self._conn.commit()
        return dropped

    def _world_insert(self, e: MemoryEvent, scope: str, key: str, value: str, valid_until: Optional[float],
                      source: str, rank: int, status: str, note: Optional[str] = None) -> None:
        self._conn.execute(
            "INSERT INTO world_facts VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (e.id, scope, key, value, e.ts, valid_until, source, rank, status, e.user_id, e.session_id, note),
        )

    def _world_current(self, scope: str, key: str) -> Optional[Row]:
        return self._conn.execute(
            "SELECT * FROM world_facts WHERE scope = ? AND key = ? AND status = 'current'", (scope, key)
        ).fetchone()

    def _world_end(self, row: Row, ts: float, status: str, note: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE world_facts SET status = ?, valid_to = ?, note = COALESCE(?, note) WHERE event_id = ?",
            (status, ts, note, row["event_id"]),
        )

    def _world(self, e: WorldFact) -> None:
        rank = source_rank(e.source)
        proposed = e.proposed if e.proposed is not None else rank <= 1
        if proposed:
            self._world_insert(e, e.scope, e.key, e.value, e.valid_until, e.source, rank, "proposed",
                               note=getattr(e, "note", None))
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

    def _procedure_current(self, scope: str, task: str, kind: str, name: str) -> Optional[Row]:
        return self._conn.execute(
            "SELECT * FROM procedures WHERE scope = ? AND task = ? AND entry_kind = ? AND name = ? "
            "AND status = 'current'", (scope, task, kind, name),
        ).fetchone()

    def _procedure_insert(self, e: MemoryEvent, scope: str, task: str, kind: str, name: str, use_when: str,
                          body: str, steps: Sequence[str], source: str, rank: int, status: str,
                          note: Optional[str] = None) -> None:
        version = 0
        if status == "current":
            row = self._conn.execute(
                "SELECT MAX(version) v FROM procedures WHERE scope = ? AND task = ? AND entry_kind = ? AND name = ?",
                (scope, task, kind, name),
            ).fetchone()
            version = (row["v"] or 0) + 1
        self._conn.execute(
            "INSERT INTO procedures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?) ON CONFLICT DO NOTHING",
            (e.id, scope, task, kind, name, use_when, body, json.dumps(list(steps)), version, source, rank,
             status, e.ts, e.user_id, e.session_id, note),
        )

    def _procedure_replace(self, e: MemoryEvent, scope: str, task: str, kind: str, name: str, use_when: str,
                           body: str, steps: Sequence[str], source: str, rank: int,
                           note: Optional[str] = None) -> None:
        """Make this the current entry: a new version, unless nothing changed."""
        current = self._procedure_current(scope, task, kind, name)
        if current is not None:
            if (current["body"], current["use_when"], json.loads(current["steps"])) == (body, use_when, list(steps)):
                return  # the same entry again (a file loaded at every start): no new version
            self._conn.execute(
                "UPDATE procedures SET status = 'superseded', valid_to = ? WHERE event_id = ?",
                (e.ts, current["event_id"]),
            )
        self._procedure_insert(e, scope, task, kind, name, use_when, body, steps, source, rank, "current", note)

    def _procedure(self, e: ProcedureEntry) -> None:
        rank = source_rank(e.source)
        proposed = e.proposed if e.proposed is not None else rank <= 1
        args = (e.scope, e.task, e.entry_kind, e.name, e.use_when, e.body, e.steps, e.source, rank)
        if proposed:
            self._procedure_insert(e, *args, "proposed", note=e.note)
            return
        current = self._procedure_current(e.scope, e.task, e.entry_kind, e.name)
        if current is not None and current["rank"] > rank:
            self._procedure_insert(e, *args, "disputed", note=f"conflicts with a {current['source']} entry")
            return
        self._procedure_replace(e, *args, note=e.note)

    def _procedure_confirmed(self, e: ProcedureConfirmed) -> None:
        self._conn.execute(
            "UPDATE procedures SET status = 'confirmed' WHERE event_id = ? AND status IN ('proposed', 'disputed')",
            (e.entry_id,),
        )
        self._procedure_replace(e, e.scope, e.task, e.entry_kind, e.name, e.use_when, e.body, e.steps,
                                "confirmed", 3, note=f"confirmed {e.entry_id}")

    def _procedure_rejected(self, e: ProcedureRejected) -> None:
        self._conn.execute(
            "UPDATE procedures SET status = 'rejected', note = ? WHERE event_id = ? AND status IN ('proposed', 'disputed')",
            (e.reason, e.entry_id),
        )

    def _procedure_retired(self, e: ProcedureRetired) -> None:
        current = self._procedure_current(e.scope, e.task, e.entry_kind, e.name)
        if current is not None and source_rank(e.source) >= current["rank"]:
            self._conn.execute(
                "UPDATE procedures SET status = 'retired', valid_to = ?, note = COALESCE(?, note) WHERE event_id = ?",
                (e.ts, e.reason, current["event_id"]),
            )

    def _procedure_followed(self, e: ProcedureFollowed) -> None:
        self._conn.execute(
            "INSERT INTO procedure_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (e.id, e.ts, e.session_id, e.user_id, e.entry_id, e.task, e.entry_kind, e.name, e.version,
             json.dumps(list(e.skipped)), int(e.steps_tracked)),
        )

    def _decision(self, e: DecisionRecorded) -> None:
        self._conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (e.id, e.ts, e.session_id, e.user_id, e.agent, e.task, e.goal, e.plan, e.chosen,
             json.dumps(list(e.rejected)), e.outcome),
        )

    def _lesson(self, e: LessonLearned) -> None:
        if e.same_as is not None:
            n = self._conn.execute(
                "UPDATE lessons_learned SET sessions = sessions + 1, last_seen = ? WHERE event_id = ?",
                (e.ts, e.same_as),
            ).rowcount
            if n:
                return  # merged; a lesson whose original is gone (forgotten) starts over below
        self._conn.execute(
            "INSERT INTO lessons_learned VALUES (?, ?, ?, ?, 1, ?, ?, ?) ON CONFLICT DO NOTHING",
            (e.id, e.scope, e.text, e.task, e.ts, e.ts, e.user_id),
        )

    def _feedback(self, e: Feedback) -> None:
        self._conn.execute(
            'INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING',
            (e.id, e.ts, e.user_id, e.agent, e.session_id, e.feedback_kind, e.rating, e.comment,
             e.wrong, e.right, e.target_event_id),
        )

    def _negative(self, e: Negative) -> None:
        self._conn.execute(
            "INSERT INTO negatives VALUES (?, ?, ?, ?, ?, ?, 1, ?) ON CONFLICT DO NOTHING",
            (e.id, e.ts, e.scope, e.statement, e.reason, e.source, e.user_id),
        )

    def _freeze(self, e: Freeze) -> None:
        self._conn.execute(
            "INSERT INTO freezes VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT DO NOTHING", (e.id, e.ts, e.scope, e.source, e.reason)
        )

    def _linked(self, e: UsersLinked) -> None:
        self._conn.execute("INSERT INTO user_links VALUES (?, ?, ?, ?, ?, 1) ON CONFLICT DO NOTHING",
                           (e.id, e.ts, e.user_a, e.user_b, e.reason))

    def _unlinked(self, e: UsersUnlinked) -> None:
        self._conn.execute("UPDATE user_links SET active = 0 WHERE event_id = ?", (e.link_id,))

    def _unfreeze(self, e: Unfreeze) -> None:
        self._conn.execute("UPDATE freezes SET active = 0 WHERE event_id = ?", (e.freeze_id,))

    def _revoke(self, e: NegativeRevoked) -> None:
        self._conn.execute("UPDATE negatives SET active = 0 WHERE event_id = ?", (e.negative_id,))

    def _forgotten(self, e: Forgotten) -> None:
        if e.user_id is None:
            return
        self._conn.execute(
            "INSERT INTO tombstones VALUES (?, ?, ?, ?) ON CONFLICT (user_id, key) DO UPDATE SET "
            "pattern = excluded.pattern, ts = excluded.ts", (e.user_id, e.key, e.pattern, e.ts)
        )

    def _checkpoint(self, e: Checkpoint) -> None:
        if e.session_id is None:
            return
        self._conn.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
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

    def tool_profiles(self, agent: Optional[str] = None) -> dict[str, ToolProfile]:
        """`{tool: profile}` for every tool, in two queries however many tools there are.
        Each equals `tool_profile(tool, agent)`."""
        where, params = ("WHERE agent = ?", (_agent_key(agent),)) if agent is not None else ("", ())
        with self._lock:
            stats = self._conn.execute(
                "SELECT tool, SUM(calls) c, SUM(failures) f, SUM(total_ms) ms, SUM(timed_calls) t "
                f"FROM tool_stats {where} GROUP BY tool ORDER BY tool", params
            ).fetchall()
            rows = self._conn.execute(
                "SELECT tool, signature, SUM(count) n, MAX(last_seen) seen, MAX(fix) fix "
                f"FROM tool_errors {where} GROUP BY tool, signature ORDER BY tool, n DESC", params
            ).fetchall()
        errors: dict[str, list[ToolError]] = {}
        for r in rows:
            errors.setdefault(r["tool"], []).append(ToolError(r["signature"], r["n"], r["seen"], r["fix"]))
        return {
            r["tool"]: ToolProfile(tool=r["tool"], agent=agent, calls=r["c"] or 0, failures=r["f"] or 0,
                                   total_ms=r["ms"] or 0.0, timed_calls=r["t"] or 0, errors=errors.get(r["tool"], []))
            for r in stats
        }

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
        best: dict[str, Row] = {}
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

    def user_links(self) -> list[tuple[str, str, str, Optional[str]]]:
        """Active links, oldest first: `(link id, user_a, user_b, reason)`."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM user_links WHERE active = 1 ORDER BY ts, event_id").fetchall()
        return [(r["event_id"], r["user_a"], r["user_b"], r["reason"]) for r in rows]

    def linked_users(self, user_id: str) -> list[str]:
        """`user_id` and every id reachable from it by active links (one person), `user_id` first, the rest sorted."""
        with self._lock:
            rows = self._conn.execute("SELECT user_a, user_b FROM user_links WHERE active = 1").fetchall()
        if not rows:
            return [user_id]
        near: dict[str, set[str]] = {}
        for r in rows:
            near.setdefault(r["user_a"], set()).add(r["user_b"])
            near.setdefault(r["user_b"], set()).add(r["user_a"])
        seen, todo = {user_id}, [user_id]
        while todo:
            for other in near.get(todo.pop(), ()):
                if other not in seen:
                    seen.add(other)
                    todo.append(other)
        return [user_id, *sorted(seen - {user_id})]

    def freezes(self) -> list[FreezeRecord]:
        """Active freezes, oldest first."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM freezes WHERE active = 1 ORDER BY ts").fetchall()
        return [FreezeRecord(r["event_id"], r["ts"], r["scope"], r["source"], r["reason"]) for r in rows]

    def frozen(self, scope: str, source: str) -> Optional[FreezeRecord]:
        """The active freeze covering a write to `scope` from `source`, if any."""
        for f in self.freezes():
            if (f.scope is None or f.scope == scope) and (f.source is None or f.source == source):
                return f
        return None

    def all_negatives(self) -> list[NegativeRecord]:
        """Every active negative, in every scope."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM negatives WHERE active = 1 ORDER BY ts").fetchall()
        return [NegativeRecord(r["event_id"], r["ts"], r["scope"], r["statement"], r["reason"], r["source"])
                for r in rows]

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

    # -- procedural memory (ADR 0003) ----------------------------------------

    def procedure_entries(self, scopes: Sequence[str], task: Optional[str] = None) -> list[ProcedureRecord]:
        """Current entries in `scopes` (for `task`, or every task), with the
        most specific scope winning per `(task, kind, name)`: later scopes in
        `scopes` are more specific."""
        if not scopes:
            return []
        marks = ",".join("?" * len(scopes))
        sql = f"SELECT * FROM procedures WHERE status = 'current' AND scope IN ({marks})"
        params: list[Any] = list(scopes)
        if task is not None:
            sql += " AND task = ?"
            params.append(task)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        order = {s: i for i, s in enumerate(scopes)}
        best: dict[tuple[str, str, str], Row] = {}
        for r in sorted(rows, key=lambda r: order[r["scope"]]):
            best[(r["task"], r["entry_kind"], r["name"])] = r
        return [_procedure_record(r) for r in sorted(best.values(), key=lambda r: (r["task"], r["valid_from"]))]

    def procedure_entry(self, event_id: str) -> Optional[ProcedureRecord]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM procedures WHERE event_id = ?", (event_id,)).fetchone()
        return _procedure_record(r) if r else None

    def pending_procedures(self, task: Optional[str] = None) -> list[ProcedureRecord]:
        """Proposed and disputed entries, oldest first."""
        sql = f"SELECT * FROM procedures WHERE status IN ({','.join('?' * len(PENDING))})"
        params: list[Any] = list(PENDING)
        if task is not None:
            sql += " AND task = ?"
            params.append(task)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY valid_from", params).fetchall()
        return [_procedure_record(r) for r in rows]

    def procedure_versions(self, scope: str, task: str, kind: str, name: str) -> list[ProcedureRecord]:
        """Every version that was current, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM procedures WHERE scope = ? AND task = ? AND entry_kind = ? AND name = ? "
                "AND version > 0 ORDER BY version", (scope, task, kind, name),
            ).fetchall()
        return [_procedure_record(r) for r in rows]

    def procedure_outcomes(self, task: Optional[str] = None) -> list[ProcedureOutcome]:
        """Per entry version: the runs that used it, and the feedback given in
        those runs' sessions."""
        sql = "SELECT * FROM procedure_runs" + (" WHERE task = ?" if task is not None else "") + " ORDER BY ts"
        with self._lock:
            runs = self._conn.execute(sql, (task,) if task is not None else ()).fetchall()
            sessions = {r["session_id"] for r in runs if r["session_id"]}
            feedback: dict[str, list[Row]] = {}
            if sessions:
                marks = ",".join("?" * len(sessions))
                for f in self._conn.execute(f"SELECT * FROM feedback WHERE session_id IN ({marks})",
                                            tuple(sessions)).fetchall():
                    feedback.setdefault(f["session_id"], []).append(f)
        out: dict[tuple[str, str, str, int], ProcedureOutcome] = {}
        counted: set[tuple[Any, ...]] = set()
        for r in runs:
            ident = (r["task"], r["entry_kind"], r["name"], r["version"])
            if (ident, r["session_id"]) in counted and r["session_id"] is not None:
                continue  # one run per session, however often it was pushed
            counted.add((ident, r["session_id"]))
            o = out.setdefault(ident, ProcedureOutcome(*ident))
            o.runs += 1
            if r["steps_tracked"]:
                for step in json.loads(r["skipped"]):
                    o.skipped[step] = o.skipped.get(step, 0) + 1
            for f in feedback.get(r["session_id"], []):
                if f["rating"] is not None:
                    o.rated += 1
                    o.positive += f["rating"] > 0
                    o.negative += f["rating"] < 0
                o.corrections += f["kind"] == "correction"
        return sorted(out.values(), key=lambda o: (o.task, o.kind, o.name, o.version))

    # -- learned at consolidation (#26, phase 2) --------------------------------

    def decisions(self, *, user_id: Optional[str] = None, agent: Optional[str] = None,
                  session_id: Optional[str] = None) -> list[DecisionRecord]:
        """Decision records, newest first."""
        where, params = [], []
        for col, val in (("user_id", user_id), ("agent", agent), ("session_id", session_id)):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        sql = "SELECT * FROM decisions" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY ts DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [DecisionRecord(r["event_id"], r["ts"], r["session_id"], r["user_id"], r["agent"], r["task"],
                               r["goal"], r["plan"], r["chosen"], tuple(json.loads(r["rejected"])), r["outcome"])
                for r in rows]

    def learned_lessons(self, scopes: Sequence[str]) -> list[LessonRecord]:
        """Lessons in `scopes`, the most often seen first, then the newest."""
        if not scopes:
            return []
        marks = ",".join("?" * len(scopes))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM lessons_learned WHERE scope IN ({marks}) ORDER BY sessions DESC, last_seen DESC",
                tuple(scopes),
            ).fetchall()
        return [LessonRecord(r["event_id"], r["scope"], r["text"], r["task"], r["sessions"], r["first_seen"],
                             r["last_seen"]) for r in rows]

    def all_procedures(self) -> list[ProcedureRecord]:
        """Every current entry, in every scope (for reverts)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM procedures WHERE status = 'current'").fetchall()
        return [_procedure_record(r) for r in rows]

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
            self._conn.execute("BEGIN IMMEDIATE")  # all of it or none of it
            try:
                n = self._forget(user_id, session_ids)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return n

    def _forget(self, user_id: str, session_ids: Sequence[str]) -> int:
        n = self._conn.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,)).rowcount
        n += self._conn.execute("DELETE FROM user_links WHERE user_a = ? OR user_b = ?", (user_id, user_id)).rowcount
        self._conn.execute("DELETE FROM window_vectors WHERE wid IN (SELECT wid FROM message_windows WHERE user_id = ?)",
                           (user_id,))
        uid = self._window_user(user_id)
        if uid is not None:
            n += self._conn.execute("DELETE FROM window_terms WHERE uid = ?", (uid,)).rowcount
            n += self._conn.execute("DELETE FROM window_users WHERE uid = ?", (uid,)).rowcount
        n += self._conn.execute("DELETE FROM message_windows WHERE user_id = ?", (user_id,)).rowcount
        n += self._conn.execute(
            "DELETE FROM world_facts WHERE user_id = ? AND status IN ('proposed', 'disputed', 'rejected')",
            (user_id,),
        ).rowcount
        n += self._conn.execute(
            "DELETE FROM negatives WHERE scope = ? OR user_id = ?", (f"user:{user_id}", user_id)
        ).rowcount
        n += self._conn.execute("DELETE FROM tombstones WHERE user_id = ?", (user_id,)).rowcount
        # The user's own entries, their exemplars, and what they proposed;
        # an approved copy (`ProcedureConfirmed`, scrubbed) is the
        # operator's and stays.
        n += self._conn.execute(
            "DELETE FROM procedures WHERE scope = ? OR (user_id = ? AND (entry_kind = 'exemplar' "
            "OR status IN ('proposed', 'disputed', 'rejected')))", (f"user:{user_id}", user_id),
        ).rowcount
        n += self._conn.execute("DELETE FROM procedure_runs WHERE user_id = ?", (user_id,)).rowcount
        n += self._conn.execute("DELETE FROM decisions WHERE user_id = ?", (user_id,)).rowcount
        n += self._conn.execute(
            "DELETE FROM lessons_learned WHERE scope = ? OR user_id = ?", (f"user:{user_id}", user_id)
        ).rowcount
        if session_ids:
            marks = ",".join("?" * len(session_ids))
            for table in ("checkpoints", "steps", "tool_last"):
                n += self._conn.execute(
                    f"DELETE FROM {table} WHERE session_id IN ({marks})", tuple(session_ids)
                ).rowcount
        return n

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _ids(user_id: Union[str, Sequence[str]]) -> Sequence[str]:
    return [user_id] if isinstance(user_id, str) else user_id


def _step(r: Row) -> StepRecord:
    return StepRecord(r["session_id"], r["step_id"], r["status"], r["result"], bool(r["replayable"]),
                      r["error"], bool(r["side_effects"]), r["started_at"], r["finished_at"])


_HANDLERS = {
    ToolInvoked: Knowledge._tool,
    WorldFact: Knowledge._world,
    WorldFactConfirmed: Knowledge._world_confirmed,
    WorldFactRejected: Knowledge._world_rejected,
    WorldFactRetracted: Knowledge._world_retracted,
    Freeze: Knowledge._freeze,
    Unfreeze: Knowledge._unfreeze,
    UsersLinked: Knowledge._linked,
    UsersUnlinked: Knowledge._unlinked,
    Feedback: Knowledge._feedback,
    Negative: Knowledge._negative,
    NegativeRevoked: Knowledge._revoke,
    Forgotten: Knowledge._forgotten,
    Checkpoint: Knowledge._checkpoint,
    StepStarted: Knowledge._step_started,
    StepFinished: Knowledge._step_finished,
    StepFailed: Knowledge._step_failed,
    ProcedureEntry: Knowledge._procedure,
    ProcedureConfirmed: Knowledge._procedure_confirmed,
    ProcedureRejected: Knowledge._procedure_rejected,
    ProcedureRetired: Knowledge._procedure_retired,
    ProcedureFollowed: Knowledge._procedure_followed,
    DecisionRecorded: Knowledge._decision,
    LessonLearned: Knowledge._lesson,
    Message: Knowledge._message,
}

__all__ = [
    "Knowledge",
    "WorldRecord",
    "ProcedureRecord",
    "ProcedureOutcome",
    "DecisionRecord",
    "LessonRecord",
    "source_rank",
    "FreezeRecord",
    "FeedbackRecord",
    "FeedbackSummary",
    "NegativeRecord",
    "CheckpointRecord",
    "StepRecord",
]
