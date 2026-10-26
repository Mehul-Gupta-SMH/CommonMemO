"""The SQL `Store`, on SQLite (the zero-setup default) or Postgres (ADR 0001, #11).

    SQLiteStore("mem.db")                                   # stdlib sqlite3, no install
    PostgresStore("postgresql://u@host/db?schema=shop")      # pip install "commonmemo[postgres]"

Both run the same SQL through `sqldb.Db`. On Postgres every append, claim
and settle takes the log's write lock (a transaction-scoped advisory
lock), so log positions commit in order across processes: a reader that
remembers the last position it saw never misses an event committed later
under a smaller one. That serializes writes to one log; partitioning it by
user or session is the next step when a single log's write rate is the
limit (#17).
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from ..events import MemoryEvent, event_from_dict
from ..sqldb import connect
from .base import ClaimRecord, EventCount, SessionRecord, Store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    id         TEXT NOT NULL UNIQUE,
    ts         REAL NOT NULL,
    kind       TEXT NOT NULL,
    session_id TEXT,
    user_id    TEXT,
    agent      TEXT,
    body       TEXT NOT NULL,
    run_id     TEXT
);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS events_user    ON events(user_id, seq);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT,
    agent      TEXT,
    opened_at  REAL NOT NULL,
    closed_at  REAL,
    expires_at REAL,
    state      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS claims (
    key         TEXT PRIMARY KEY,
    session_id  TEXT,
    owner       TEXT NOT NULL,
    status      TEXT NOT NULL,
    lease_until REAL
);
CREATE INDEX IF NOT EXISTS claims_session ON claims(session_id);
CREATE TABLE IF NOT EXISTS event_rollups (
    day    REAL NOT NULL,
    kind   TEXT NOT NULL,
    source TEXT NOT NULL,
    scope  TEXT NOT NULL,
    n      INTEGER NOT NULL,
    PRIMARY KEY (day, kind, source, scope)
);
"""

_DAY = 86400.0

# Per-event source and scope, for counting: not every kind has them.
_FIELD = {"sqlite": "COALESCE(json_extract(body, '$.{0}'), '')",
          "postgres": "COALESCE(body::jsonb ->> '{0}', '')"}
# The start of a bucket: ts is positive, so SQLite's truncating CAST is a floor.
_BUCKET = {"sqlite": "CAST({0} / ? AS INTEGER) * ?", "postgres": "FLOOR({0} / ?) * ?"}


def _dumps(value: Any) -> str:
    # default=str: an argument or result that is not JSON-native (a datetime,
    # a custom object) is stored as its str() rather than failing the write.
    return json.dumps(value, default=str, sort_keys=True)


class SqlStore(Store):
    """The store on any `sqldb` target: a SQLite path or a Postgres URL.
    `path` is that target; the projections and the local engine open the
    same one, so one URL is the whole setup."""

    def __init__(self, target: str | Path = ":memory:") -> None:
        self.path = str(target)
        self._conn = connect(self.path, lock="log")
        self.dialect = self._conn.dialect
        self._source, self._scope = (_FIELD[self.dialect].format(f) for f in ("source", "scope"))
        self._bucket = _BUCKET[self.dialect]
        self._lock = threading.Lock()
        schema = _SCHEMA if self.dialect == "sqlite" else _SCHEMA.replace(
            "seq        INTEGER PRIMARY KEY AUTOINCREMENT", "seq        BIGSERIAL PRIMARY KEY")
        with self._lock, self._conn.setup():
            self._conn.executescript(schema)
            if "run_id" not in self._conn.columns("events"):  # a file from before run_id existed
                self._conn.execute("ALTER TABLE events ADD COLUMN run_id TEXT")
            self._conn.execute("CREATE INDEX IF NOT EXISTS events_run ON events(run_id, seq)")
            self._conn.commit()

    @contextmanager
    def _write(self) -> Iterator[None]:
        """One transaction holding the log's write lock: all of it or none of it.
        Called with `self._lock` held."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    # -- the Log ------------------------------------------------------------

    def _insert(self, event: MemoryEvent) -> bool:
        cur = self._conn.execute(
            "INSERT INTO events (id, ts, kind, session_id, user_id, agent, body, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
            (
                event.id,
                event.ts,
                event.kind,
                event.session_id,
                event.user_id,
                event.agent,
                _dumps(event.to_dict()),
                event.run_id,
            ),
        )
        return cur.rowcount == 1

    def append(self, event: MemoryEvent) -> bool:
        # The log's write lock: positions commit in order (see the module docstring).
        with self._lock, self._write():
            new = self._insert(event)
        return new

    def append_many(self, events: Sequence[MemoryEvent]) -> list[bool]:
        # One transaction under the log's write lock: all of them or none, in order.
        with self._lock, self._write():
            return [self._insert(e) for e in events]

    def events_after(self, seq: int, *, limit: Optional[int] = None) -> list[tuple[int, MemoryEvent]]:
        # Appends hold the log's write lock, so positions commit in order and
        # "after seq" never skips an event that commits later.
        sql, params = "SELECT seq, body FROM events WHERE seq > ? ORDER BY seq", [seq]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(r["seq"], event_from_dict(json.loads(r["body"]))) for r in rows]

    def position(self, *, event_id: Optional[str] = None, at: Optional[float] = None) -> int:
        with self._lock:
            if event_id is not None:
                row = self._conn.execute("SELECT seq FROM events WHERE id = ?", (event_id,)).fetchone()
            elif at is not None:
                row = self._conn.execute("SELECT MAX(seq) AS seq FROM events WHERE ts <= ?", (at,)).fetchone()
            else:
                row = self._conn.execute("SELECT MAX(seq) AS seq FROM events").fetchone()
        return (row["seq"] or 0) if row is not None else 0

    def event_counts(self, *, since: Optional[float] = None, until: Optional[float] = None,
                     bucket: float = 86400.0) -> list[EventCount]:
        lo = float("-inf") if since is None else since
        hi = float("inf") if until is None else until
        sql = (
            f"SELECT {self._bucket.format('ts')} AS bucket, kind, {self._source} AS source, "
            f"{self._scope} AS scope, COUNT(*) AS n FROM events WHERE ts >= ? AND ts < ? GROUP BY 1, 2, 3, 4 "
            "UNION ALL "
            f"SELECT {self._bucket.format('day')} AS bucket, kind, source, scope, n FROM event_rollups "
            "WHERE day >= ? AND day < ?"
        )
        totals: dict[tuple, int] = {}
        with self._lock:
            for r in self._conn.execute(sql, (bucket, bucket, lo, hi, bucket, bucket, lo, hi)):
                k = (float(r["bucket"]), r["kind"], r["source"], r["scope"])
                totals[k] = totals.get(k, 0) + r["n"]
        return [EventCount(*k, n) for k, n in sorted(totals.items())]

    # -- claims -------------------------------------------------------------

    def claim(self, key: str, *, session_id: Optional[str], owner: str, lease_until: float,
              event: MemoryEvent) -> Optional[ClaimRecord]:
        with self._lock:
            # BEGIN IMMEDIATE takes the database's write lock before reading,
            # so a second process checking the same key waits for this one
            # to commit, then sees the claim.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT * FROM claims WHERE key = ?", (key,)).fetchone()
                if row is not None and row["status"] != "free":
                    self._conn.rollback()
                    return ClaimRecord(row["key"], row["owner"], row["status"], row["lease_until"])
                self._conn.execute(
                    "INSERT INTO claims (key, session_id, owner, status, lease_until) VALUES (?, ?, ?, 'held', ?) "
                    "ON CONFLICT(key) DO UPDATE SET session_id = excluded.session_id, owner = excluded.owner, "
                    "status = 'held', lease_until = excluded.lease_until",
                    (key, session_id, owner, lease_until),
                )
                self._insert(event)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return None

    def get_claim(self, key: str) -> Optional[ClaimRecord]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM claims WHERE key = ?", (key,)).fetchone()
        return None if row is None else ClaimRecord(row["key"], row["owner"], row["status"], row["lease_until"])

    def settle(self, key: str, *, session_id: Optional[str], status: str, event: MemoryEvent) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO claims (key, session_id, owner, status, lease_until) VALUES (?, ?, '', ?, NULL) "
                    "ON CONFLICT(key) DO UPDATE SET status = excluded.status, lease_until = NULL",
                    (key, session_id, status),
                )
                self._insert(event)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def events(
        self,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> list[MemoryEvent]:
        where, params = [], []
        if run_id is not None:
            where.append("run_id = ?")
            params.append(run_id)
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if kinds is not None:
            kinds = list(kinds)
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if since is not None:
            where.append("ts >= ?")
            params.append(since)
        sql = "SELECT body FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY seq DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [event_from_dict(json.loads(r["body"])) for r in reversed(rows)]

    def get_events(self, ids: Iterable[str]) -> list[MemoryEvent]:
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT body FROM events WHERE id IN ({','.join('?' * len(ids))}) ORDER BY seq", ids
            ).fetchall()
        return [event_from_dict(json.loads(r["body"])) for r in rows]

    # -- Working memory -----------------------------------------------------

    def _row_to_session(self, row: Any) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            agent=row["agent"],
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
            expires_at=row["expires_at"],
            state=json.loads(row["state"]),
        )

    def open_session(self, record: SessionRecord) -> SessionRecord:
        existing = self.get_session(record.session_id)
        if existing is not None:
            if existing.closed_at is not None:
                with self._lock:
                    self._conn.execute(
                        "UPDATE sessions SET closed_at = NULL WHERE session_id = ?",
                        (record.session_id,),
                    )
                    self._conn.commit()
                existing.closed_at = None
            return existing
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions "
                "(session_id, user_id, agent, opened_at, closed_at, expires_at, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (session_id) DO UPDATE SET "
                "user_id = excluded.user_id, agent = excluded.agent, opened_at = excluded.opened_at, "
                "closed_at = excluded.closed_at, expires_at = excluded.expires_at, state = excluded.state",
                (
                    record.session_id,
                    record.user_id,
                    record.agent,
                    record.opened_at,
                    record.closed_at,
                    record.expires_at,
                    _dumps(record.state),
                ),
            )
            self._conn.commit()
        return record

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        record = self._row_to_session(row)
        if record.expires_at is not None and record.expires_at <= time.time():
            return None
        return record

    def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET state = ? WHERE session_id = ?",
                (_dumps(state), session_id),
            )
            self._conn.commit()

    def session_ids(self, user_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT session_id FROM sessions WHERE user_id = ?", (user_id,)).fetchall()
        return [r["session_id"] for r in rows]

    def close_session(self, session_id: str, closed_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET closed_at = ? WHERE session_id = ?",
                (closed_at, session_id),
            )
            self._conn.commit()

    # -- lifecycle ----------------------------------------------------------

    def forget_user(self, user_id: str) -> int:
        with self._lock, self._write():
            # Events recorded inside one of the user's sessions are the
            # user's too, even when the event itself carried no user_id.
            n = self._conn.execute(
                "DELETE FROM claims WHERE session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)",
                (user_id,),
            ).rowcount
            n += self._conn.execute(
                "DELETE FROM events WHERE user_id = ? OR session_id IN "
                "(SELECT session_id FROM sessions WHERE user_id = ?)",
                (user_id, user_id),
            ).rowcount
            n += self._conn.execute(
                "DELETE FROM sessions WHERE user_id = ?", (user_id,)
            ).rowcount
        return n

    def prune(self, *, older_than: float) -> int:
        with self._lock, self._write():
            self._rollup(older_than)  # keep daily counts of what is about to go
            n = self._conn.execute("DELETE FROM events WHERE ts < ?", (older_than,)).rowcount
            n += self._conn.execute(
                "DELETE FROM sessions WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (time.time(),),
            ).rowcount
        return n

    def _rollup(self, older_than: float) -> None:
        """Add daily counts of the events older than `older_than` to the
        rollups. Called with the lock held, in prune's transaction."""
        rows = self._conn.execute(
            f"SELECT {self._bucket.format('ts')} AS day, kind, {self._source} AS source, {self._scope} AS scope, "
            "COUNT(*) AS n FROM events WHERE ts < ? GROUP BY 1, 2, 3, 4",
            (_DAY, _DAY, older_than),
        ).fetchall()
        self._conn.executemany(
            "INSERT INTO event_rollups (day, kind, source, scope, n) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(day, kind, source, scope) DO UPDATE SET n = event_rollups.n + excluded.n",
            [(r["day"], r["kind"], r["source"], r["scope"], r["n"]) for r in rows],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SQLiteStore(SqlStore):
    """`SQLiteStore("mem.db")`, or `SQLiteStore(":memory:")` for tests."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        super().__init__(path)


class PostgresStore(SqlStore):
    """`PostgresStore("postgresql://user@host:5432/db")`; add `?schema=name`
    to keep this memory in its own schema. Needs `commonmemo[postgres]`."""

    def __init__(self, url: str) -> None:
        super().__init__(url)
