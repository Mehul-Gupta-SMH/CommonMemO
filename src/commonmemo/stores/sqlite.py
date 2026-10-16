"""SQLite `Store` -- the zero-setup default (stdlib `sqlite3`, no install)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from ..events import MemoryEvent, event_from_dict
from .base import ClaimRecord, SessionRecord, Store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    id         TEXT NOT NULL UNIQUE,
    ts         REAL NOT NULL,
    kind       TEXT NOT NULL,
    session_id TEXT,
    user_id    TEXT,
    agent      TEXT,
    body       TEXT NOT NULL
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
"""


def _dumps(value: Any) -> str:
    # default=str: an argument or result that is not JSON-native (a datetime,
    # a custom object) is stored as its str() rather than failing the write.
    return json.dumps(value, default=str, sort_keys=True)


class SQLiteStore(Store):
    """`SQLiteStore("mem.db")`, or `SQLiteStore(":memory:")` for tests."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- event tier ---------------------------------------------------------

    def _insert(self, event: MemoryEvent) -> bool:
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO events (id, ts, kind, session_id, user_id, agent, body) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.ts,
                event.kind,
                event.session_id,
                event.user_id,
                event.agent,
                _dumps(event.to_dict()),
            ),
        )
        return cur.rowcount == 1

    def append(self, event: MemoryEvent) -> bool:
        with self._lock:
            new = self._insert(event)
            self._conn.commit()
        return new

    def events_after(self, seq: int, *, limit: Optional[int] = None) -> list[tuple[int, MemoryEvent]]:
        # SQLite has one writer at a time, so positions commit in order and
        # "after seq" never skips an event that commits later.
        sql, params = "SELECT seq, body FROM events WHERE seq > ? ORDER BY seq", [seq]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(r["seq"], event_from_dict(json.loads(r["body"]))) for r in rows]

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
    ) -> list[MemoryEvent]:
        where, params = [], []
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

    # -- session tier -------------------------------------------------------

    def _row_to_session(self, row: sqlite3.Row) -> SessionRecord:
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
                "INSERT OR REPLACE INTO sessions "
                "(session_id, user_id, agent, opened_at, closed_at, expires_at, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
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
        with self._lock:
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
            self._conn.commit()
        return n

    def prune(self, *, older_than: float) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM events WHERE ts < ?", (older_than,)).rowcount
            n += self._conn.execute(
                "DELETE FROM sessions WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (time.time(),),
            ).rowcount
            self._conn.commit()
        return n

    def close(self) -> None:
        with self._lock:
            self._conn.close()
