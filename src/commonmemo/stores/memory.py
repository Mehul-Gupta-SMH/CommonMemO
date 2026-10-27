"""`MemoryStore`: the `Store` contract in plain Python, with no SQL and no persistence (#95).

    Memory("memory://")            # or MemoryStore()

It is the reference for anyone writing a store that is not SQL (DynamoDB, a document or key-value
database): every guarantee in `Store` is here as a few lines of ordinary code under one lock, and
`tests/test_store_contract.py` runs the same suite against it and the SQL stores. It is also handy
for tests and for agents whose memory should end with the process.

Nothing survives the process, and every call returns copies, never what the store holds, as a
database would.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Iterable, Optional

from ..events import MemoryEvent, event_from_dict
from .base import ClaimRecord, EventCount, SessionRecord, Store

_DAY = 86400.0


def _bucket(ts: float, size: float) -> float:
    return float(int(ts / size) * size)  # ts is positive: truncation is a floor, as in the SQL stores


class MemoryStore(Store):
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        self._log: list[tuple[int, MemoryEvent]] = []  # (position, event), positions only grow
        self._ids: set[str] = set()
        self._next = 1
        self._sessions: dict[str, SessionRecord] = {}
        self._claims: dict[str, dict[str, Any]] = {}
        self._rollups: dict[tuple[float, str, str, str], int] = {}

    # -- the Log ------------------------------------------------------------

    def _insert(self, event: MemoryEvent) -> bool:
        if event.id in self._ids:
            return False
        self._ids.add(event.id)
        self._log.append((self._next, event_from_dict(event.to_dict())))  # a copy, as a database would hold
        self._next += 1
        return True

    def append(self, event: MemoryEvent) -> bool:
        with self._lock:
            return self._insert(event)

    def append_many(self, events: Iterable[MemoryEvent]) -> list[bool]:
        with self._lock:
            return [self._insert(e) for e in events]

    def events(self, *, session_id: Optional[str] = None, user_id: Optional[str] = None,
               kinds: Optional[Iterable[str]] = None, since: Optional[float] = None, limit: Optional[int] = None,
               run_id: Optional[str] = None) -> list[MemoryEvent]:
        wanted = None if kinds is None else set(kinds)
        with self._lock:
            found = [e for _, e in self._log
                     if (run_id is None or e.run_id == run_id) and (session_id is None or e.session_id == session_id)
                     and (user_id is None or e.user_id == user_id) and (wanted is None or e.kind in wanted)
                     and (since is None or e.ts >= since)]
        found = found if limit is None else found[len(found) - limit:] if limit > 0 else []
        return [event_from_dict(e.to_dict()) for e in found]

    def get_events(self, ids: Iterable[str]) -> list[MemoryEvent]:
        wanted = set(ids)
        with self._lock:
            return [event_from_dict(e.to_dict()) for _, e in self._log if e.id in wanted]

    def events_after(self, seq: int, *, limit: Optional[int] = None) -> list[tuple[int, MemoryEvent]]:
        with self._lock:  # appends happen under the lock: a position is visible only with every earlier one
            found = [(n, e) for n, e in self._log if n > seq]
        found = found if limit is None else found[:limit]
        return [(n, event_from_dict(e.to_dict())) for n, e in found]

    def position(self, *, event_id: Optional[str] = None, at: Optional[float] = None) -> int:
        with self._lock:
            if event_id is not None:
                return next((n for n, e in self._log if e.id == event_id), 0)
            if at is not None:
                return max((n for n, e in self._log if e.ts <= at), default=0)
            return self._log[-1][0] if self._log else 0

    @staticmethod
    def _label(event: MemoryEvent, field: str) -> str:
        value = event.to_dict().get(field)
        return "" if value is None else str(value)

    def event_counts(self, *, since: Optional[float] = None, until: Optional[float] = None,
                     bucket: float = 86400.0) -> list[EventCount]:
        lo = float("-inf") if since is None else since
        hi = float("inf") if until is None else until
        totals: dict[tuple[float, str, str, str], int] = {}
        with self._lock:
            for _, e in self._log:
                if lo <= e.ts < hi:
                    k = (_bucket(e.ts, bucket), e.kind, self._label(e, "source"), self._label(e, "scope"))
                    totals[k] = totals.get(k, 0) + 1
            for (day, kind, source, scope), n in self._rollups.items():
                if lo <= day < hi:
                    k = (_bucket(day, bucket), kind, source, scope)
                    totals[k] = totals.get(k, 0) + n
        return [EventCount(*k, n) for k, n in sorted(totals.items())]

    # -- claims -------------------------------------------------------------

    def claim(self, key: str, *, session_id: Optional[str], owner: str, lease_until: float,
              event: MemoryEvent) -> Optional[ClaimRecord]:
        with self._lock:
            row = self._claims.get(key)
            if row is not None and row["status"] != "free":
                return ClaimRecord(key, row["owner"], row["status"], row["lease_until"])
            self._claims[key] = {"session_id": session_id, "owner": owner, "status": "held", "lease_until": lease_until}
            self._insert(event)
        return None

    def get_claim(self, key: str) -> Optional[ClaimRecord]:
        with self._lock:
            row = self._claims.get(key)
            return None if row is None else ClaimRecord(key, row["owner"], row["status"], row["lease_until"])

    def settle(self, key: str, *, session_id: Optional[str], status: str, event: MemoryEvent) -> None:
        with self._lock:
            row = self._claims.get(key)
            if row is None:
                self._claims[key] = {"session_id": session_id, "owner": "", "status": status, "lease_until": None}
            else:
                row.update(status=status, lease_until=None)
            self._insert(event)

    # -- Working memory -----------------------------------------------------

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or (record.expires_at is not None and record.expires_at <= time.time()):
                return None
            return copy.deepcopy(record)

    def open_session(self, record: SessionRecord) -> SessionRecord:
        with self._lock:
            existing = self.get_session(record.session_id)
            if existing is not None:
                self._sessions[record.session_id].closed_at = None
                existing.closed_at = None
                return existing
            self._sessions[record.session_id] = copy.deepcopy(record)
            return record

    def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].state = copy.deepcopy(state)

    def session_ids(self, user_id: str) -> list[str]:
        with self._lock:
            return [s.session_id for s in self._sessions.values() if s.user_id == user_id]

    def close_session(self, session_id: str, closed_at: float) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].closed_at = closed_at

    # -- lifecycle ----------------------------------------------------------

    def forget_user(self, user_id: str) -> int:
        with self._lock:
            mine = {s.session_id for s in self._sessions.values() if s.user_id == user_id}
            claims = [k for k, c in self._claims.items() if c["session_id"] in mine]
            for k in claims:
                del self._claims[k]
            before = len(self._log)
            self._log = [(n, e) for n, e in self._log if not (e.user_id == user_id or e.session_id in mine)]
            gone = before - len(self._log)
            self._ids = {e.id for _, e in self._log}
            for s in mine:
                del self._sessions[s]
            return len(claims) + gone + len(mine)

    def prune(self, *, older_than: float) -> int:
        with self._lock:
            old = [e for _, e in self._log if e.ts < older_than]
            for e in old:  # daily counts of what is about to go, as the SQL stores keep
                k = (_bucket(e.ts, _DAY), e.kind, self._label(e, "source"), self._label(e, "scope"))
                self._rollups[k] = self._rollups.get(k, 0) + 1
            self._log = [(n, e) for n, e in self._log if e.ts >= older_than]
            self._ids = {e.id for _, e in self._log}
            now = time.time()
            expired = [s for s, r in self._sessions.items() if r.expires_at is not None and r.expires_at <= now]
            for s in expired:
                del self._sessions[s]
            return len(old) + len(expired)
