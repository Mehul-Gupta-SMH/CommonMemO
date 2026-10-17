"""`LocalEngine` -- the offline, deterministic engine.

No LLM, no network, no install: recaps are rule-based digests of a
session's events, retrieval is keyword overlap, and user facts come only
from explicit `UserFact` events. It exists so CommonMemo works end to end
out of the box and so tests are reproducible -- it is not trying to compete
with the engines CommonMemo wraps (see `engines/base.py`).

Facts are kept with their history: a new value for a key closes the old
row (`valid_to`) instead of deleting it, the same "invalidate, don't
discard" rule Graphiti follows.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Sequence

from ..events import BOOKKEEPING, ExternalEvent, MemoryEvent, Message, ToolInvoked, UserFact, describe
from ..stores.base import SessionRecord
from .base import Engine, Recap, Fact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    user_id    TEXT,
    agent      TEXT,
    summary    TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at   REAL NOT NULL,
    sources    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS episodes_user ON episodes(user_id);
CREATE TABLE IF NOT EXISTS facts (
    user_id    TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    valid_from REAL NOT NULL,
    valid_to   REAL,
    source     TEXT NOT NULL,
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS facts_user ON facts(user_id, key);
CREATE TABLE IF NOT EXISTS external_ids (
    session_id  TEXT NOT NULL,
    system      TEXT NOT NULL,
    external_id TEXT NOT NULL,
    PRIMARY KEY (session_id, system)
);
CREATE INDEX IF NOT EXISTS external_ids_lookup ON external_ids(system, external_id);
"""

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has i in is it of on or that the to was with you".split())


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP}


def _clip(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def summarize(session: SessionRecord, events: Sequence[MemoryEvent]) -> str:
    """A rule-based, one-paragraph digest of a session."""
    messages = [e for e in events if isinstance(e, Message)]
    tools = [e for e in events if isinstance(e, ToolInvoked)]
    external = [e for e in events if isinstance(e, ExternalEvent)]
    facts = [e for e in events if isinstance(e, UserFact)]
    described = [d for d in (describe(e) for e in events) if d]
    other = [
        e for e in events
        if not isinstance(e, (Message, ToolInvoked, UserFact, *BOOKKEEPING)) and describe(e) is None
    ]

    parts = [f"Session {session.session_id}" + (f" with agent {session.agent}" if session.agent else "")]
    first_user = next((m for m in messages if m.role == "user"), None)
    last_reply = next((m for m in reversed(messages) if m.role == "assistant"), None)
    if first_user:
        parts.append(f"user asked: {_clip(first_user.content)}")
    if tools:
        names = sorted({t.name for t in tools})
        failed = sum(1 for t in tools if t.error)
        parts.append(
            f"tools used: {', '.join(names)} ({len(tools)} calls"
            + (f", {failed} failed" if failed else "")
            + ")"
        )
    if external:
        sources = sorted({e.source for e in external})
        parts.append(f"external events: {', '.join(sources)}")
    if other:
        kinds = sorted({e.kind for e in other})
        parts.append(f"events: {', '.join(kinds)}")
    if facts:
        parts.append("learned: " + "; ".join(f"{f.key}={f.value}" for f in facts))
    if described:
        parts.append("noted: " + " ".join(described))
    if last_reply:
        parts.append(f"final answer: {_clip(last_reply.content)}")
    return ". ".join(parts) + "."


class LocalEngine(Engine):
    """`LocalEngine("mem.db")`, or `LocalEngine()` for an in-memory one."""

    name = "local"

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

    async def ingest(
        self, session: SessionRecord, events: Sequence[MemoryEvent]
    ) -> list[Recap]:
        if not events:
            return []
        recap = Recap(
            id=f"ep_{session.session_id}",
            session_id=session.session_id,
            user_id=session.user_id,
            agent=session.agent,
            summary=summarize(session, events),
            started_at=min(e.ts for e in events),
            ended_at=max(e.ts for e in events),
            source_event_ids=tuple(e.id for e in events),
        )
        with self._lock:
            # Idempotent per session: a re-ingest replaces, never duplicates.
            self._conn.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    recap.id,
                    recap.session_id,
                    recap.user_id,
                    recap.agent,
                    recap.summary,
                    recap.started_at,
                    recap.ended_at,
                    json.dumps(list(recap.source_event_ids)),
                ),
            )
            self._conn.execute("DELETE FROM facts WHERE session_id = ?", (session.session_id,))
            for fact in sorted((e for e in events if isinstance(e, UserFact)), key=lambda e: e.ts):
                user_id = fact.user_id or session.user_id
                if user_id is None:
                    continue
                self._conn.execute(
                    "UPDATE facts SET valid_to = ? WHERE user_id = ? AND key = ? AND valid_to IS NULL",
                    (fact.ts, user_id, fact.key),
                )
                self._conn.execute(
                    "INSERT INTO facts VALUES (?, ?, ?, ?, NULL, ?, ?)",
                    (user_id, fact.key, fact.value, fact.ts, json.dumps([fact.id]), session.session_id),
                )
            self._conn.commit()
        return [recap]

    def _row_to_recap(self, row: sqlite3.Row, score: Optional[float] = None) -> Recap:
        return Recap(
            id=row["id"],
            session_id=row["session_id"],
            user_id=row["user_id"],
            agent=row["agent"],
            summary=row["summary"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            source_event_ids=tuple(json.loads(row["sources"])),
            score=score,
        )

    async def search_recaps(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 5,
    ) -> list[Recap]:
        where, params = [], []
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if agent is not None:
            where.append("agent = ?")
            params.append(agent)
        sql = "SELECT * FROM episodes" + (" WHERE " + " AND ".join(where) if where else "")
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        wanted = _tokens(query)
        scored = []
        for row in rows:
            overlap = len(wanted & _tokens(row["summary"]))
            if wanted and not overlap:
                continue
            score = overlap / len(wanted) if wanted else 0.0
            scored.append((score, row["ended_at"], row))
        # Best match first; recency breaks ties (and orders an empty query).
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [self._row_to_recap(r, s) for s, _, r in scored[:limit]]

    async def user_profile(self, user_id: str) -> list[Fact]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE user_id = ? AND valid_to IS NULL ORDER BY key",
                (user_id,),
            ).fetchall()
        return [
            Fact(
                user_id=r["user_id"],
                key=r["key"],
                value=r["value"],
                valid_from=r["valid_from"],
                source_event_ids=tuple(json.loads(r["source"])),
            )
            for r in rows
        ]

    async def fact_history(self, user_id: str, key: str) -> list[tuple[str, float, Optional[float]]]:
        """Every value `key` has had for `user_id`: (value, valid_from, valid_to), oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT value, valid_from, valid_to FROM facts WHERE user_id = ? AND key = ? "
                "ORDER BY valid_from",
                (user_id, key),
            ).fetchall()
        return [(r["value"], r["valid_from"], r["valid_to"]) for r in rows]

    # -- ids another system assigned to a session ------------------------
    # A wrapped engine whose backend picks its own ids (a Graphiti episode
    # uuid) records them here, next to the recap they belong to.

    def set_external_id(self, session_id: str, system: str, external_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO external_ids VALUES (?, ?, ?)", (session_id, system, external_id)
            )
            self._conn.commit()

    def delete_external_id(self, session_id: str, system: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM external_ids WHERE session_id = ? AND system = ?", (session_id, system)
            )
            self._conn.commit()

    def external_id(self, session_id: str, system: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT external_id FROM external_ids WHERE session_id = ? AND system = ?", (session_id, system)
            ).fetchone()
        return row["external_id"] if row else None

    def external_ids(self, system: str, session_ids: Sequence[str]) -> dict[str, str]:
        """`{external_id: session_id}` for the given sessions."""
        if not session_ids:
            return {}
        marks = ",".join("?" * len(session_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT session_id, external_id FROM external_ids WHERE system = ? AND session_id IN ({marks})",
                (system, *session_ids),
            ).fetchall()
        return {r["external_id"]: r["session_id"] for r in rows}

    async def forget_user(self, user_id: str) -> int:
        with self._lock:
            n = self._conn.execute(
                "DELETE FROM external_ids WHERE session_id IN (SELECT session_id FROM episodes WHERE user_id = ?)",
                (user_id,),
            ).rowcount
            n += self._conn.execute("DELETE FROM episodes WHERE user_id = ?", (user_id,)).rowcount
            n += self._conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,)).rowcount
            self._conn.commit()
        return n

    async def aclose(self) -> None:
        with self._lock:
            self._conn.close()

