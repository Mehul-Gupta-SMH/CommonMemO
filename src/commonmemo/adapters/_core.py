"""What every adapter needs: sessions to write under, a session to search from, and ids that make a
second write of the same thing a no-op."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..memory import Memory
    from ..session import Session


def stable_id(*parts: str) -> str:
    """An event id derived from where the record came from, so recording it twice is one event."""
    return "ad_" + hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:24]


def writer(memory: "Memory", session_id: Optional[str], user_id: Optional[str], agent: Optional[str]) -> "Session":
    """The session an SDK's records are written under: the SDK's own session, resumed."""
    return memory.session(session_id, user_id=user_id, agent=agent)


def reader(memory: "Memory", user_id: Optional[str], agent: Optional[str]) -> "Session":
    """One long-lived session per (agent, user) that searches run from. It is never closed, so
    searching does not consolidate anything, and it expires like any other session."""
    return memory.session(f"adapter-search:{agent or ''}:{user_id or ''}", user_id=user_id, agent=agent)
