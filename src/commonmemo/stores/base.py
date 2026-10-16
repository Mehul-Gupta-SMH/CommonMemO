"""The `Store` contract: where the tiers CommonMemo owns itself live.

CommonMemo owns the **event** tier (append-only log) and the **session** tier
(live state keyed by `session_id`). Both are simple, hot and must never be
lost, so they sit in a store we control rather than in a wrapped engine.
Episodic and user memory belong to an `Engine` (see `engines/base.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..events import MemoryEvent


@dataclass
class SessionRecord:
    """One session's row: who it belongs to, its lifetime, and its state."""

    session_id: str
    user_id: Optional[str]
    agent: Optional[str]
    opened_at: float
    closed_at: Optional[float] = None
    expires_at: Optional[float] = None
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


class Store(ABC):
    """Persistence for events and sessions. Implementations must be safe to
    call from multiple threads of one process."""

    # -- event tier ---------------------------------------------------------

    @abstractmethod
    def append(self, event: MemoryEvent) -> bool:
        """Persist one event; return whether it was new. Appending an id that
        already exists is a no-op returning False (ids are idempotency
        keys), never an overwrite."""

    @abstractmethod
    def events(
        self,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        kinds: Optional[Iterable[str]] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryEvent]:
        """Events matching every given filter, oldest first. `limit` keeps
        the most recent `limit` events (still returned oldest first)."""

    # -- session tier -------------------------------------------------------

    @abstractmethod
    def open_session(self, record: SessionRecord) -> SessionRecord:
        """Create the session, or resume it if `record.session_id` already
        exists: the existing record comes back (its owner, agent and state
        untouched) with `closed_at` cleared, so a closed session can be
        continued -- from another process or another harness."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """The session's record, or `None` if unknown or expired."""

    @abstractmethod
    def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Replace the session's state."""

    @abstractmethod
    def session_ids(self, user_id: str) -> list[str]:
        """Every session id that belongs to `user_id`."""

    @abstractmethod
    def close_session(self, session_id: str, closed_at: float) -> None:
        """Mark the session closed. Its state and events are kept."""

    # -- lifecycle ----------------------------------------------------------

    @abstractmethod
    def forget_user(self, user_id: str) -> int:
        """Delete every event and session belonging to `user_id`; return how
        many rows were removed."""

    @abstractmethod
    def prune(self, *, older_than: float) -> int:
        """Delete events older than `older_than` (epoch seconds) and expired
        sessions; return how many rows were removed."""

    def close(self) -> None:
        """Release resources. Default: nothing to release."""
