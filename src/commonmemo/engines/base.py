"""The `Engine` contract: who owns Recaps and the UserProfile.

CommonMemo wraps an engine rather than building one. Extraction, retrieval,
contradiction handling and their optimizations are what dedicated engines
(Graphiti, Mem0, ...) are good at, so the derived tiers -- Recaps and the
UserProfile -- are delegated to them behind this one interface. CommonMemo's
job is the tier model, the Log they are derived from, and the integrations.

Two rules every engine must keep:

1. **Provenance.** Every `Recap` and `Fact` carries the ids of the events
   it came from (`source_event_ids`), so a memory can always be traced back
   to what actually happened.
2. **Explicit facts win.** A `UserFact` event is a direct statement; an
   engine must store it as-is (latest value per key supersedes the previous
   one). LLM-extracted facts are added alongside, never in place of it.

Engine methods are async: every real engine underneath (Graphiti, Mem0's
AsyncMemory) is async-first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from ..events import MemoryEvent
from ..stores.base import SessionRecord, Store

if TYPE_CHECKING:
    from ..ranking import Ranker


@dataclass(frozen=True)
class Recap:
    """A consolidated summary of one session."""

    id: str
    session_id: str
    user_id: Optional[str]
    agent: Optional[str]
    summary: str
    started_at: float
    ended_at: float
    source_event_ids: tuple[str, ...] = ()
    score: Optional[float] = None
    """Relevance to the query that retrieved it; `None` outside a search."""


@dataclass(frozen=True)
class Fact:
    """A durable fact about a user, current as of `valid_from`."""

    user_id: str
    key: str
    value: str
    valid_from: float
    source_event_ids: tuple[str, ...] = field(default_factory=tuple)


class Engine(ABC):
    """Owns the Recaps and UserProfile tiers."""

    name: str = "engine"

    @abstractmethod
    async def ingest(
        self, session: SessionRecord, events: Sequence[MemoryEvent]
    ) -> list[Recap]:
        """Consolidate one closed session's events into recaps and facts.

        Called once per session close. Must be idempotent per
        `session.session_id`: ingesting the same session again replaces
        what the previous ingest produced instead of duplicating it.
        """

    @abstractmethod
    async def search_recaps(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 5,
    ) -> list[Recap]:
        """Recaps most relevant to `query`, best first."""

    @abstractmethod
    async def user_profile(self, user_id: str) -> list[Fact]:
        """The UserProfile: the user's current facts (superseded values excluded)."""

    @abstractmethod
    async def forget_user(self, user_id: str) -> int:
        """Delete everything this engine holds about `user_id`."""

    def ranker(self) -> Optional["Ranker"]:
        """A ranker by meaning, for engines that embed text (see
        `ranking.py`); None leaves ranking to `Memory`'s default."""
        return None

    def bind(self, store: "Store") -> None:
        """Called once by `Memory` with its store, before any other method.
        An engine that keeps local state can colocate it with the store
        (same SQLite file). Default: nothing to do."""

    async def aclose(self) -> None:
        """Release resources. Default: nothing to release."""
