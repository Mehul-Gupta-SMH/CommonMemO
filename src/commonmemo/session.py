"""`Session` -- the session tier, and the scope everything else hangs off.

A session is one conversation or one workflow run. While it is open:

- every event recorded through it (or through `@mem.track` while it is the
  current session) is stamped with its `session_id`, `user_id` and `agent`
  and appended to the event log;
- `session.state` is working memory, persisted on every write, so a
  session resumed by id (even from another process or harness) continues
  where it stopped.

Closing it hands the session's events to the engine for consolidation into
episodic and user memory.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .context import MemoryContext
from .engines.base import Episode, Fact
from .events import MemoryEvent, Message, UserFact
from .stores.base import SessionRecord

if TYPE_CHECKING:
    from .memory import Memory

_current: contextvars.ContextVar[Optional["Session"]] = contextvars.ContextVar(
    "commonmemo_current_session", default=None
)


def current_session() -> Optional["Session"]:
    """The session entered most recently in this task/thread, if any."""
    return _current.get()


class SessionState(MutableMapping[str, Any]):
    """A dict that writes through to the store on every change."""

    def __init__(self, session: "Session", initial: dict[str, Any]) -> None:
        self._session = session
        self._data = dict(initial)

    def _save(self) -> None:
        self._session.memory.store.save_state(self._session.session_id, self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._save()

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"SessionState({self._data!r})"

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class Session:
    """Create through `Memory.session(...)`, then use as a context manager
    (`async with` or `with`), or call `aclose()` yourself."""

    def __init__(self, memory: "Memory", record: SessionRecord) -> None:
        self.memory = memory
        self._record = record
        self.state = SessionState(self, record.state)
        self.episodes: list[Episode] = []
        """What consolidation produced at close (empty until then)."""
        self._token: Optional[contextvars.Token] = None
        self._closed = not record.is_open

    @property
    def session_id(self) -> str:
        return self._record.session_id

    @property
    def user_id(self) -> Optional[str]:
        return self._record.user_id

    @property
    def agent(self) -> Optional[str]:
        return self._record.agent

    @property
    def closed(self) -> bool:
        return self._closed

    # -- writing ------------------------------------------------------------

    def record(self, event: MemoryEvent) -> MemoryEvent:
        """Append `event` to the log, stamped with this session's envelope."""
        if self._closed:
            raise RuntimeError(
                f"commonmemo: session {self.session_id!r} is closed; open a new one "
                "or resume it with Memory.session(session_id=...) before recording"
            )
        bound = event.bind(session_id=self.session_id, user_id=self.user_id, agent=self.agent)
        self.memory.store.append(bound)
        return bound

    def message(self, role: str, content: str) -> MemoryEvent:
        """Shorthand for recording a `Message`."""
        return self.record(Message(role=role, content=content))

    def remember(self, key: str, value: str) -> MemoryEvent:
        """Record an explicit user fact (see `UserFact`)."""
        if self.user_id is None:
            raise ValueError(
                "commonmemo: remember() needs a user -- open the session with "
                "Memory.session(user_id=...)"
            )
        return self.record(UserFact(key=key, value=value))

    def events(self, **filters: Any) -> list[MemoryEvent]:
        """This session's events, oldest first (filters as `Store.events`)."""
        return self.memory.store.events(session_id=self.session_id, **filters)

    # -- reading ------------------------------------------------------------

    async def context(
        self, query: Optional[str] = None, *, episodes: int = 3, recent: int = 10
    ) -> MemoryContext:
        """Everything the four tiers know that is relevant to `query`: user
        facts, the best-matching past episodes (excluding this session),
        this session's state and its `recent` latest events."""
        ctx = await self.memory.context(
            query, user_id=self.user_id, agent=self.agent, episodes=episodes
        )
        ctx.episodes = [e for e in ctx.episodes if e.session_id != self.session_id]
        # Facts stated in this session reach the engine only at close, but
        # they are true now: overlay them (latest per key wins).
        if self.user_id is not None:
            facts = {f.key: f for f in ctx.facts}
            for e in self.events(kinds=[UserFact.kind]):
                facts[e.key] = Fact(
                    user_id=self.user_id, key=e.key, value=e.value,
                    valid_from=e.ts, source_event_ids=(e.id,),
                )
            ctx.facts = sorted(facts.values(), key=lambda f: f.key)
        ctx.state = self.state.to_dict()
        ctx.recent = self.events(limit=recent) if recent else []
        return ctx

    # -- closing ------------------------------------------------------------

    async def aclose(self, *, consolidate: bool = True) -> list[Episode]:
        """Close the session and, unless `consolidate=False`, hand its
        events to the engine. Closing twice is a no-op."""
        if self._closed:
            return self.episodes
        self._closed = True
        closed_at = time.time()
        self.memory.store.close_session(self.session_id, closed_at)
        self._record.closed_at = closed_at
        if consolidate:
            self.episodes = await self.memory.engine.ingest(self._record, self.events())
        return self.episodes

    async def __aenter__(self) -> "Session":
        self._token = _current.set(self)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        try:
            await self.aclose()
        finally:
            self._reset_current()

    def __enter__(self) -> "Session":
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.aclose())
            else:
                raise RuntimeError(
                    "commonmemo: a plain `with` session cannot consolidate inside a "
                    "running event loop -- use `async with mem.session(...)` there"
                )
        finally:
            self._reset_current()

    def _reset_current(self) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"<Session {self.session_id} user={self.user_id} agent={self.agent} {status}>"
