"""`Memory` -- the one object an application holds.

    mem = Memory("sqlite:///mem.db")          # store + default local engine

    @mem.track                                # event tier, automatically
    def search_web(query: str) -> str: ...

    async with mem.session(user_id="u1", agent="coordinator") as s:
        ctx = await s.context(prompt)         # all four tiers, for this turn
        ...
    # on exit: consolidation -> episodic + user memory
"""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from typing import Any, Callable, Optional, TypeVar, Union, overload

from .context import MemoryContext
from .engines import Engine, LocalEngine, engine_from_name
from .events import MemoryEvent, ToolInvoked
from .session import Session, current_session
from .stores import SQLiteStore, Store, store_from_url
from .stores.base import SessionRecord

F = TypeVar("F", bound=Callable[..., Any])


def _clip(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


class Memory:
    """Event, session, episodic and user memory behind one handle.

    `store` is a `Store` or a URL (`sqlite:///mem.db`, `sqlite://:memory:`).
    `engine` is an `Engine` or a name; the default `"local"` engine shares
    the store's SQLite file, so one path is the whole setup.
    """

    def __init__(
        self,
        store: Union[str, Store] = "sqlite://:memory:",
        engine: Union[str, Engine] = "local",
        *,
        session_ttl: Optional[float] = None,
    ) -> None:
        self.store = store_from_url(store) if isinstance(store, str) else store
        if isinstance(engine, str):
            options: dict[str, Any] = {}
            if engine == "local" and isinstance(self.store, SQLiteStore):
                options["path"] = self.store.path
            engine = engine_from_name(engine, **options)
        engine.bind(self.store)
        self.engine = engine
        self.session_ttl = session_ttl
        """Default lifetime of a session in seconds (`None`: no expiry)."""

    # -- sessions -----------------------------------------------------------

    def session(
        self,
        session_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> Session:
        """Open a new session, or resume an existing one by `session_id`.

        Resuming checks ownership: a session that belongs to one user can
        never be resumed as another (that would leak one user's working
        memory into another's turn).
        """
        if session_id is not None and user_id is not None:
            existing = self.store.get_session(session_id)
            if existing is not None and existing.user_id not in (None, user_id):
                raise PermissionError(
                    f"commonmemo: session {session_id!r} belongs to another user; "
                    f"it cannot be resumed as user {user_id!r}"
                )
        now = time.time()
        ttl = self.session_ttl if ttl is None else ttl
        record = self.store.open_session(
            SessionRecord(
                session_id=session_id or uuid.uuid4().hex,
                user_id=user_id,
                agent=agent,
                opened_at=now,
                expires_at=now + ttl if ttl is not None else None,
            )
        )
        return Session(self, record)

    # -- events -------------------------------------------------------------

    def record(self, event: MemoryEvent) -> MemoryEvent:
        """Record `event` in the current session if one is open, else
        straight to the log with whatever envelope the event carries."""
        session = current_session()
        if session is not None and session.memory is self and not session.closed:
            return session.record(event)
        self.store.append(event)
        return event

    @overload
    def track(self, fn: F) -> F: ...

    @overload
    def track(
        self, *, name: Optional[str] = None, capture_result: bool = True, max_chars: int = 2000
    ) -> Callable[[F], F]: ...

    def track(
        self,
        fn: Optional[F] = None,
        *,
        name: Optional[str] = None,
        capture_result: bool = True,
        max_chars: int = 2000,
    ) -> Any:
        """Decorator: every call of the function becomes a `ToolInvoked`
        event (arguments, result or error, duration). Works on sync and
        async functions, bare (`@mem.track`) or configured
        (`@mem.track(name="search", capture_result=False)`).

        The call itself is never changed: the return value and any
        exception pass through untouched, and a failure to record is never
        allowed to fail the call (memory is an observer, not a gate).
        """

        def decorate(func: F) -> F:
            tool_name = name or func.__name__
            sig = inspect.signature(func)

            def arguments(args: tuple, kwargs: dict) -> dict[str, Any]:
                try:
                    bound = sig.bind_partial(*args, **kwargs)
                except TypeError:
                    return {"args": [_clip(a, max_chars) for a in args], **kwargs}
                return {k: v if isinstance(v, (int, float, bool, type(None))) else _clip(v, max_chars)
                        for k, v in bound.arguments.items()}

            def emit(args: tuple, kwargs: dict, started: float, result: Any, error: Optional[BaseException]) -> None:
                try:
                    self.record(
                        ToolInvoked(
                            name=tool_name,
                            arguments=arguments(args, kwargs),
                            result=_clip(result, max_chars) if capture_result and error is None else None,
                            error=f"{type(error).__name__}: {error}" if error is not None else None,
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                    )
                except Exception:  # noqa: BLE001 -- observer must never break the call
                    pass

            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    started = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        emit(args, kwargs, started, None, exc)
                        raise
                    emit(args, kwargs, started, result, None)
                    return result

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    emit(args, kwargs, started, None, exc)
                    raise
                emit(args, kwargs, started, result, None)
                return result

            return wrapper  # type: ignore[return-value]

        return decorate(fn) if fn is not None else decorate

    # -- reading ------------------------------------------------------------

    async def context(
        self,
        query: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        episodes: int = 3,
    ) -> MemoryContext:
        """Long-term context only (user facts + matching episodes). Inside a
        session, prefer `Session.context`, which adds the session tier."""
        found = (
            await self.engine.search_episodes(query or "", user_id=user_id, agent=agent, limit=episodes)
            if episodes
            else []
        )
        facts = await self.engine.user_facts(user_id) if user_id is not None else []
        return MemoryContext(episodes=found, facts=facts)

    # -- lifecycle ----------------------------------------------------------

    async def forget_user(self, user_id: str) -> int:
        """Delete everything about `user_id` from every tier. Returns the
        number of rows removed."""
        return self.store.forget_user(user_id) + await self.engine.forget_user(user_id)

    def prune(self, *, older_than_days: float) -> int:
        """Drop events older than `older_than_days` and expired sessions.
        Episodes and facts derived from them are kept (they carry their own
        provenance ids)."""
        return self.store.prune(older_than=time.time() - older_than_days * 86400)

    async def aclose(self) -> None:
        await self.engine.aclose()
        self.store.close()


__all__ = ["Memory", "LocalEngine"]
