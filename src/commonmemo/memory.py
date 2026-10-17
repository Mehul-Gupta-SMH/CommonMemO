"""`Memory` -- the one object an application holds.

    mem = Memory("sqlite:///mem.db")          # store + default local engine

    @mem.track                                # into the Log, automatically
    def search_web(query: str) -> str: ...

    @mem.step(side_effects=True)              # resumable: replayed if done
    def charge_card(order_id: str) -> str: ...

    async with mem.session(user_id="u1", agent="coordinator") as s:
        ctx = await s.context(prompt)         # what memory pushes this turn
        tools = mem.agent_tools(s)            # what agents can pull / write
        ...
    # on exit: consolidation -> a recap + UserProfile facts
"""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from typing import Any, Callable, Optional, TypeVar, Union, overload

from .agent_tools import build_agent_tools
from .context import MemoryContext, MemoryPolicy
from .engines import Engine, Fact, LocalEngine, engine_from_name
from .events import (
    MemoryEvent,
    Negative,
    NegativeRevoked,
    ToolInvoked,
    WorldFact,
    WorldFactConfirmed,
    WorldFactRejected,
    WorldFactRetracted,
    Snapshot,
)
from .knowledge import FeedbackSummary, Knowledge, NegativeRecord, WorldRecord, source_rank
from .history import Diff, History, MemoryView, Point, as_of, diff
from .resume import make_step
from .runs import Run, current_run
from .session import Session, current_session
from .tool_memory import ToolProfile
from .stores import SQLiteStore, Store, store_from_url
from .stores.base import SessionRecord

F = TypeVar("F", bound=Callable[..., Any])


def _clip(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


class Memory:
    """The Log, Working memory, Recaps and the UserProfile behind one handle.

    `store` is a `Store` or a URL (`sqlite:///mem.db`, `sqlite://:memory:`).
    `engine` is an `Engine` or a name; the default `"local"` engine shares
    the store's SQLite file, so one path is the whole setup.
    """

    def __init__(
        self,
        store: Union[str, Store] = "sqlite://:memory:",
        engine: Union[str, Engine] = "local",
        *,
        project: Optional[str] = None,
        policy: Optional[MemoryPolicy] = None,
        session_ttl: Optional[float] = None,
        world_fact_approval: bool = True,
    ) -> None:
        self.store = store_from_url(store) if isinstance(store, str) else store
        if isinstance(engine, str):
            options: dict[str, Any] = {}
            if engine == "local" and isinstance(self.store, SQLiteStore):
                options["path"] = self.store.path
            engine = engine_from_name(engine, **options)
        engine.bind(self.store)
        self.engine = engine
        self.knowledge = Knowledge(getattr(self.store, "path", ":memory:"))
        """Tool memory, world facts, feedback, negatives, checkpoints and
        steps -- projections of the event log (see `knowledge.py`)."""
        self.knowledge.catch_up(self.store)  # apply whatever an earlier process left unapplied
        self.worker_id = uuid.uuid4().hex
        """Names this handle when it claims a side-effecting step."""
        self.project = project
        """Names the `project:<name>` scope for world facts and negatives."""
        self.policy = policy or MemoryPolicy()
        """What is pushed per turn (see `MemoryPolicy`)."""
        self.policy_for_recall = MemoryPolicy(
            max_facts=50, max_world=50, max_lessons=20, recaps=5, min_recap_score=0.0, recent=0
        )
        """What `memory_recall` searches: wider, with no relevance threshold."""
        self.session_ttl = session_ttl
        """Default lifetime of a session in seconds (`None`: no expiry)."""
        self.world_fact_approval = world_fact_approval
        """Agent-written world facts wait for `confirm_fact` before anyone but
        the user or session that proposed them sees them (default). A world
        fact reaches every user in its scope; without approval, one
        conversation could change what everyone is told."""

    @property
    def project_scope(self) -> str:
        return f"project:{self.project}" if self.project else "global"

    # -- sessions -----------------------------------------------------------

    def session(
        self,
        session_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        ttl: Optional[float] = None,
        run_id: Optional[str] = None,
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
        return Session(self, record, run_id=run_id)

    def run(self, run_id: Optional[str] = None) -> Run:
        """Tag every event written inside it with `run_id` (generated when
        omitted): `with mem.run("deploy-42"): ...`. See `runs.py`."""
        return Run(run_id)

    # -- events -------------------------------------------------------------

    def record(self, event: MemoryEvent) -> MemoryEvent:
        """Record `event` in the current session if one is open, else
        straight to the log with whatever envelope the event carries."""
        session = current_session()
        if session is not None and session.memory is self and not session.closed:
            return session.record(event)
        return self._append(event)

    def _append(self, event: MemoryEvent) -> MemoryEvent:
        """Log `event` (tagged with the active run, if it has no `run_id`),
        then bring the projections up to date with the log; return what was
        logged. A duplicate delivery (same event id) is not logged again, so
        it is never counted twice; catching up still runs, so an event an
        earlier process logged but never applied is applied now."""
        run_id = current_run()
        if run_id is not None and event.run_id is None:
            event = event.bind(run_id=run_id)
        self.store.append(event)
        self.knowledge.catch_up(self.store)
        return event

    def step(
        self,
        fn: Optional[F] = None,
        *,
        name: Optional[str] = None,
        key: Optional[Callable[..., Any]] = None,
        side_effects: bool = False,
        lease: float = 300.0,
    ) -> Any:
        """Decorator: make a function a resumable step (see `resume.py`).
        Inside a session, a call that already finished in an earlier run of
        the same session returns its stored result instead of running.
        Outside a session it is a plain call. A `side_effects=True` step is
        claimed before it runs, so only one worker ever runs it; `lease` is
        how many seconds the claim tells other workers it may still be
        running."""

        def decorate(func: F) -> F:
            return make_step(self, func, name=name, key=key, side_effects=side_effects,  # type: ignore[return-value]
                             lease=lease)

        return decorate(fn) if fn is not None else decorate

    def agent_tools(self, session: Session, *, writes: bool = True) -> list[Callable[..., str]]:
        """Memory tools bound to `session`, for agents to pull memory and
        (unless `writes=False`) write it -- see `agent_tools.py`."""
        return build_agent_tools(session, writes=writes)

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
        recaps: int = 3,
    ) -> MemoryContext:
        """Long-term context without a session (user facts, recaps,
        negatives, world facts, lessons). Inside a session, prefer
        `Session.context`, which adds Working memory."""
        from dataclasses import replace

        scopes = ["global"] + ([self.project_scope] if self.project else []) \
            + ([f"agent:{agent}"] if agent else []) + ([f"user:{user_id}"] if user_id else [])
        return await self._long_term(
            query, user_id=user_id, agent=agent, policy=replace(self.policy, recaps=recaps), scopes=scopes
        )

    async def _long_term(
        self,
        query: Optional[str],
        *,
        user_id: Optional[str],
        agent: Optional[str],
        policy: MemoryPolicy,
        scopes: list[str],
        local_only: bool = False,
    ) -> MemoryContext:
        engine = getattr(self.engine, "index", self.engine) if local_only else self.engine
        found = (
            await engine.search_recaps(query or "", user_id=user_id, agent=agent, limit=policy.recaps)
            if policy.recaps
            else []
        )
        found = [e for e in found if (e.score or 0.0) >= policy.min_recap_score]
        facts = self._visible_facts(user_id, await engine.user_profile(user_id)) if user_id is not None else []
        world = self.world_facts([s for s in scopes if not s.startswith(("user:", "agent:"))])
        if query and len(world) > policy.max_world:
            words = set(query.lower().split())
            world.sort(key=lambda w: -len(words & set(f"{w.key} {w.value}".lower().split())))
        return MemoryContext(
            negatives=self.knowledge.negatives(scopes)[: policy.max_negatives],
            facts=facts[: policy.max_facts],
            world=world[: policy.max_world],
            lessons=self.knowledge.lessons(user_id=user_id, agent=agent, limit=policy.max_lessons),
            tool_notes=self.tool_notes(agent),
            recaps=found,
            policy=policy,
        )

    def _visible_facts(self, user_id: str, facts: list[Fact]) -> list[Fact]:
        """`facts` minus what the user asked to forget. A tombstone hides
        facts from before it; a fact stated again afterwards shows again."""
        stones = self.knowledge.tombstones(user_id)
        if not stones:
            return facts
        out = []
        for f in facts:
            hidden = False
            for key, pattern, ts in stones:
                before = f.valid_from <= ts
                if f.key == key and before:
                    hidden = True
                elif pattern and pattern.lower() in f.value.lower() and (before or f.valid_from == 0):
                    hidden = True
            if not hidden:
                out.append(f)
        return out

    # -- structured kinds ---------------------------------------------------

    def learn(self, key: str, value: str, *, scope: Optional[str] = None, ttl: Optional[float] = None,
              source: str = "explicit") -> MemoryEvent:
        """Record a world fact outside any session (see `Session.learn`)."""
        return self.record(self._world_fact(key, value, scope=scope, ttl=ttl, source=source))

    def _world_fact(self, key: str, value: str, *, scope: Optional[str], ttl: Optional[float],
                    source: str) -> WorldFact:
        # Decided now and recorded in the event, so rebuilding the
        # projections from the log gives the same answer under any setting.
        proposed = self.world_fact_approval and source_rank(source) <= 1
        return WorldFact(key=key, value=value, scope=scope or self.project_scope,
                         valid_until=time.time() + ttl if ttl else None, source=source, proposed=proposed)

    def pending_world_facts(self, scopes: Optional[list[str]] = None) -> list[WorldRecord]:
        """World facts waiting for review, oldest first: proposals (agent
        writes) and disputed facts (a less trusted source contradicting a
        current fact). Settle each with `confirm_fact` or `reject_fact`."""
        return self.knowledge.pending_world_facts(scopes)

    def _pending(self, fact_id: str) -> WorldRecord:
        record = self.knowledge.world_fact(fact_id)
        if record is None or not record.pending:
            raise ValueError(f"commonmemo: no pending world fact {fact_id!r} (see pending_world_facts())")
        return record

    def confirm_fact(self, fact_id: str, *, value: Optional[str] = None) -> MemoryEvent:
        """Approve a pending world fact: it becomes current for everyone in
        its scope, with an operator's rank. `value` corrects it on the way."""
        record = self._pending(fact_id)
        return self._append_outside_session(WorldFactConfirmed(
            fact_id=fact_id, key=record.key, value=record.value if value is None else value,
            scope=record.scope, valid_until=record.valid_until,
        ))

    def reject_fact(self, fact_id: str, *, reason: Optional[str] = None) -> MemoryEvent:
        """Turn down a pending world fact. It stays in the history."""
        self._pending(fact_id)
        return self._append_outside_session(WorldFactRejected(fact_id=fact_id, reason=reason))

    def retract_fact(self, key: str, *, scope: Optional[str] = None, reason: Optional[str] = None) -> MemoryEvent:
        """`key` is no longer true and nothing replaces it ("we no longer
        offer refunds"). History is kept."""
        return self._append_outside_session(
            WorldFactRetracted(key=key, scope=scope or self.project_scope, reason=reason)
        )

    def _append_outside_session(self, event: MemoryEvent) -> MemoryEvent:
        # Review actions are an operator's, not part of whatever session
        # happens to be current, so they never take a session's envelope.
        return self._append(event)

    def world_facts(self, scopes: Optional[list[str]] = None, *, include_stale: bool = False) -> list[WorldRecord]:
        """Current world facts in `scopes` (default: global + the project)."""
        scopes = scopes or (["global"] + ([self.project_scope] if self.project else []))
        return self.knowledge.world_facts(scopes, include_stale=include_stale)

    def never(self, statement: str, *, scope: str = "global", reason: Optional[str] = None) -> MemoryEvent:
        """Record a negative outside a session; `scope` is a full scope
        string (`global`, `project:x`, `agent:x`, `user:x`)."""
        user_id = scope.split(":", 1)[1] if scope.startswith("user:") else None
        return self.record(Negative(statement=statement, scope=scope, reason=reason, user_id=user_id))

    def revoke_negative(self, negative_id: str) -> MemoryEvent:
        return self.record(NegativeRevoked(negative_id=negative_id))

    def negatives(self, scopes: list[str]) -> list[NegativeRecord]:
        return self.knowledge.negatives(scopes)

    def feedback_summary(self, *, user_id: Optional[str] = None, agent: Optional[str] = None) -> FeedbackSummary:
        return self.knowledge.feedback_summary(user_id=user_id, agent=agent)

    def tool_profile(self, tool: str, agent: Optional[str] = None) -> ToolProfile:
        """What tool memory knows about `tool` (for `agent`, or all agents)."""
        return self.knowledge.tool_profile(tool, agent)

    def tool_notes(self, agent: Optional[str] = None) -> dict[str, list[str]]:
        """`{tool: advice lines}` for every tool with something worth saying.
        An agent's own experience comes first, then what other agents saw."""
        out: dict[str, list[str]] = {}
        for tool in self.knowledge.tools():
            notes = self.tool_profile(tool, agent).notes() if agent else []
            for line in self.tool_profile(tool).notes():
                if line not in notes:
                    notes.append(line)
            if notes:
                out[tool] = notes
        return out

    # -- history (ADR 0004) --------------------------------------------------

    def history(self) -> History:
        """Read-only metrics over the Log: velocity, source mix, churn,
        review flow, staleness, growth, and what one run wrote."""
        return History(self)

    def snapshot(self, label: str) -> MemoryEvent:
        """Name the current log position, for `as_of` and `diff`. Nothing is
        copied: the log up to here is the snapshot."""
        return self._append_outside_session(Snapshot(label=label))

    def as_of(self, point: Point) -> MemoryView:
        """The kinds CommonMemo owns as they were at `point`: a log position
        (`int`), a time (`float`, epoch seconds) or a snapshot label (`str`).
        Replayed from the Log into a scratch projection; data forgotten since
        is gone from it too."""
        return as_of(self, point)

    def diff(self, since: Point, until: Point = None) -> Diff:
        """What changed between two points (`until` defaults to now)."""
        return diff(self, since, until)

    def rebuild_knowledge(self) -> None:
        """Recompute the structured projections from the event log."""
        self.knowledge.rebuild(self.store)

    # -- lifecycle ----------------------------------------------------------

    async def forget_user(self, user_id: str) -> int:
        """Delete everything about `user_id` from every tier and kind. Tool
        memory (argument shapes and scrubbed error signatures only) and world
        facts are not about the user and stay. Returns rows removed."""
        sessions = self.store.session_ids(user_id)
        n = self.knowledge.forget_user(user_id, sessions)
        return n + self.store.forget_user(user_id) + await self.engine.forget_user(user_id)

    def prune(self, *, older_than_days: float) -> int:
        """Drop events older than `older_than_days` and expired sessions.
        Recaps and facts derived from them are kept (they carry their own
        provenance ids)."""
        return self.store.prune(older_than=time.time() - older_than_days * 86400)

    async def aclose(self) -> None:
        await self.engine.aclose()
        self.knowledge.close()
        self.store.close()


__all__ = ["Memory", "LocalEngine"]
