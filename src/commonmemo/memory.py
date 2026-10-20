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
import logging
import re
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar, Union, overload

from .agent_tools import build_agent_tools
from .context import MemoryContext, MemoryPolicy
from .facts import FactKey, FactRegistry, ImportResult
from .engines import Engine, Fact, LocalEngine, Recap, engine_from_name
from .ranking import KeywordRanker, Ranker, rank
from .events import (
    MemoryEvent,
    Negative,
    NegativeRevoked,
    ProcedureConfirmed,
    ProcedureEntry,
    ProcedureRejected,
    ProcedureRetired,
    ToolInvoked,
    UserFact,
    WorldFact,
    WorldFactConfirmed,
    WorldFactRejected,
    WorldFactRetracted,
    Snapshot,
    Freeze,
    Unfreeze,
)
from .knowledge import (
    FeedbackSummary,
    FreezeRecord,
    Knowledge,
    NegativeRecord,
    ProcedureOutcome,
    ProcedureRecord,
    WorldRecord,
    source_rank,
)
from .procedures import ProcedureImport, TaskProcedures, check_kind, match, parse_steps, scrub
from .reins import (
    REVERTIBLE,
    Alert,
    AlertPolicy,
    Budget,
    RevertResult,
    after_write,
    check_alerts,
    over_budget,
    revert,
)
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


if TYPE_CHECKING:
    from pathlib import Path

    from .traces import TurnTrace, Why

MORE_PROBE = 5
"""How many recaps past the push's limit are looked at, to count what was left out."""


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
        budgets: Optional[dict[str, Union[Budget, int]]] = None,
        alerts: Optional[AlertPolicy] = None,
        fact_keys: Optional[Iterable[FactKey]] = None,
        ranker: Optional[Ranker] = None,
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
            max_facts=50, max_world=50, max_lessons=20, recaps=5, min_recap_score=0.0, recent=0,
            recaps_across_agents=self.policy.recaps_across_agents,
        )
        """What `memory_recall` searches: wider, with no relevance threshold."""
        self.session_ttl = session_ttl
        """Default lifetime of a session in seconds (`None`: no expiry)."""
        self.ranker: Ranker = ranker or self.engine.ranker() or KeywordRanker()
        """Ranks facts and recall results against a request (see `ranking.py`):
        yours, else the engine's embedder, else shared words."""
        self.fact_keys = FactRegistry()
        """The optional world-fact key registry (see `facts.py`): what keys
        exist, their allowed values and lifetimes, and which are pinned."""
        self.fact_keys.register(*(fact_keys or ()))
        self.world_fact_approval = world_fact_approval
        """Agent-written world facts wait for `confirm_fact` before anyone but
        the user or session that proposed them sees them (default). A world
        fact reaches every user in its scope; without approval, one
        conversation could change what everyone is told."""
        self.budgets: dict[str, Budget] = {
            src: b if isinstance(b, Budget) else Budget(writes=b) for src, b in (budgets or {}).items()
        }
        """Per-source caps on direct world-fact writes (`{"agent": 100}` is 100
        an hour); writes past a cap become proposals. None by default."""
        self.alert_policy = alerts or AlertPolicy()
        """When memory warns: on by default, logged to `commonmemo.alerts`."""
        self._last_alert_check = 0.0
        self.listeners: list[Callable[[MemoryEvent], None]] = []
        """Called with every event after it is logged (exporters use this,
        e.g. `otel.instrument`). A listener that raises is logged and skipped."""

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
        task: Optional[str] = None,
    ) -> Session:
        """Open a new session, or resume an existing one by `session_id`.
        `task` names the type of task it does (`weekly_report`), so its
        procedures are pushed without matching (see `procedures.py`).

        Resuming checks ownership: a session that belongs to one user can
        never be resumed as another (that would leak one user's working
        memory into another's turn). A resumed session's new events carry
        the resuming `agent`, so one session continued across harnesses
        shows who did what.
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
        return Session(self, record, run_id=run_id, agent=agent, task=task)

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
        after_write(self, event)
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 -- an exporter must never break a write
                logging.getLogger("commonmemo").exception("commonmemo: event listener failed")
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
        exclude_session: Optional[str] = None,
        all_facts: bool = False,
        task: Optional[str] = None,
        procedures: bool = True,
    ) -> MemoryContext:
        """`all_facts` leaves the profile uncut, for a caller that adds facts
        to it and cuts it itself (`Session.context`). `task` names the task,
        else the request is matched against the known ones; `procedures=False`
        skips that (a mid-run refresh has no use for it)."""
        engine = getattr(self.engine, "index", self.engine) if local_only else self.engine
        # A user's recaps are theirs whichever agent (harness) they worked
        # in; without a user, an agent's recaps stay its own.
        recap_agent = None if user_id is not None and policy.recaps_across_agents else agent
        # Look a little past the limit, to tell the agent what was left out.
        found = (
            await engine.search_recaps(query or "", user_id=user_id, agent=recap_agent,
                                       limit=policy.recaps + MORE_PROBE)
            if policy.recaps
            else []
        )
        if user_id is not None:
            found = self._visible_recaps(user_id, found)
        found = [e for e in found if e.session_id != exclude_session]
        shown = [e for e in found if (e.score or 0.0) >= policy.min_recap_score][: policy.recaps]
        near = [e for e in found if e not in shown and (e.score or 0.0) >= policy.min_recap_score / 2]
        facts = self._visible_facts(user_id, await engine.user_profile(user_id)) if user_id is not None else []
        world = self.world_facts([s for s in scopes if not s.startswith(("user:", "agent:"))])
        world_turn: list[WorldRecord] = []
        world_more = len(world) - policy.max_world
        if self.fact_keys:
            # With a registry, only pinned facts go in the stable half; the
            # rest are ranked by the request and pushed with the turn, so the
            # stable half stays small (and cached) however many facts exist.
            pinned = [w for w in world if self.fact_keys.pinned(w.key)]
            unpinned = [w for w in world if not self.fact_keys.pinned(w.key)]
            matched, _ = await rank(self.ranker, query, unpinned, self._world_text)
            world, world_turn = pinned, matched[: policy.max_world]
            world_more = max(0, len(pinned) - policy.max_world) + len(unpinned) - len(world_turn)
        elif len(world) > policy.max_world:
            related, rest = await rank(self.ranker, query, world, self._world_text)
            world = related + rest  # the ones the request is about make the cut first
        negatives = self.knowledge.negatives(scopes)
        lessons = self.knowledge.lessons(user_id=user_id, agent=agent, limit=10**6)
        procedure = await self._procedure_for(query, task, user_id, policy) if procedures else None
        return MemoryContext(
            procedure=procedure,
            negatives=negatives[: policy.max_negatives],
            facts=facts if all_facts else await self._pick_facts(query, facts, policy.max_facts),
            world=world[: policy.max_world],
            world_turn=world_turn,
            lessons=lessons[: policy.max_lessons],
            tool_notes=self.tool_notes(agent),
            recaps=shown,
            policy=policy,
            agent=agent,
            more={
                "recaps": len(near),
                "facts": len(facts) - policy.max_facts,
                "world": world_more,
                "negatives": len(negatives) - policy.max_negatives,
                "lessons": len(lessons) - policy.max_lessons,
            },
        )

    async def _procedure_for(self, query: Optional[str], task: Optional[str], user_id: Optional[str],
                             policy: MemoryPolicy) -> Optional[TaskProcedures]:
        """The task the turn is about: named, else matched by its `use_when`
        lines (ADR 0003). None when there is none."""
        if task is not None:
            found = self.procedures(task, user_id=user_id)
            return found if found else None
        entries = self.knowledge.procedure_entries(self._procedure_scopes(user_id))
        hit = await match(self.ranker, query, entries, policy.min_procedure_score)
        if hit is None:
            return None
        found = self.procedures(hit[0], user_id=user_id)
        found.how, found.score = "matched", round(hit[1], 3)
        return found

    def _world_text(self, w: WorldRecord) -> str:
        spec = self.fact_keys.get(w.key)
        return f"{w.key} {w.value} {spec.description if spec else ''}"

    async def _pick_facts(self, query: Optional[str], facts: list[Fact], limit: int) -> list[Fact]:
        """The `limit` facts to push: all of them if they fit; else the ones
        related to the request first (#21), then the rest in order."""
        if len(facts) <= limit:
            return facts
        related, rest = await rank(self.ranker, query, facts, lambda f: f"{f.key} {f.value}")
        return (related + rest)[:limit]

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

    def _forgotten_values(self, user_id: str) -> list[str]:
        """Text the user asked to forget: each forgotten key's values from
        before its tombstone, and each tombstone pattern. A value stated
        again after the tombstone is current again, so it is not included."""
        stones = self.knowledge.tombstones(user_id)
        if not stones:
            return []
        stated = [(e.key, e.value, e.ts) for e in self.store.events(user_id=user_id, kinds=[UserFact.kind])]
        out: set[str] = set()
        for key, pattern, ts in stones:
            if pattern:
                out.add(pattern)
            old = {v for k, v, at in stated if k == key and at <= ts}
            again = {v for k, v, at in stated if k == key and at > ts}
            out |= old - again
        return sorted((v for v in out if v.strip()), key=len, reverse=True)

    def _visible_recaps(self, user_id: str, recaps: list[Recap]) -> list[Recap]:
        """`recaps` with what the user asked to forget redacted. A recap is
        a summary of a past session, so it can quote a fact that was later
        forgotten; without this, the fact would come back through it."""
        values = self._forgotten_values(user_id)
        if not values:
            return recaps
        pattern = re.compile("|".join(re.escape(v) for v in values), re.IGNORECASE)
        return [replace(r, summary=pattern.sub("[forgotten]", r.summary)) if pattern.search(r.summary) else r
                for r in recaps]

    # -- structured kinds ---------------------------------------------------

    def learn(self, key: str, value: str, *, scope: Optional[str] = None, ttl: Optional[float] = None,
              source: str = "explicit") -> MemoryEvent:
        """Record a world fact outside any session (see `Session.learn`)."""
        return self.record(self._world_fact(key, value, scope=scope, ttl=ttl, source=source))

    def _world_fact(self, key: str, value: str, *, scope: Optional[str], ttl: Optional[float],
                    source: str) -> WorldFact:
        # Decided now and recorded in the event, so rebuilding the
        # projections from the log gives the same answer under any setting.
        scope = scope or self.project_scope
        spec = self.fact_keys.get(key)
        trusted = source_rank(source) > 1
        if trusted and spec and spec.allowed and value not in spec.allowed:
            raise ValueError(f"commonmemo: {value!r} is not an allowed value for {key} ({', '.join(spec.allowed)})")
        note = None if trusted else self.fact_keys.check(key, value)
        proposed = (
            (self.world_fact_approval and not trusted)
            or note is not None
            or self.knowledge.frozen(scope, source) is not None
            or over_budget(self, source)
        )
        if ttl is None and spec is not None and spec.lifetime:
            ttl = spec.lifetime  # a registered key goes stale by default
        return WorldFact(key=key, value=value, scope=scope, valid_until=time.time() + ttl if ttl else None,
                         source=source, proposed=proposed, note=note)

    def register_fact_key(self, key: str, description: str = "", *, allowed: Optional[Iterable[str]] = None,
                          lifetime: Union[None, float, str] = None, pinned: bool = False) -> FactKey:
        """Add a key to the world-fact registry (see `facts.py`)."""
        spec = FactKey(key, description, tuple(allowed) if allowed is not None else None, lifetime, pinned)
        self.fact_keys.register(spec)
        return spec

    def load_facts(self, source: Union[str, "Path", dict[str, Any]]) -> ImportResult:
        """Register keys and write facts from a JSON or YAML file (or a dict),
        skipping facts whose current value already matches (see `facts.py`)."""
        from .facts import load

        return load(self, source)

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

    # -- procedural memory (ADR 0003) --------------------------------------------

    def _procedure_scopes(self, user_id: Optional[str]) -> list[str]:
        """Broadest first, so the more specific entry wins."""
        return ["global"] + ([self.project_scope] if self.project else []) + ([f"user:{user_id}"] if user_id else [])

    def save_procedure(self, task: str, kind: str, name: str, body: str, *, use_when: str = "",
                       steps: Optional[Iterable[str]] = None, scope: Optional[str] = None,
                       source: str = "explicit") -> MemoryEvent:
        """Record how `task` is done: a `template`, `rule`, `procedure` or
        `exemplar` named `name`. A changed body is a new version. `steps`
        default to a procedure's numbered lines. From an agent or a learner
        (`source="agent"` / `"extracted"`), an entry for everyone is a
        proposal (`confirm_procedure`); in a user's scope, an agent's entry
        applies to that user at once. See `procedures.py`."""
        return self.record(self._procedure_entry(task, kind, name, body, use_when=use_when, steps=steps,
                                                 scope=scope or self.project_scope, source=source))

    def _procedure_entry(self, task: str, kind: str, name: str, body: str, *, use_when: str,
                         steps: Optional[Iterable[str]], scope: str, source: str) -> ProcedureEntry:
        check_kind(kind)
        rank = source_rank(source)
        if kind == "exemplar" and rank <= 1 and not scope.startswith("user:"):
            raise ValueError("commonmemo: an exemplar from an agent or a learner is the user's data: write it in "
                             "the user's scope; an operator shares it with confirm_procedure(..., scrubbed=True)")
        # Decided now and recorded, so a rebuild from the Log agrees.
        own = scope.startswith("user:") and source == "agent"
        proposed = rank <= 1 and not own
        steps = tuple(steps) if steps is not None else (parse_steps(body) if kind == "procedure" else ())
        # A user's own entry carries their id even outside a session, so
        # `forget_user` removes it from the Log too.
        user_id = scope.split(":", 1)[1] if scope.startswith("user:") else None
        return ProcedureEntry(task=task, entry_kind=kind, name=name, body=body.strip(), use_when=use_when.strip(),
                              steps=steps, scope=scope, source=source, proposed=proposed, user_id=user_id)

    def load_procedures(self, source: Any, *, scope: Optional[str] = None) -> ProcedureImport:
        """Write the entries in a directory of Markdown files, a JSON/YAML
        file, or a list of dicts, as an operator's (current at once). Entries
        whose current version already matches are skipped, so loading at
        every start is safe (see `procedures.py`)."""
        from .procedures import load

        return load(self, source, scope=scope)

    def procedures(self, task: str, *, user_id: Optional[str] = None) -> TaskProcedures:
        """How `task` is done, as `user_id` sees it: current entries, the most
        specific scope winning (user over project over global)."""
        return TaskProcedures(task, self.knowledge.procedure_entries(self._procedure_scopes(user_id), task))

    def tasks(self, *, user_id: Optional[str] = None) -> dict[str, str]:
        """`{task: its use_when lines}` for every task with current entries."""
        out: dict[str, set[str]] = {}
        for e in self.knowledge.procedure_entries(self._procedure_scopes(user_id)):
            out.setdefault(e.task, set()).update([e.use_when] if e.use_when else [])
        return {t: "; ".join(sorted(lines)) for t, lines in sorted(out.items())}

    def pending_procedures(self, task: Optional[str] = None) -> list[ProcedureRecord]:
        """Entries waiting for review: proposals, and entries a less trusted
        source wrote over a more trusted one. Settle each with
        `confirm_procedure` or `reject_procedure`."""
        return self.knowledge.pending_procedures(task)

    def confirm_procedure(self, entry_id: str, *, scope: Optional[str] = None, body: Optional[str] = None,
                          scrubbed: bool = False) -> MemoryEvent:
        """Approve a pending entry, or share a user's own entry more widely:
        it becomes current in `scope` (default: the entry's own scope, or the
        project for a user's entry) with an operator's rank. `body` corrects
        it on the way. An exemplar leaves its user's scope only with
        `scrubbed=True` (you checked it for personal data); the user's known
        profile values are redacted from it too."""
        record = self.knowledge.procedure_entry(entry_id)
        if record is None or not (record.pending or (record.status == "current" and record.scope.startswith("user:"))):
            raise ValueError(f"commonmemo: no pending or user-scoped procedure entry {entry_id!r} "
                             "(see pending_procedures())")
        own = record.scope.startswith("user:")
        target = scope or (self.project_scope if own and not record.pending else record.scope)
        text = record.body if body is None else body
        if own and not target.startswith("user:") and record.kind == "exemplar":
            if not scrubbed:
                raise ValueError("commonmemo: sharing an exemplar beyond its user needs scrubbed=True: check it for "
                                 "personal data first (pass a cleaned body= if needed)")
            text = scrub(text, self._profile_values(record.scope.split(":", 1)[1]))
        return self._append_outside_session(ProcedureConfirmed(
            entry_id=entry_id, task=record.task, entry_kind=record.kind, name=record.name, body=text,
            use_when=record.use_when, steps=record.steps, scope=target,
        ))

    def _profile_values(self, user_id: str) -> list[str]:
        from .agent_tools import run_sync

        values = [e.value for e in self.store.events(user_id=user_id, kinds=[UserFact.kind])]
        try:
            values += [f.value for f in run_sync(self.engine.user_profile(user_id))]
        except Exception:  # noqa: BLE001 -- the stated facts are redacted either way
            logging.getLogger("commonmemo").warning("commonmemo: couldn't read %s's profile to scrub", user_id)
        return values

    def reject_procedure(self, entry_id: str, *, reason: Optional[str] = None) -> MemoryEvent:
        """Turn down a pending entry. It stays in the history."""
        record = self.knowledge.procedure_entry(entry_id)
        if record is None or not record.pending:
            raise ValueError(f"commonmemo: no pending procedure entry {entry_id!r} (see pending_procedures())")
        return self._append_outside_session(ProcedureRejected(entry_id=entry_id, reason=reason))

    def retire_procedure(self, task: str, kind: str, name: str, *, scope: Optional[str] = None,
                         reason: Optional[str] = None) -> MemoryEvent:
        """`name` no longer applies to `task`. History is kept."""
        return self._append_outside_session(ProcedureRetired(
            task=task, entry_kind=check_kind(kind), name=name, scope=scope or self.project_scope, reason=reason,
        ))

    def procedure_outcomes(self, task: Optional[str] = None) -> list[ProcedureOutcome]:
        """Per entry version: how many runs used it, the feedback in those
        runs, and the steps runs skipped. Which version worked better."""
        return self.knowledge.procedure_outcomes(task)

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

    # -- reins (ADR 0004, phase 2) ---------------------------------------------

    def revert(self, *, to: Point = None, run_id: Optional[str] = None, source: Optional[str] = None,
               session_id: Optional[str] = None, kinds: Iterable[str] = REVERTIBLE, scope: Optional[str] = None,
               reason: Optional[str] = None, dry_run: bool = False) -> RevertResult:
        """Bring world facts and negatives back: `to=` a point (log position,
        time or snapshot), or undo a cause (`run_id=`, `source=`,
        `session_id=`). Appends compensating events tagged
        `run_id="revert:<id>"`; never deletes. `dry_run=True` only plans.
        See `reins.py`."""
        return revert(self, to=to, run_id=run_id, source=source, session_id=session_id, kinds=kinds,
                      scope=scope, reason=reason, dry_run=dry_run)

    def freeze(self, *, scope: Optional[str] = None, source: Optional[str] = None,
               reason: Optional[str] = None) -> MemoryEvent:
        """Until `unfreeze`, direct world-fact writes in `scope` from `source`
        (either may be None for "any") are recorded as proposals."""
        return self._append_outside_session(Freeze(scope=scope, source=source, reason=reason))

    def unfreeze(self, freeze_id: str) -> MemoryEvent:
        return self._append_outside_session(Unfreeze(freeze_id=freeze_id))

    def freezes(self) -> list[FreezeRecord]:
        """Active freezes."""
        return self.knowledge.freezes()

    def alerts(self, *, now: Optional[float] = None) -> list[Alert]:
        """Check now for velocity spikes, churn and a review backlog
        (thresholds in `alert_policy`)."""
        return check_alerts(self, now=now)

    def rebuild_knowledge(self) -> None:
        """Recompute the structured projections from the event log."""
        self.knowledge.rebuild(self.store)

    # -- lifecycle ----------------------------------------------------------

    def traces(self, *, session_id: Optional[str] = None, user_id: Optional[str] = None,
               limit: Optional[int] = None) -> list[TurnTrace]:
        """Turn traces, oldest first: what each push gave the agent, by
        reference, and what it left out (see `traces.py`)."""
        from .traces import traces

        return traces(self, session_id=session_id, user_id=user_id, limit=limit)

    def why(self, key: str, *, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[Why]:
        """Why the agent got `key` (a fact or world fact key, or an item id):
        the latest push of it in the session (or for the user), and the
        events it came from. None if it was never pushed."""
        from .traces import why

        return why(self, key, session_id=session_id, user_id=user_id)

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
