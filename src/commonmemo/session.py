"""`Session` -- one run: its Working memory, and the scope everything else hangs off.

A session is one conversation or one workflow run. While it is open:

- every event recorded through it (or through `@mem.track` while it is the
  current session) is stamped with its `session_id`, `user_id` and `agent`
  and appended to the Log;
- `session.state` is working memory, persisted on every write, so a
  session resumed by id (even from another process or harness) continues
  where it stopped.

Closing it hands the session's events to the engine for consolidation into
a recap and UserProfile facts.

Beyond the four tiers, a session is where the structured kinds are written:
`learn` (world facts), `feedback` / `correct`, `never` (negatives),
`forget` (tombstones) and `checkpoint` (resume points) -- see
`knowledge.py` for how each is kept.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .context import MemoryContext, MemoryPolicy
from .engines.base import Recap, Fact
from .events import (
    Checkpoint,
    ContextPushed,
    Feedback,
    Forgotten,
    MemoryEvent,
    Message,
    Negative,
    StepFailed,
    StepFinished,
    UserFact,
)
from .engines.local import content_words
from .resume import ResumePoint
from .stores.base import SessionRecord

if TYPE_CHECKING:
    from .memory import Memory

REFRESH_QUERY_CHARS = 2000
"""How much of the new text `Session.refresh` looks at."""

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

    def __init__(
        self, memory: "Memory", record: SessionRecord, *, run_id: Optional[str] = None, agent: Optional[str] = None
    ) -> None:
        self.memory = memory
        self._record = record
        self._agent = agent or record.agent
        """Who is working in this handle: the agent that resumed the session
        (another harness, say), else the one that opened it."""
        self.run_id = run_id
        """Tags this session's events; else the active `mem.run(...)`, if any."""
        self.state = SessionState(self, record.state)
        self.recaps: list[Recap] = []
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
        return self._agent

    @property
    def closed(self) -> bool:
        return self._closed

    # -- writing ------------------------------------------------------------

    def record(self, event: MemoryEvent) -> MemoryEvent:
        """Append `event` to the log, stamped with this session's envelope."""
        bound = self._bound(event)
        self.memory._append(bound)
        return bound

    def _bound(self, event: MemoryEvent) -> MemoryEvent:
        """`event` stamped with this session's envelope, not yet recorded."""
        if self._closed:
            raise RuntimeError(
                f"commonmemo: session {self.session_id!r} is closed; open a new one "
                "or resume it with Memory.session(session_id=...) before recording"
            )
        from .runs import current_run

        return event.bind(session_id=self.session_id, user_id=self.user_id, agent=self.agent,
                          run_id=self.run_id or current_run())

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

    def learn(
        self,
        key: str,
        value: str,
        *,
        scope: Optional[str] = None,
        ttl: Optional[float] = None,
        source: str = "explicit",
    ) -> MemoryEvent:
        """Record a world fact (not about the user). `scope` defaults to the
        memory's project (`project:<name>`), else `global`. `ttl` in seconds
        marks it time-sensitive. A fact from an agent (`source="agent"`) is a
        proposal until `Memory.confirm_fact`, unless the memory was opened
        with `world_fact_approval=False`: until then only this user (or, with
        no user, this session) sees it."""
        return self.record(self.memory._world_fact(key, value, scope=scope, ttl=ttl, source=source))

    def feedback(
        self,
        rating: Optional[float] = None,
        comment: Optional[str] = None,
        *,
        target_event_id: Optional[str] = None,
    ) -> MemoryEvent:
        """Record a rating in [-1, 1] (or a thumbs: 1 / -1) and/or a comment
        on the agent's work."""
        if rating is not None and not -1.0 <= rating <= 1.0:
            raise ValueError("commonmemo: feedback rating must be in [-1, 1]")
        kind = "rating" if rating is not None else "comment"
        return self.record(Feedback(feedback_kind=kind, rating=rating, comment=comment, target_event_id=target_event_id))

    def correct(self, wrong: str, right: str, *, comment: Optional[str] = None) -> MemoryEvent:
        """Record a correction: "not `wrong`, but `right`"."""
        return self.record(Feedback(feedback_kind="correction", wrong=wrong, right=right, comment=comment, rating=-1.0))

    def never(self, statement: str, *, reason: Optional[str] = None, scope: str = "user") -> MemoryEvent:
        """Record a standing "never do this" rule. `scope` is `user` (this
        user), `agent` (this agent, every user), `project`, or `global`."""
        resolved = {
            "user": f"user:{self.user_id}" if self.user_id else None,
            "agent": f"agent:{self.agent}" if self.agent else None,
            "project": self.memory.project_scope if self.memory.project else None,
            "global": "global",
        }.get(scope)
        if resolved is None:
            raise ValueError(f"commonmemo: never(scope={scope!r}) needs a session with that scope set")
        return self.record(Negative(statement=statement, scope=resolved, reason=reason))

    def forget(self, key: str, *, pattern: Optional[str] = None) -> MemoryEvent:
        """Forget the user's fact `key` (and any extracted fact containing
        `pattern`), and keep it from being re-learned from old sessions."""
        if self.user_id is None:
            raise ValueError("commonmemo: forget() needs a user")
        return self.record(Forgotten(key=key, pattern=pattern))

    def checkpoint(self, label: str, data: Optional[dict[str, Any]] = None) -> MemoryEvent:
        """Record a restart point: what a process needs to continue from
        here. `data` must be JSON-serializable."""
        return self.record(Checkpoint(label=label, data=dict(data or {})))

    def resume_point(self) -> ResumePoint:
        """Where this session's process stands: latest checkpoint, steps that
        completed, were interrupted or failed, and events since the
        checkpoint."""
        k = self.memory.knowledge
        cp = k.latest_checkpoint(self.session_id)
        steps = k.steps(self.session_id)
        since = self.events(since=cp.ts) if cp else []
        return ResumePoint(
            session_id=self.session_id,
            checkpoint=cp,
            completed=[s for s in steps if s.status == "finished"],
            in_doubt=[s for s in steps if s.status == "started"],
            failed=[s for s in steps if s.status == "failed"],
            since_checkpoint=[e for e in since if not (cp and e.id == cp.event_id)],
        )

    def resolve_step(self, step_id: str, *, result: Any = None, failed: bool = False) -> MemoryEvent:
        """Settle an interrupted step by hand: it took effect (`result`, which
        later runs replay), or it did not (`failed=True`, so it runs again)."""
        import json

        from .resume import settle_step

        if failed:
            return settle_step(self, step_id, StepFailed(step_id=step_id, error="resolved by hand: did not take effect"))
        return settle_step(self, step_id, StepFinished(step_id=step_id, result=json.dumps(result), replayable=True))

    def events(self, **filters: Any) -> list[MemoryEvent]:
        """This session's events, oldest first (filters as `Store.events`)."""
        return self.memory.store.events(session_id=self.session_id, **filters)

    # -- reading ------------------------------------------------------------

    @property
    def scopes(self) -> list[str]:
        """The scopes this session can see, broadest first."""
        out = ["global"]
        if self.memory.project:
            out.append(self.memory.project_scope)
        if self.agent:
            out.append(f"agent:{self.agent}")
        if self.user_id:
            out.append(f"user:{self.user_id}")
        return out

    async def context(
        self,
        query: Optional[str] = None,
        *,
        policy: Optional[MemoryPolicy] = None,
        recaps: Optional[int] = None,
        recent: Optional[int] = None,
        local_only: bool = False,
    ) -> MemoryContext:
        """What memory pushes for a turn about `query` (see `MemoryPolicy`):
        negatives, user facts, world facts, lessons and tool notes (stable);
        relevant past recaps, session state, recent events and where a
        resumed process stands (turn). `recaps` / `recent` override the
        policy. `local_only` skips the wrapped engine's own search."""
        policy = policy or self.memory.policy
        if recaps is not None or recent is not None:
            from dataclasses import replace

            policy = replace(
                policy,
                **({"recaps": recaps} if recaps is not None else {}),
                **({"recent": recent} if recent is not None else {}),
            )
        ctx = await self.memory._long_term(
            query, user_id=self.user_id, agent=self.agent, policy=policy,
            scopes=self.scopes, local_only=local_only,
        )
        ctx.recaps = [e for e in ctx.recaps if e.session_id != self.session_id]
        # Facts stated in this session reach the engine only at close, but
        # they are true now: overlay them (latest per key wins), then apply
        # tombstones again -- a fact forgotten in this session stays gone.
        if self.user_id is not None:
            facts = {f.key: f for f in ctx.facts}
            for e in self.events(kinds=[UserFact.kind]):
                facts[e.key] = Fact(
                    user_id=self.user_id, key=e.key, value=e.value,
                    valid_from=e.ts, source_event_ids=(e.id,),
                )
            ctx.facts = self.memory._visible_facts(self.user_id, sorted(facts.values(), key=lambda f: f.key))
            ctx.facts = ctx.facts[: policy.max_facts]
        # This user's (or, with no user, this session's) own unapproved world
        # facts: true for them as far as they told us, never for anyone else.
        world_scopes = [s for s in self.scopes if not s.startswith(("user:", "agent:"))]
        pending = self.memory.knowledge.pending_world_facts(
            world_scopes, user_id=self.user_id, session_id=None if self.user_id else self.session_id
        )
        ctx.world = ctx.world + [w for w in pending if w.status == "proposed"]
        ctx.state = self.state.to_dict()
        ctx.recent = self.events(limit=policy.recent) if policy.recent else []
        resume = self.resume_point()
        ctx.resume = None if resume.is_fresh else resume
        return ctx

    async def push_turn(self, query: str, **kwargs: Any) -> MemoryContext:
        """`context(query, ...)` for the start of a turn, recorded as pushed,
        so later `refresh` calls in the turn push only what's new."""
        ctx = await self.context(query, **kwargs)
        self.record(ContextPushed(reason="turn", recap_ids=tuple(r.id for r in ctx.recaps),
                                  chars=len(ctx.turn_text())))
        return ctx

    async def refresh(self, text: str) -> Optional[MemoryContext]:
        """Memory to push mid-run, after `text` (a tool result, the agent's
        own message) came up: recaps of earlier sessions it shares at least
        `MemoryPolicy.refresh_min_overlap` words with, not pushed yet this
        turn. None when there is nothing new, refresh is off, or the turn
        used its `max_refreshes`. Pushes are recorded (`ContextPushed`)."""
        policy = self.memory.policy
        if not policy.refresh or not text.strip():
            return None
        pushes = self._pushes_this_turn()
        if sum(p.reason == "refresh" for p in pushes) >= policy.max_refreshes:
            return None
        pushed = {rid for p in pushes for rid in p.recap_ids}
        from dataclasses import replace

        query = text[:REFRESH_QUERY_CHARS]
        # It runs after every tool call, so it searches the local recap index
        # only: no graph or LLM round trip per step, whatever the engine.
        found = await self.memory._long_term(
            query, user_id=self.user_id, agent=self.agent, scopes=self.scopes, local_only=True,
            policy=replace(policy, recaps=max(5, policy.recaps), min_recap_score=0.0),
        )
        words = content_words(query)
        new = [r for r in found.recaps
               if r.id not in pushed and r.session_id != self.session_id
               and len(words & content_words(r.summary)) >= policy.refresh_min_overlap][: policy.recaps]
        if not new:
            return None
        ctx = MemoryContext(recaps=new, policy=policy, agent=self.agent)
        self.record(ContextPushed(reason="refresh", recap_ids=tuple(r.id for r in new), chars=len(ctx.turn_text())))
        return ctx

    def _pushes_this_turn(self) -> list[ContextPushed]:
        pushes = self.events(kinds=[ContextPushed.kind])
        starts = [i for i, p in enumerate(pushes) if p.reason == "turn"]
        return pushes[starts[-1]:] if starts else pushes

    # -- closing ------------------------------------------------------------

    async def aclose(self, *, consolidate: bool = True) -> list[Recap]:
        """Close the session and, unless `consolidate=False`, hand its
        events to the engine. Closing twice is a no-op."""
        if self._closed:
            return self.recaps
        self._closed = True
        closed_at = time.time()
        self.memory.store.close_session(self.session_id, closed_at)
        self._record.closed_at = closed_at
        if consolidate:
            self.recaps = await self.memory.engine.ingest(self._record, self.events())
        return self.recaps

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
