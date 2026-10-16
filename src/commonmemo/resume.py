"""Resume memory: continue a stopped process from where it was.

Two pieces, both scoped to a session (resume one with
`Memory.session(session_id=...)`, from any process):

- **checkpoints** -- `session.checkpoint(label, data)` records what the
  process needs to carry on from here. `session.resume_point()` returns the
  latest one, plus everything that happened after it.
- **steps** -- `@mem.step` makes a function's completed calls replayable:
  re-running the same code in a resumed session returns each finished
  step's stored result instead of running it again, so the process picks up
  at the first step that did not finish -- "start as is".

A step that **started but never finished** (the process died inside it) is
*in doubt*: it may or may not have had its effect. A pure step is simply
run again. A step declared `side_effects=True` -- a payment, an email --
raises `StepInDoubt` instead, because running it twice could do the thing
twice; the application decides (check the external system, then
`resolve_step`).

**The guarantee for `side_effects=True`:** across every worker and process
sharing the store, the function body runs at most once per step until the
step is settled. Before running, a worker *claims* the step: one atomic
store write that records `StepStarted` only if nobody holds the step. Of
two workers resuming one session at once, exactly one wins; the other gets
`StepInProgress` (a `StepInDoubt`) while the winner's lease is live, and
`StepInDoubt` once it has expired. An expired lease never means "run it
again": the holder may have died after the effect. The claim is released
only by `StepFinished` (a replayable result: settled for good) or
`StepFailed` (the function raised, or `resolve_step(failed=True)`), and a
step whose result could not be stored is released too, so it runs again
like any other unreplayable step. Pure steps are not claimed: two workers
may both run one, which by definition is harmless.

A step's identity is its name plus its arguments (hashed), so a loop over
different inputs records one step per input; pass `key=` to choose the
identity yourself. Results must be JSON-serializable to be replayed; a
result that is not is recorded as not replayable and the step runs again.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .events import MemoryEvent, StepFailed, StepFinished, StepStarted

if TYPE_CHECKING:
    from .knowledge import CheckpointRecord, StepRecord
    from .memory import Memory


class StepInDoubt(RuntimeError):
    """A side-effecting step started in an earlier run and never recorded
    finishing: running it again might repeat its effect."""

    def __init__(self, session_id: str, step_id: str) -> None:
        super().__init__(
            f"commonmemo: step {step_id!r} of session {session_id!r} started before but never finished, "
            "and it is declared side_effects=True -- running it again could repeat its effect. "
            "Check whether it took effect, then call session.resolve_step(step_id, result=...) "
            "(it did) or session.resolve_step(step_id, failed=True) (it did not, run it again)."
        )
        self.session_id = session_id
        self.step_id = step_id


class StepInProgress(StepInDoubt):
    """A side-effecting step is claimed by another worker whose lease has
    not expired: it may still be running. Wait and retry; once the lease
    expires without a result, the step is in doubt (`StepInDoubt`)."""

    def __init__(self, session_id: str, step_id: str, owner: str, lease_until: float) -> None:
        RuntimeError.__init__(
            self,
            f"commonmemo: step {step_id!r} of session {session_id!r} is claimed by worker {owner!r} until "
            f"{lease_until:.0f} (epoch seconds) and may still be running; retry later instead of running it again.",
        )
        self.session_id = session_id
        self.step_id = step_id
        self.owner = owner
        self.lease_until = lease_until


@dataclass
class ResumePoint:
    """Where a session's process stands."""

    session_id: str
    checkpoint: Optional["CheckpointRecord"] = None
    completed: list["StepRecord"] = field(default_factory=list)
    in_doubt: list["StepRecord"] = field(default_factory=list)
    failed: list["StepRecord"] = field(default_factory=list)
    since_checkpoint: list[MemoryEvent] = field(default_factory=list)
    """Events recorded after the latest checkpoint -- work done since."""

    @property
    def is_fresh(self) -> bool:
        """Nothing to resume: no checkpoint and no step ever ran."""
        return self.checkpoint is None and not (self.completed or self.in_doubt or self.failed)

    def lines(self) -> list[str]:
        out = []
        if self.checkpoint is not None:
            data = json.dumps(self.checkpoint.data, default=str, sort_keys=True)
            out.append(f"Last checkpoint '{self.checkpoint.label}': {data[:400]}")
        if self.completed:
            out.append("Completed steps (do not redo): " + ", ".join(s.step_id for s in self.completed[-10:]))
        if self.in_doubt:
            out.append("Interrupted steps (may or may not have taken effect): "
                       + ", ".join(s.step_id for s in self.in_doubt))
        if self.failed:
            out.append("Failed steps: " + ", ".join(f"{s.step_id} ({s.error})" for s in self.failed[-5:]))
        if self.since_checkpoint:
            out.append(f"{len(self.since_checkpoint)} events recorded since the last checkpoint")
        return out


def _step_id(name: str, key: Optional[Callable[..., Any]], args: tuple, kwargs: dict) -> str:
    if key is not None:
        return f"{name}:{key(*args, **kwargs)}"
    if not args and not kwargs:
        return name
    digest = hashlib.sha256(
        json.dumps([args, kwargs], default=repr, sort_keys=True).encode()
    ).hexdigest()[:12]
    return f"{name}:{digest}"


def _claim_key(session_id: str, step_id: str) -> str:
    return f"step:{session_id}:{step_id}"


def _held(memory: "Memory", session_id: str, step_id: str) -> StepInDoubt:
    """The error for a side-effecting step someone else started."""
    claim = memory.store.get_claim(_claim_key(session_id, step_id))
    if claim is not None and claim.status == "held" and claim.lease_until is not None \
            and claim.lease_until > time.time():
        return StepInProgress(session_id, step_id, claim.owner, claim.lease_until)
    return StepInDoubt(session_id, step_id)


def settle_step(session: Any, step_id: str, event: MemoryEvent) -> MemoryEvent:
    """Record `StepFinished` or `StepFailed` for `step_id` and release or
    keep its claim to match, in one store write."""
    memory = session.memory
    bound = session._bound(event)
    done = isinstance(bound, StepFinished) and bound.replayable
    memory.store.settle(_claim_key(session.session_id, step_id), session_id=session.session_id,
                        status="done" if done else "free", event=bound)
    memory.knowledge.catch_up(memory.store)
    return bound


def _encode(result: Any) -> tuple[Optional[str], bool]:
    try:
        return json.dumps(result), True
    except (TypeError, ValueError):
        return None, False


def make_step(
    memory: "Memory",
    func: Callable[..., Any],
    *,
    name: Optional[str],
    key: Optional[Callable[..., Any]],
    side_effects: bool,
    lease: float = 300.0,
) -> Callable[..., Any]:
    from .session import current_session

    step_name = name or func.__qualname__

    def plan(args: tuple, kwargs: dict) -> tuple[Any, Optional[str], str]:
        """(session, step_id, what to do): "plain" outside a session,
        "replay" when a stored result exists, "run" for a pure step,
        "claimed" when this worker now holds a side-effecting step."""
        session = current_session()
        if session is None or session.memory is not memory or session.closed:
            return None, None, "plain"  # outside a session: just a function call
        sid = _step_id(step_name, key, args, kwargs)
        rec = memory.knowledge.step(session.session_id, sid)
        if rec is not None and rec.status == "finished" and rec.replayable:
            return session, sid, "replay"
        if not side_effects:
            return session, sid, "run"
        if rec is not None and rec.status == "started":
            raise _held(memory, session.session_id, sid)
        started = session._bound(StepStarted(step_id=sid, side_effects=True))
        claim = memory.store.claim(_claim_key(session.session_id, sid), session_id=session.session_id,
                                   owner=memory.worker_id, lease_until=time.time() + lease, event=started)
        memory.knowledge.catch_up(memory.store)
        if claim is None:
            return session, sid, "claimed"
        rec = memory.knowledge.step(session.session_id, sid)
        if claim.status == "done" and rec is not None and rec.status == "finished" and rec.replayable:
            return session, sid, "replay"  # another worker finished it a moment ago
        raise _held(memory, session.session_id, sid)

    def replay(session: Any, sid: str) -> Any:
        rec = memory.knowledge.step(session.session_id, sid)
        return json.loads(rec.result) if rec.result is not None else None

    def begin(session: Any, sid: str, how: str) -> None:
        if how == "run":
            session.record(StepStarted(step_id=sid, side_effects=False))

    def finish(session: Any, sid: str, how: str, result: Any) -> None:
        encoded, ok = _encode(result)
        event = StepFinished(step_id=sid, result=encoded, replayable=ok)
        if how == "claimed":
            settle_step(session, sid, event)
        else:
            session.record(event)

    def fail(session: Any, sid: str, how: str, exc: BaseException) -> None:
        event = StepFailed(step_id=sid, error=f"{type(exc).__name__}: {exc}")
        if how == "claimed":
            settle_step(session, sid, event)
        else:
            session.record(event)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_step(*args: Any, **kwargs: Any) -> Any:
            session, sid, how = plan(args, kwargs)
            if how == "plain":
                return await func(*args, **kwargs)
            if how == "replay":
                return replay(session, sid)
            begin(session, sid, how)
            try:
                result = await func(*args, **kwargs)
            except BaseException as exc:
                fail(session, sid, how, exc)
                raise
            finish(session, sid, how, result)
            return result

        return async_step

    @functools.wraps(func)
    def step(*args: Any, **kwargs: Any) -> Any:
        session, sid, how = plan(args, kwargs)
        if how == "plain":
            return func(*args, **kwargs)
        if how == "replay":
            return replay(session, sid)
        begin(session, sid, how)
        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            fail(session, sid, how, exc)
            raise
        finish(session, sid, how, result)
        return result

    return step


__all__ = ["ResumePoint", "StepInDoubt", "StepInProgress", "make_step", "settle_step"]
