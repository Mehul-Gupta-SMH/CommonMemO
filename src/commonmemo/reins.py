"""Reins on memory: revert, freeze, budgets and alerts (ADR 0004, phase 2).

    mem.revert(to="before v3")                  # world facts and negatives back to a snapshot
    mem.revert(run_id="commonadk:r-8812")       # undo what one run wrote
    mem.revert(run_id="revert:<id>")            # undo a revert
    mem.freeze(source="agent")                  # agent world-fact writes become proposals
    mem.alerts()                                # velocity spikes, churn, review backlog

**A revert never deletes from the Log.** It works out what memory should be
and appends compensating events: an old value back, a retraction, a
negative re-added or revoked. Those events carry `run_id="revert:<id>"`, and a
`Reverted` event records who reverted what and why. So a revert is
audited, can itself be reverted, and rebuilds agree with live memory.

- **By time** (`to=`): the reverted kinds return to how they were at a point.
- **By cause** (`run_id=`, `source=`, `session_id=`): the Log is replayed
  *without* those events, and memory is brought to that state. Changes made
  later by others survive.

Reverts cover the kinds CommonMemo owns that make claims: world facts,
negatives and procedures (a restored entry is its next version). Tool memory and feedback are observations of what happened, so
they are never "reverted"; Recaps and the UserProfile are held by the
engine (ADR 0004 decision: owned kinds only). Because the replay uses
today's Log, nothing a user asked to forget can come back.

**Freeze** and **budgets** both turn direct world-fact writes into
proposals (ADR 0002) rather than dropping them, so nothing is lost and
everything waits for review.

**Alerts** are on by default: after writes, at most once per
`AlertPolicy.check_every` seconds, memory checks for velocity spikes, churn
and a review backlog. Alerts are logged to `commonmemo.alerts` and passed to
`AlertPolicy.on_alert`. `AlertPolicy(auto_freeze=True)` also freezes the
source of a velocity spike; it is off by default.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Optional

from .events import (
    MemoryEvent,
    Negative,
    NegativeRevoked,
    ProcedureEntry,
    ProcedureRejected,
    ProcedureRetired,
    Reverted,
    UserFact,
    WorldFact,
    WorldFactRejected,
    WorldFactRetracted,
)
from .history import Point, as_of
from .knowledge import Knowledge

if TYPE_CHECKING:
    from .memory import Memory

log = logging.getLogger("commonmemo.alerts")

REVERTIBLE = ("world_fact", "negative", "procedure")
"""Kinds a revert restores. Tool memory and feedback are observations, and
Recaps and the UserProfile live in the engine (ADR 0004)."""


# -- revert --------------------------------------------------------------------


class _Without:
    """The Log with some events left out: what memory would be without them."""

    def __init__(self, memory: "Memory", skip: set[str]) -> None:
        self._store, self._skip = memory.store, skip

    def events_after(self, seq: int, *, limit: Optional[int] = None) -> list[tuple[int, MemoryEvent]]:
        return [(s, e) for s, e in self._store.events_after(seq) if e.id not in self._skip]


def _replay(source: object) -> Knowledge:
    k = Knowledge(":memory:", windows=False)
    k.catch_up(source)  # type: ignore[arg-type]
    return k


def _world_state(k: Knowledge, now: float) -> dict[tuple[str, str], tuple[str, Optional[float]]]:
    return {(w.scope, w.key): (w.value, w.valid_until)
            for s in k.scopes_in_use() for w in k.world_facts([s], now=now, include_stale=True)}


def _negative_state(k: Knowledge) -> dict[tuple[str, str], str]:
    """`(scope, statement) -> event id` of the active negatives."""
    return {(n.scope, n.statement): n.event_id for n in k.all_negatives()}


def _procedure_state(k: Knowledge) -> dict[tuple[str, str, str, str], tuple[str, str, tuple[str, ...]]]:
    """`(scope, task, kind, name) -> (body, use_when, steps)` of the current entries."""
    return {(p.scope, p.task, p.kind, p.name): (p.body, p.use_when, p.steps) for p in k.all_procedures()}


@dataclass
class RevertResult:
    revert_id: str
    """The `Reverted` event's id; its changes carry `run_id="revert:<id>"`."""
    events: list[MemoryEvent] = field(default_factory=list)
    """The compensating events, appended unless this was a dry run."""
    dry_run: bool = False

    @property
    def run_id(self) -> str:
        return f"revert:{self.revert_id}"

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out


def plan_revert(memory: "Memory", *, to: Point = None, run_id: Optional[str] = None,
                source: Optional[str] = None, session_id: Optional[str] = None,
                kinds: Iterable[str] = REVERTIBLE, scope: Optional[str] = None) -> tuple[str, list[MemoryEvent]]:
    """`(description, compensating events)` for a revert; nothing is written."""
    by_cause = run_id is not None or source is not None or session_id is not None
    if by_cause == (to is not None):
        raise ValueError("commonmemo: revert either to= a point, or by cause (run_id=, source=, session_id=)")
    kinds = tuple(kinds)
    unknown = set(kinds) - set(REVERTIBLE)
    if unknown:
        raise ValueError(f"commonmemo: can't revert {sorted(unknown)}; revertible kinds are {REVERTIBLE}")

    now = time.time()
    current = _replay(memory.store)
    if by_cause:
        # Every event matching all the given causes is left out of the replay.
        candidates = memory.store.events(run_id=run_id, session_id=session_id)
        skip = {e.id for e in candidates if source is None or getattr(e, "source", None) == source}
        target = _replay(_Without(memory, skip))
        what = " ".join(f"{k}={v}" for k, v in (("run", run_id), ("source", source), ("session", session_id)) if v)
    else:
        view = as_of(memory, to)
        target, what = view.knowledge, f"to {to!r} (log position {view.seq})"

    events: list[MemoryEvent] = []
    in_scope = (lambda s: s == scope) if scope is not None else (lambda s: True)

    if "world_fact" in kinds:
        before, after = _world_state(target, now), _world_state(current, now)
        for (s, key), (value, until) in sorted(before.items()):
            if in_scope(s) and after.get((s, key)) != (value, until):
                # Changed or removed since: put the old value back.
                events.append(WorldFact(key=key, value=value, scope=s, valid_until=until,
                                        source="revert", proposed=False))
        for (s, key) in sorted(set(after) - set(before)):
            if in_scope(s):
                events.append(WorldFactRetracted(key=key, scope=s, source="revert", reason=f"reverted {what}"))
        # Proposals that only exist because of what is being reverted are turned down.
        target_pending = {w.event_id for w in target.pending_world_facts()}
        for w in current.pending_world_facts():
            if w.event_id not in target_pending and in_scope(w.scope):
                events.append(WorldFactRejected(fact_id=w.event_id, reason=f"reverted {what}"))

    if "negative" in kinds:
        before_n, after_n = _negative_state(target), _negative_state(current)
        for (s, statement), _ in sorted(before_n.items()):
            if in_scope(s) and (s, statement) not in after_n:
                user_id = s.split(":", 1)[1] if s.startswith("user:") else None
                events.append(Negative(statement=statement, scope=s, source="revert", user_id=user_id))
        for (s, statement), event_id in sorted(after_n.items()):
            if in_scope(s) and (s, statement) not in before_n:
                events.append(NegativeRevoked(negative_id=event_id))

    if "procedure" in kinds:
        before_p, after_p = _procedure_state(target), _procedure_state(current)
        for (s, task, kind, name), (body, use_when, steps) in sorted(before_p.items()):
            if in_scope(s) and after_p.get((s, task, kind, name)) != (body, use_when, steps):
                user_id = s.split(":", 1)[1] if s.startswith("user:") else None
                events.append(ProcedureEntry(task=task, entry_kind=kind, name=name, body=body, use_when=use_when,
                                             steps=steps, scope=s, source="revert", proposed=False, user_id=user_id))
        for (s, task, kind, name) in sorted(set(after_p) - set(before_p)):
            if in_scope(s):
                events.append(ProcedureRetired(task=task, entry_kind=kind, name=name, scope=s, source="revert",
                                               reason=f"reverted {what}"))
        target_pending = {p.event_id for p in target.pending_procedures()}
        for p in current.pending_procedures():
            if p.event_id not in target_pending and in_scope(p.scope):
                events.append(ProcedureRejected(entry_id=p.event_id, reason=f"reverted {what}"))

    current.close()
    target.close()
    return what, events


def revert(memory: "Memory", *, to: Point = None, run_id: Optional[str] = None, source: Optional[str] = None,
           session_id: Optional[str] = None, kinds: Iterable[str] = REVERTIBLE, scope: Optional[str] = None,
           reason: Optional[str] = None, dry_run: bool = False) -> RevertResult:
    kinds = tuple(kinds)
    what, planned = plan_revert(memory, to=to, run_id=run_id, source=source, session_id=session_id,
                                kinds=kinds, scope=scope)
    revert_id = uuid.uuid4().hex
    tag = f"revert:{revert_id}"
    events = [e.bind(run_id=tag) for e in planned]
    if dry_run:
        return RevertResult(revert_id, events, dry_run=True)
    for e in events:
        memory._append_outside_session(e)
    memory._append_outside_session(Reverted(id=revert_id, target=what, kinds=kinds, scope=scope, reason=reason,
                                            changes=len(events), run_id=tag))
    return RevertResult(revert_id, events)


# -- budgets -------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """At most `writes` direct world-fact writes from one source per `per`
    seconds; writes past it become proposals."""

    writes: int
    per: float = 3600.0


def over_budget(memory: "Memory", source: str) -> bool:
    budget = memory.budgets.get(source)
    if budget is None:
        return False
    since = time.time() - budget.per
    used = sum(r.n for r in memory.store.event_counts(since=since, bucket=budget.per * 2)
               if r.kind == WorldFact.kind and r.source == source)
    return used >= budget.writes


# -- alerts --------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    kind: str  # "velocity" | "churn" | "review_backlog"
    subject: str
    """The source, key or queue the alert is about."""
    message: str
    value: float
    threshold: float
    at: float = field(default_factory=time.time)


@dataclass
class AlertPolicy:
    """When memory warns (on by default). Thresholds are deliberately loose;
    tune them per deployment."""

    enabled: bool = True
    window: float = 3600.0
    """Velocity is measured over this many seconds..."""
    baseline: float = 14 * 86400.0
    """...against the average rate over this many seconds before it."""
    velocity_ratio: float = 10.0
    velocity_min: int = 50
    """A spike needs at least this many writes in the window, and this ratio."""
    flip_flops: int = 3
    churn_window: float = 7 * 86400.0
    backlog: int = 100
    """Alert when more proposals than this wait for review."""
    check_every: float = 60.0
    """After a write, check at most this often."""
    on_alert: Optional[Callable[[Alert], None]] = None
    auto_freeze: bool = False
    """Freeze the source of a velocity spike (opt-in)."""


_WATCHED = (WorldFact.kind, UserFact.kind, Negative.kind)


def check_alerts(memory: "Memory", *, now: Optional[float] = None) -> list[Alert]:
    p, now = memory.alert_policy, time.time() if now is None else now
    out: list[Alert] = []

    def by_source(since: float, until: float) -> dict[str, int]:
        totals: dict[str, int] = {}
        for r in memory.store.event_counts(since=since, until=until, bucket=10**12):
            if r.kind in _WATCHED:
                totals[r.source or "unknown"] = totals.get(r.source or "unknown", 0) + r.n
        return totals

    recent = by_source(now - p.window, now)
    earlier = by_source(now - p.window - p.baseline, now - p.window)
    windows = max(1.0, p.baseline / p.window)
    for src, n in sorted(recent.items()):
        rate = earlier.get(src, 0) / windows
        ratio = n / max(rate, 1.0)
        if n >= p.velocity_min and ratio >= p.velocity_ratio:
            out.append(Alert("velocity", src, f"{n} writes from {src!r} in the last {p.window / 60:.0f} min, "
                                              f"{ratio:.0f}x the usual rate", ratio, p.velocity_ratio, now))

    for c in memory.history().churn(since=now - p.churn_window, min_changes=1):
        if c.flip_flops >= p.flip_flops:
            out.append(Alert("churn", f"{c.kind}:{c.key}", f"{c.key} flip-flopped {c.flip_flops} times",
                             c.flip_flops, p.flip_flops, now))

    waiting = len(memory.knowledge.pending_world_facts())
    if waiting > p.backlog:
        out.append(Alert("review_backlog", "world_facts", f"{waiting} world facts waiting for review",
                         waiting, p.backlog, now))
    return out


def after_write(memory: "Memory", event: MemoryEvent) -> None:
    """Throttled alert check after a watched write; logs and notifies."""
    p = memory.alert_policy
    if not p.enabled or event.kind not in _WATCHED:
        return
    now = time.time()
    if now - memory._last_alert_check < p.check_every:
        return
    memory._last_alert_check = now
    for alert in check_alerts(memory, now=now):
        log.warning("commonmemo alert (%s): %s", alert.kind, alert.message)
        if p.on_alert is not None:
            try:
                p.on_alert(alert)
            except Exception:  # noqa: BLE001 -- an alert hook must never fail a write
                log.exception("commonmemo: on_alert raised")
        already = any(f.scope is None and f.source == alert.subject for f in memory.knowledge.freezes())
        if p.auto_freeze and alert.kind == "velocity" and not already:
            memory.freeze(source=alert.subject, reason=f"auto: {alert.message}")


__all__ = [
    "Alert",
    "AlertPolicy",
    "Budget",
    "REVERTIBLE",
    "RevertResult",
    "check_alerts",
    "plan_revert",
    "revert",
]
