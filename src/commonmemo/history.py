"""Memory history: how memory is changing, and what it looked like before.

Everything here is read from the Log, which is append-only and ordered
(ADR 0004). Nothing in this module writes memory.

    h = mem.history()
    h.velocity(bucket=3600)      # events per hour, by kind, source and scope
    h.source_mix()               # share of changes by source
    h.churn()                    # keys whose value keeps changing (flip-flops)
    h.review_flow()              # proposals, approvals, time to review
    h.staleness(); h.growth()
    h.by_run("deploy-42")        # what one run wrote

    mem.snapshot("before v3")    # a named log position
    then = mem.as_of("before v3")       # or a log position, or a time
    mem.diff("before v3")               # what changed since then

`as_of` replays the Log into a scratch projection: the live projections are
never touched. It covers the kinds CommonMemo owns (world facts, negatives,
feedback, tool memory, checkpoints and steps); Recaps and the UserProfile
held by an engine are not replayed. Because it replays today's Log, data a
user asked to forget is gone from every past view too: `forget_user` wins.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional, Union

from .events import (
    MemoryEvent,
    Snapshot,
    UserFact,
    WorldFact,
    WorldFactConfirmed,
    WorldFactRejected,
)
from .knowledge import FeedbackSummary, Knowledge, NegativeRecord, StepRecord, WorldRecord
from .stores.base import EventCount, Store
from .tool_memory import ToolProfile

if TYPE_CHECKING:
    from .memory import Memory

DAY = 86400.0

Point = Union[int, float, str, None]
"""A point in the log: an `int` log position, a `float` epoch time (the last
event logged at or before it), a `str` snapshot label, or `None` for now."""


# -- point-in-time views -------------------------------------------------------


class _Upto:
    """A read-only store that ends at log position `seq`."""

    def __init__(self, store: Store, seq: int) -> None:
        self._store, self._seq = store, seq

    def events_after(self, seq: int, *, limit: Optional[int] = None) -> list[tuple[int, MemoryEvent]]:
        return [(s, e) for s, e in self._store.events_after(seq) if s <= self._seq]


def resolve(memory: "Memory", point: Point) -> tuple[int, float]:
    """`point` as `(log position, time)`."""
    store = memory.store
    if point is None:
        return store.position(), time.time()
    if isinstance(point, bool):
        raise TypeError("commonmemo: a point is a log position, a time or a snapshot label")
    if isinstance(point, int):
        return point, time.time()
    if isinstance(point, float):
        return store.position(at=point), point
    snaps = [e for e in store.events(kinds=[Snapshot.kind]) if e.label == point]
    if not snaps:
        raise KeyError(f"commonmemo: no snapshot labelled {point!r}")
    snap = snaps[-1]
    return store.position(event_id=snap.id), snap.ts


@dataclass
class MemoryView:
    """Memory as of one point in the log (`Memory.as_of`). Read-only."""

    seq: int
    """The last log position included."""
    at: float
    """The time the view is as of; expiry (`valid_until`) is judged against it."""
    knowledge: Knowledge
    project_scope: str

    def _scopes(self, scopes: Optional[list[str]]) -> list[str]:
        return scopes or (["global"] + ([self.project_scope] if self.project_scope != "global" else []))

    def world_facts(self, scopes: Optional[list[str]] = None, *, include_stale: bool = False) -> list[WorldRecord]:
        return self.knowledge.world_facts(self._scopes(scopes), now=self.at, include_stale=include_stale)

    def pending_world_facts(self, scopes: Optional[list[str]] = None) -> list[WorldRecord]:
        return self.knowledge.pending_world_facts(scopes)

    def negatives(self, scopes: list[str]) -> list[NegativeRecord]:
        return self.knowledge.negatives(scopes)

    def tool_profile(self, tool: str, agent: Optional[str] = None) -> ToolProfile:
        return self.knowledge.tool_profile(tool, agent)

    def feedback_summary(self, *, user_id: Optional[str] = None, agent: Optional[str] = None) -> FeedbackSummary:
        return self.knowledge.feedback_summary(user_id=user_id, agent=agent)

    def steps(self, session_id: str) -> list[StepRecord]:
        return self.knowledge.steps(session_id)

    def close(self) -> None:
        self.knowledge.close()


def as_of(memory: "Memory", point: Point) -> MemoryView:
    seq, at = resolve(memory, point)
    scratch = Knowledge(":memory:")
    scratch.catch_up(_Upto(memory.store, seq))  # type: ignore[arg-type]
    return MemoryView(seq=seq, at=at, knowledge=scratch, project_scope=memory.project_scope)


@dataclass(frozen=True)
class Change:
    """One difference between two views. `before` / `after` is None when the
    entry did not exist on that side."""

    kind: str  # "world_fact" | "pending_world_fact" | "negative" | "tool" | "feedback"
    key: str
    before: Optional[str]
    after: Optional[str]

    @property
    def change(self) -> str:
        return "added" if self.before is None else "removed" if self.after is None else "changed"


@dataclass
class Diff:
    """What changed between two points (`Memory.diff`)."""

    since: int
    until: int
    changes: list[Change] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.changes)

    def summary(self) -> dict[str, dict[str, int]]:
        """`{kind: {"added": n, "changed": n, "removed": n}}`."""
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"added": 0, "changed": 0, "removed": 0})
        for c in self.changes:
            out[c.kind][c.change] += 1
        return dict(out)


def _snapshot_of(view: MemoryView) -> dict[tuple[str, str], str]:
    k = view.knowledge
    out: dict[tuple[str, str], str] = {}
    for scope in k.scopes_in_use():
        for w in k.world_facts([scope], now=view.at, include_stale=True):
            out[("world_fact", f"{w.scope}/{w.key}")] = w.value
        for n in k.negatives([scope]):
            # The same rule re-added (by a revert, say) is the same negative.
            out[("negative", f"{n.scope}/{n.statement}")] = n.reason or ""
    for w in k.pending_world_facts():
        out[("pending_world_fact", f"{w.scope}/{w.key}#{w.event_id}")] = f"{w.value} ({w.status})"
    for tool in k.tools():
        p = k.tool_profile(tool)
        out[("tool", tool)] = f"calls={p.calls} failures={p.failures} signatures={len(p.errors)}"
    fb = k.feedback_summary()
    if fb.count:
        out[("feedback", "all")] = f"count={fb.count} corrections={fb.corrections}"
    return out


def diff(memory: "Memory", since: Point, until: Point = None) -> Diff:
    a, b = as_of(memory, since), as_of(memory, until)
    try:
        before, after = _snapshot_of(a), _snapshot_of(b)
    finally:
        a.close()
        b.close()
    changes = [
        Change(kind, key, before.get((kind, key)), after.get((kind, key)))
        for kind, key in sorted(set(before) | set(after))
        if before.get((kind, key)) != after.get((kind, key))
    ]
    return Diff(since=a.seq, until=b.seq, changes=changes)


# -- metrics -------------------------------------------------------------------


@dataclass(frozen=True)
class Churn:
    """A key whose value kept changing in the window."""

    kind: str  # "world_fact" | "user_fact"
    key: str  # "<scope>/<key>" or "<user_id>/<key>"
    changes: int
    """Value changes (the first value is not a change)."""
    flip_flops: int
    """Changes back to a value the key already had (A -> B -> A counts one)."""
    values: tuple[str, ...]


@dataclass(frozen=True)
class ReviewFlow:
    proposed: int
    confirmed: int
    rejected: int
    pending_now: int
    disputed_now: int
    median_review_seconds: Optional[float]

    @property
    def rejection_rate(self) -> Optional[float]:
        settled = self.confirmed + self.rejected
        return self.rejected / settled if settled else None


@dataclass(frozen=True)
class Staleness:
    world_facts: int
    expired: int
    """Past `valid_until`: no longer pushed as current."""
    ages: dict[str, int]
    """Current facts by age: `<1d`, `1-7d`, `7-30d`, `30-90d`, `>90d`."""


@dataclass(frozen=True)
class Growth:
    world_facts: int
    pending_world_facts: int
    profile_keys: dict[str, int]
    """Explicit UserProfile keys per user, most first (top 20)."""
    error_signatures: dict[str, int]
    """Distinct error signatures per tool."""


_AGE_BUCKETS = (("<1d", 1), ("1-7d", 7), ("7-30d", 30), ("30-90d", 90))


class History:
    """Read-only metrics over the Log (`Memory.history()`)."""

    def __init__(self, memory: "Memory") -> None:
        self.memory = memory

    def velocity(self, *, since: Optional[float] = None, until: Optional[float] = None, bucket: float = DAY,
                 kinds: Optional[Iterable[str]] = None) -> list[EventCount]:
        """Events per `bucket` seconds, by kind, source and scope, oldest
        first. Pruned stretches are counted per day, from the rollups."""
        rows = self.memory.store.event_counts(since=since, until=until, bucket=bucket)
        wanted = set(kinds) if kinds is not None else None
        return [r for r in rows if wanted is None or r.kind in wanted]

    def source_mix(self, *, since: Optional[float] = None, until: Optional[float] = None,
                   kinds: Iterable[str] = (WorldFact.kind, UserFact.kind, "negative")) -> dict[str, float]:
        """Share of writes per source, for the kinds that record one."""
        totals: dict[str, int] = defaultdict(int)
        for r in self.velocity(since=since, until=until, bucket=10**12, kinds=kinds):
            totals[r.source or "unknown"] += r.n
        n = sum(totals.values())
        return {s: c / n for s, c in sorted(totals.items(), key=lambda kv: -kv[1])} if n else {}

    def churn(self, *, since: Optional[float] = None, until: Optional[float] = None,
              min_changes: int = 2) -> list[Churn]:
        """Keys that changed at least `min_changes` times, most first.
        Proposals are left out: they change nothing until approved."""
        store = self.memory.store
        series: dict[tuple[str, str], list[str]] = defaultdict(list)
        kinds = [WorldFact.kind, WorldFactConfirmed.kind, UserFact.kind]
        for e in store.events(kinds=kinds, since=since):
            if until is not None and e.ts >= until:
                continue
            if isinstance(e, WorldFact) and e.proposed:
                continue
            if isinstance(e, (WorldFact, WorldFactConfirmed)):
                series[("world_fact", f"{e.scope}/{e.key}")].append(e.value)
            elif isinstance(e, UserFact) and e.user_id is not None:
                series[("user_fact", f"{e.user_id}/{e.key}")].append(e.value)
        out = []
        for (kind, key), values in series.items():
            changes, flips, seen = 0, 0, [values[0]]
            for prev, cur in zip(values, values[1:]):
                if cur != prev:
                    changes += 1
                    flips += cur in seen
                    seen.append(cur)
            if changes >= min_changes:
                out.append(Churn(kind, key, changes, flips, tuple(values)))
        return sorted(out, key=lambda c: (-c.changes, -c.flip_flops, c.key))

    def review_flow(self, *, since: Optional[float] = None, until: Optional[float] = None) -> ReviewFlow:
        """World-fact proposals and how they were settled (ADR 0002)."""
        store, k = self.memory.store, self.memory.knowledge
        events = [e for e in store.events(kinds=[WorldFact.kind, WorldFactConfirmed.kind, WorldFactRejected.kind])
                  if until is None or e.ts < until]
        proposed_at = {e.id: e.ts for e in events if isinstance(e, WorldFact) and e.proposed}
        in_window = [e for e in events if since is None or e.ts >= since]
        settled = [e for e in in_window if isinstance(e, (WorldFactConfirmed, WorldFactRejected))]
        waits = [e.ts - proposed_at[e.fact_id] for e in settled if e.fact_id in proposed_at]
        pending = k.pending_world_facts()
        return ReviewFlow(
            proposed=sum(1 for e in in_window if isinstance(e, WorldFact) and e.proposed),
            confirmed=sum(1 for e in settled if isinstance(e, WorldFactConfirmed)),
            rejected=sum(1 for e in settled if isinstance(e, WorldFactRejected)),
            pending_now=sum(1 for w in pending if w.status == "proposed"),
            disputed_now=sum(1 for w in pending if w.status == "disputed"),
            median_review_seconds=statistics.median(waits) if waits else None,
        )

    def staleness(self, *, now: Optional[float] = None) -> Staleness:
        """Age of the current world facts, and how many have expired."""
        now = time.time() if now is None else now
        k = self.memory.knowledge
        facts = [w for s in k.scopes_in_use() for w in k.world_facts([s], now=now, include_stale=True)]
        ages = {label: 0 for label, _ in _AGE_BUCKETS} | {">90d": 0}
        for w in facts:
            days = (now - w.valid_from) / DAY
            label = next((label for label, limit in _AGE_BUCKETS if days < limit), ">90d")
            ages[label] += 1
        expired = sum(1 for w in facts if w.valid_until is not None and w.valid_until <= now)
        return Staleness(world_facts=len(facts), expired=expired, ages=ages)

    def growth(self) -> Growth:
        """How big memory is getting, where it is cheap to know exactly."""
        store, k = self.memory.store, self.memory.knowledge
        keys: dict[str, set[str]] = defaultdict(set)
        for e in store.events(kinds=[UserFact.kind]):
            if e.user_id is not None:
                keys[e.user_id].add(e.key)
        top = sorted(((u, len(ks)) for u, ks in keys.items()), key=lambda kv: (-kv[1], kv[0]))[:20]
        return Growth(
            world_facts=sum(len(k.world_facts([s], include_stale=True)) for s in k.scopes_in_use()),
            pending_world_facts=len(k.pending_world_facts()),
            profile_keys=dict(top),
            error_signatures={t: len(k.tool_profile(t).errors) for t in k.tools()},
        )

    def by_run(self, run_id: str) -> dict[str, int]:
        """What one run wrote: event counts by kind."""
        counts: dict[str, int] = defaultdict(int)
        for e in self.memory.store.events(run_id=run_id):
            counts[e.kind] += 1
        return dict(counts)


__all__ = [
    "Change",
    "Churn",
    "Diff",
    "Growth",
    "History",
    "MemoryView",
    "Point",
    "ReviewFlow",
    "Staleness",
]
