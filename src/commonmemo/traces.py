"""Turn traces: what memory pushed into an agent's context, and why (#25).

Every push (`Session.push`, `push_turn`, `refresh`) appends a `ContextPushed`
event to the Log. It records references, not text: which facts, world facts,
negatives, lessons and recaps were pushed, with each one's source and the
events it came from, plus what was left out (`more`) and how big the push
was. The pushed text itself is kept only with `MemoryPolicy.trace_text`.

    mem.traces(session_id="s1")            # what each push gave the agent
    mem.why("refund_window", session_id="s1")
    # -> the trace item, and the events it came from (who said it, when)

Traces live in the Log with the user's id on them, so `forget_user` and
`prune` remove them like any other event. The exporters (`otel.py`), the CLI
(`commonmemo trace`) and the inspector (`commonmemo inspect`) all read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .context import MemoryContext
from .events import ContextPushed, MemoryEvent

if TYPE_CHECKING:
    from .memory import Memory


def trace_items(ctx: MemoryContext) -> tuple[dict[str, Any], ...]:
    """One reference per item `ctx` pushes: kind, key or id, and provenance."""
    items: list[dict[str, Any]] = []
    items += [{"kind": "negative", "id": n.event_id, "scope": n.scope, "source": n.source, "events": [n.event_id]}
              for n in ctx.negatives]
    items += [{"kind": "fact", "key": f.key, "source": "profile", "events": list(f.source_event_ids)}
              for f in ctx.facts]
    items += [{"kind": "world", "key": w.key, "scope": w.scope, "source": w.source, "status": w.status,
               "events": [w.event_id]} for w in ctx.world]
    items += [{"kind": "world", "key": w.key, "scope": w.scope, "source": w.source, "status": w.status,
               "half": "turn", "events": [w.event_id]} for w in ctx.world_turn]
    items += [{"kind": "lesson", "id": f.event_id, "source": f.kind, "events": [f.event_id]} for f in ctx.lessons]
    items += [{"kind": "recap", "id": r.id, "session": r.session_id, "agent": r.agent, "score": r.score,
               "events": list(r.source_event_ids)} for r in ctx.recaps]
    return tuple(items)


@dataclass(frozen=True)
class TurnTrace:
    """One push, as `Memory.traces` returns it."""

    event_id: str
    ts: float
    session_id: Optional[str]
    user_id: Optional[str]
    agent: Optional[str]
    reason: str
    items: tuple[dict[str, Any], ...]
    more: dict[str, int]
    chars: int
    text: Optional[str] = None

    @classmethod
    def from_event(cls, e: ContextPushed) -> "TurnTrace":
        items = tuple(e.items) or tuple({"kind": "recap", "id": rid, "events": []} for rid in e.recap_ids)
        return cls(e.id, e.ts, e.session_id, e.user_id, e.agent, e.reason, items, dict(e.more or {}),
                   e.chars, e.text)

    def find(self, key: str) -> list[dict[str, Any]]:
        """The items pushed under `key` (a fact or world fact key, or an id)."""
        return [i for i in self.items if key in (i.get("key"), i.get("id"))]

    def lines(self, labels: Optional[dict[str, str]] = None) -> list[str]:
        """The trace in plain text, one line per item, for terminals and logs.
        `labels` maps ids to readable text (see `views.session_view`)."""
        labels = labels or {}
        out = [f"{self.reason} push, {self.chars} chars" + (f", by {self.agent}" if self.agent else "")]
        for i in self.items:
            what = i.get("key") or labels.get(i.get("id") or "") or i.get("id")
            extra = ", ".join(f"{k} {i[k]}" for k in ("scope", "source", "status", "score", "session")
                              if i.get(k) is not None)
            out.append(f"  {i['kind']}: {what}" + (f" ({extra})" if extra else ""))
        cut = ", ".join(f"{n} {k}" for k, n in self.more.items() if n > 0)
        if cut:
            out.append(f"  left out: {cut}")
        return out


@dataclass(frozen=True)
class Why:
    """Why the agent got an item: the trace that pushed it, the item, and the
    events it came from (`Memory.why`)."""

    trace: TurnTrace
    item: dict[str, Any]
    sources: list[MemoryEvent] = field(default_factory=list)

    def lines(self) -> list[str]:
        what = self.item.get("key") or self.item.get("id")
        out = [f"{self.item['kind']} {what}: pushed in a {self.trace.reason} push to session {self.trace.session_id}"]
        for e in self.sources:
            source = getattr(e, "source", None)
            out.append(f"  from {e.kind} {e.id} in session {e.session_id}" + (f", source {source}" if source else ""))
        if not self.sources:
            out.append("  (its source events are no longer in the Log: pruned or forgotten)")
        return out


def traces(memory: "Memory", *, session_id: Optional[str] = None, user_id: Optional[str] = None,
           limit: Optional[int] = None) -> list[TurnTrace]:
    events = memory.store.events(session_id=session_id, user_id=user_id, kinds=[ContextPushed.kind], limit=limit)
    return [TurnTrace.from_event(e) for e in events]


def why(memory: "Memory", key: str, *, session_id: Optional[str] = None,
        user_id: Optional[str] = None) -> Optional[Why]:
    """The latest push of `key` in the session (or for the user), with its sources."""
    for trace in reversed(traces(memory, session_id=session_id, user_id=user_id)):
        found = trace.find(key)
        if found:
            item = found[0]
            return Why(trace, item, memory.store.get_events(item.get("events") or []))
    return None


__all__ = ["TurnTrace", "Why", "trace_items", "traces", "why"]
