"""Read-only views of what memory holds, for the CLI and the inspector (#25).

Each view is plain data (dicts and lists) built from the Log, the
projections and the engine; `*_lines` render it for terminals and CI logs.
Nothing here writes: in particular no view opens a session, which would
reopen a closed one.

    user_view(mem, "ada")       # profile with provenance, recaps, constraints, tombstones, feedback
    session_view(mem, "s1")     # timeline, where a resume would start, turn traces
    overview(mem)               # recent sessions and users
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from .agent_tools import run_sync
from .context import _line
from .events import BOOKKEEPING, ContextPushed, describe
from .resume import ResumePoint
from .traces import TurnTrace

if TYPE_CHECKING:
    from .memory import Memory


def _when(ts: Optional[float]) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "-"


def event_line(event: Any) -> str:
    """One readable line for any event."""
    text = describe(event)
    if text:
        return text
    if isinstance(event, ContextPushed):
        return f"memory pushed ({event.reason}): {len(event.items) or len(event.recap_ids)} items, {event.chars} chars"
    if isinstance(event, BOOKKEEPING):
        fields = {k: v for k, v in event.payload.items() if v not in (None, "", (), [], {})}
        return f"{event.kind}: {fields}"
    return _line(event)


def user_view(mem: "Memory", user_id: str) -> dict[str, Any]:
    profile = mem._visible_facts(user_id, run_sync(mem.engine.user_profile(user_id)))
    recaps = mem._visible_recaps(user_id, run_sync(mem.engine.search_recaps("", user_id=user_id, limit=50)))
    summary = mem.feedback_summary(user_id=user_id)
    return {
        "user_id": user_id,
        "facts": [{"key": f.key, "value": f.value, "since": _when(f.valid_from), "events": list(f.source_event_ids)}
                  for f in profile],
        "recaps": [{"session": r.session_id, "agent": r.agent, "ended": _when(r.ended_at), "summary": r.summary}
                   for r in sorted(recaps, key=lambda r: r.ended_at, reverse=True)],
        "negatives": [{"statement": n.statement, "reason": n.reason, "source": n.source, "since": _when(n.ts)}
                      for n in mem.knowledge.negatives([f"user:{user_id}"])],
        "forgotten": [{"key": k, "pattern": p, "when": _when(ts)} for k, p, ts in mem.knowledge.tombstones(user_id)],
        "feedback": {"count": summary.count, "recent": [r.line() for r in summary.recent]},
        "sessions": mem.store.session_ids(user_id),
    }


def session_view(mem: "Memory", session_id: str) -> Optional[dict[str, Any]]:
    record = mem.store.get_session(session_id)
    if record is None:
        return None
    events = mem.store.events(session_id=session_id)
    k = mem.knowledge
    cp = k.latest_checkpoint(session_id)
    steps = k.steps(session_id)
    resume = ResumePoint(
        session_id=session_id, checkpoint=cp,
        completed=[s for s in steps if s.status == "finished"],
        in_doubt=[s for s in steps if s.status == "started"],
        failed=[s for s in steps if s.status == "failed"],
        since_checkpoint=[e for e in events if cp and e.ts >= cp.ts and e.id != cp.event_id],
    )
    return {
        "session_id": session_id,
        "user_id": record.user_id,
        "agent": record.agent,
        "opened": _when(record.opened_at),
        "closed": _when(record.closed_at) if record.closed_at else None,
        "state": dict(record.state),
        "timeline": [{"when": _when(e.ts), "kind": e.kind, "agent": e.agent, "line": event_line(e), "id": e.id}
                     for e in events],
        "resume": [] if resume.is_fresh else resume.lines(),
        "traces": (traces := [TurnTrace.from_event(e) for e in events if isinstance(e, ContextPushed)]),
        "labels": _labels(mem, traces),
    }


def _labels(mem: "Memory", traces: list[TurnTrace]) -> dict[str, str]:
    """Readable text for items a trace names only by id (negatives, lessons),
    looked up now from their source events. Traces themselves keep no text."""
    ids = {i["id"] for t in traces for i in t.items if i["kind"] in ("negative", "lesson") and i.get("id")}
    return {e.id: describe(e) or e.kind for e in mem.store.get_events(ids)}


def overview(mem: "Memory", *, limit: int = 2000) -> dict[str, Any]:
    """Recent sessions (newest first) and the users they belong to, from the
    latest `limit` events."""
    sessions: dict[str, dict[str, Any]] = {}
    for e in reversed(mem.store.events(limit=limit)):
        if e.session_id and e.session_id not in sessions:
            sessions[e.session_id] = {"session_id": e.session_id, "user_id": e.user_id, "agent": e.agent,
                                      "last": _when(e.ts), "events": 0}
        if e.session_id:
            sessions[e.session_id]["events"] += 1
    users = sorted({s["user_id"] for s in sessions.values() if s["user_id"]})
    return {"sessions": list(sessions.values()), "users": users,
            "world": [{"key": w.key, "value": w.value, "scope": w.scope, "source": w.source}
                      for w in mem.world_facts()],
            "pending": [{"key": w.key, "value": w.value, "scope": w.scope, "user_id": w.user_id}
                        for w in mem.pending_world_facts()]}


# -- text ----------------------------------------------------------------------


def user_lines(view: dict[str, Any]) -> list[str]:
    out = [f"user {view['user_id']}"]
    out += ["", "profile:"] + [f"  {f['key']}: {f['value']}  (since {f['since']})" for f in view["facts"]] \
        if view["facts"] else ["", "profile: (empty)"]
    if view["negatives"]:
        out += ["", "never:"] + [f"  {n['statement']}" + (f" ({n['reason']})" if n["reason"] else "")
                                 for n in view["negatives"]]
    if view["forgotten"]:
        out += ["", "forgotten:"] + [f"  {f['key']} ({f['when']})" for f in view["forgotten"]]
    if view["recaps"]:
        out += ["", "past sessions:"] + [f"  {r['ended']} [{r['agent'] or '-'}] {r['summary']}" for r in view["recaps"]]
    out += ["", f"feedback: {view['feedback']['count']}"] + [f"  {line}" for line in view["feedback"]["recent"]]
    out += ["", f"sessions: {', '.join(view['sessions']) or '-'}"]
    return out


def session_lines(view: dict[str, Any], *, traces_only: bool = False) -> list[str]:
    out = [f"session {view['session_id']}  user {view['user_id'] or '-'}  agent {view['agent'] or '-'}  "
           f"opened {view['opened']}" + (f"  closed {view['closed']}" if view["closed"] else "  (open)")]
    if not traces_only:
        out += ["", "timeline:"] + [f"  {t['when']} {t['line']}" for t in view["timeline"]]
        if view["resume"]:
            out += ["", "a resume would start from:"] + [f"  {line}" for line in view["resume"]]
    out += ["", f"turn traces: {len(view['traces'])}"]
    for trace in view["traces"]:
        lines = trace.lines(view.get("labels"))
        out += [f"  {_when(trace.ts)} " + lines[0]] + [f"  {line}" for line in lines[1:]]
    return out


__all__ = ["event_line", "overview", "session_lines", "session_view", "user_lines", "user_view"]
