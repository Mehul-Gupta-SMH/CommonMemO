"""Turn traces as OpenTelemetry spans (#25), so memory shows up next to the
LLM call in the tools a team already uses (Jaeger, Honeycomb, Grafana, and
Langfuse or LangSmith through their OTLP endpoints).

    pip install "commonmemo[otel]"            # the OpenTelemetry API and SDK; add your exporter

    from commonmemo.otel import instrument
    instrument(mem)                            # every push becomes a span, as it happens

Each push (`ContextPushed`) becomes a `commonmemo.push` span, a child of
whatever span is current: typically the agent's LLM call. Attributes:

| Attribute | Value |
|---|---|
| `commonmemo.reason` | `start`, `turn` or `refresh` |
| `commonmemo.chars` | characters pushed (~4 per token) |
| `commonmemo.items` | number of items pushed |
| `commonmemo.items.<kind>` | items per kind: `fact`, `world`, `negative`, `lesson`, `recap` |
| `commonmemo.facts`, `commonmemo.world_facts` | the keys pushed |
| `commonmemo.recaps` | recap ids pushed |
| `commonmemo.left_out.<kind>` | what the push left out (`MemoryContext.more`) |
| `session.id`, `commonmemo.agent` | where it happened |
| `enduser.id` | only with `include_user=True` |

Each item is also a span event (`commonmemo.item`) with its kind, key or id,
source and score. Pushed text is never exported unless the trace kept it
(`MemoryPolicy.trace_text`) and `include_text=True`. `export(mem, ...)` sends
stored traces after the fact, e.g. from a test run.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Optional

from .events import ContextPushed, MemoryEvent
from .traces import TurnTrace

if TYPE_CHECKING:
    from .memory import Memory

SPAN_NAME = "commonmemo.push"


def _tracer(tracer: Any = None) -> Any:
    if tracer is not None:
        return tracer
    try:
        from opentelemetry import trace
    except ImportError as e:  # pragma: no cover -- exercised only without the extra
        raise ImportError('commonmemo.otel needs OpenTelemetry: pip install "commonmemo[otel]"') from e
    return trace.get_tracer("commonmemo")


def attributes(trace: TurnTrace, *, include_user: bool = False, include_text: bool = False) -> dict[str, Any]:
    """The span attributes for one turn trace."""
    kinds = Counter(i["kind"] for i in trace.items)
    attrs: dict[str, Any] = {
        "commonmemo.reason": trace.reason,
        "commonmemo.chars": trace.chars,
        "commonmemo.items": len(trace.items),
        **{f"commonmemo.items.{k}": n for k, n in sorted(kinds.items())},
        **{f"commonmemo.left_out.{k}": n for k, n in sorted(trace.more.items()) if n > 0},
    }
    for attr, kind, field in (("commonmemo.facts", "fact", "key"), ("commonmemo.world_facts", "world", "key"),
                              ("commonmemo.recaps", "recap", "id")):
        values = [str(i[field]) for i in trace.items if i["kind"] == kind and i.get(field)]
        if values:
            attrs[attr] = values
    if trace.session_id:
        attrs["session.id"] = trace.session_id
    if trace.agent:
        attrs["commonmemo.agent"] = trace.agent
    if include_user and trace.user_id:
        attrs["enduser.id"] = trace.user_id
    if include_text and trace.text:
        attrs["commonmemo.text"] = trace.text
    return attrs


def emit(trace: TurnTrace, *, tracer: Any = None, include_user: bool = False, include_text: bool = False) -> None:
    """Send one turn trace as a span, timed at the push."""
    start = int(trace.ts * 1e9)
    span = _tracer(tracer).start_span(SPAN_NAME, start_time=start,
                                      attributes=attributes(trace, include_user=include_user,
                                                            include_text=include_text))
    for item in trace.items:
        span.add_event("commonmemo.item", {k: v for k, v in {
            "kind": item["kind"], "key": item.get("key"), "id": item.get("id"), "source": item.get("source"),
            "scope": item.get("scope"), "score": item.get("score"),
        }.items() if v is not None}, timestamp=start)
    span.end(end_time=start)


def instrument(memory: "Memory", *, tracer: Any = None, include_user: bool = False,
               include_text: bool = False) -> Callable[[MemoryEvent], None]:
    """Emit a span for every push from now on; returns the listener, which
    `memory.listeners.remove(...)` takes off again."""
    resolved = _tracer(tracer)

    def listener(event: MemoryEvent) -> None:
        if isinstance(event, ContextPushed):
            emit(TurnTrace.from_event(event), tracer=resolved, include_user=include_user, include_text=include_text)

    memory.listeners.append(listener)
    return listener


def export(memory: "Memory", *, session_id: Optional[str] = None, user_id: Optional[str] = None,
           tracer: Any = None, include_user: bool = False, include_text: bool = False) -> int:
    """Send stored turn traces as spans; returns how many."""
    resolved = _tracer(tracer)
    found = memory.traces(session_id=session_id, user_id=user_id)
    for trace in found:
        emit(trace, tracer=resolved, include_user=include_user, include_text=include_text)
    return len(found)


__all__ = ["SPAN_NAME", "attributes", "emit", "export", "instrument"]
