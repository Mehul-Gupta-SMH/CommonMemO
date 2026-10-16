"""CommonADK integration: memory for all six CommonADK targets at once.

CommonADK's runners already normalize every SDK's execution into one event
model (`commonadk.runners.events`) and fire each event through an
observe-only `HookRegistry`. One catch-all hook is therefore enough to give
Google ADK, OpenAI Agents, Claude Agent SDK, CrewAI, AutoGen and LangGraph
runs the same event memory::

    from commonmemo import Memory
    from commonmemo.integrations.commonadk import run_with_memory

    mem = Memory("sqlite:///mem.db")
    async with mem.session(user_id="u1", agent="coordinator") as s:
        trace = await run_with_memory(s, runner, project, "coordinator", prompt)

How each CommonADK event lands in the event tier:

- every event is kept verbatim as an `ExternalEvent(source="commonadk.<type>")`
  -- token counts, cost, transfers and errors included, nothing dropped;
- `run_started.prompt` additionally becomes a `Message(role="user")` and
  `run_finished.final_text` a `Message(role="assistant")`;
- `tool_call` becomes a `ToolInvoked` (instead of an `ExternalEvent`), so a
  CommonADK tool call and an `@mem.track`-ed call read the same downstream.

Event ids are derived from CommonADK's `run_id` and `seq`, so a hook that
fires twice for the same event never records it twice.

Gap, until CommonADK grows a pre-run context step: its hooks are
observe-only, so memory cannot be added to an agent's *instructions* yet.
`run_with_memory` puts it in front of the prompt instead
(`prompt_with_context`). See docs/design.md, "CommonADK integration".

CommonADK is imported lazily, only when a function here is called --
`import commonmemo` never needs it installed.
"""

from __future__ import annotations

from typing import Any, Optional

from ..context import MemoryContext
from ..events import ExternalEvent, MemoryEvent, Message, ToolInvoked
from ..session import Session

_INSTALL_HINT = 'commonmemo: the CommonADK integration needs commonadk -- pip install "commonmemo[commonadk]"'


def _hook_registry_cls() -> type:
    try:
        from commonadk.runners.hooks import HookRegistry
    except ImportError as exc:  # pragma: no cover -- exercised only without commonadk
        raise ImportError(_INSTALL_HINT) from exc
    return HookRegistry


def translate(event: Any) -> list[MemoryEvent]:
    """CommonADK event -> CommonMemo events (see the module docstring)."""
    data = event.to_dict()
    kind = data.pop("type", getattr(event, "kind", "event"))
    run_id, seq, ts = data.get("run_id", ""), data.get("seq"), data.get("ts")
    base_id = f"commonadk_{run_id}_{seq}"
    common: dict[str, Any] = {"ts": ts} if ts is not None else {}

    if kind == "tool_call":
        return [
            ToolInvoked(
                id=base_id,
                agent=data.get("agent_name"),
                name=data.get("tool_name", ""),
                arguments=data.get("arguments") or {},
                result=data.get("result_summary"),
                error=data.get("error"),
                duration_ms=data.get("duration_ms"),
                **common,
            )
        ]

    out: list[MemoryEvent] = [
        ExternalEvent(
            id=base_id,
            agent=data.get("agent_name"),
            source=f"commonadk.{kind}",
            data=data,
            **common,
        )
    ]
    if kind == "run_started" and data.get("prompt"):
        out.append(Message(id=f"{base_id}_msg", role="user", content=data["prompt"], **common))
    elif kind == "run_finished" and data.get("final_text"):
        out.append(Message(id=f"{base_id}_msg", role="assistant", content=data["final_text"], **common))
    return out


def attach(session: Session, hooks: Optional[Any] = None) -> Any:
    """Register a catch-all hook that records every CommonADK event into
    `session`. Pass an existing `HookRegistry` to add to it (your own hooks
    keep working), or none to get a fresh one. Returns the registry."""
    if hooks is None:
        hooks = _hook_registry_cls()()

    def commonmemo_record(event: Any) -> None:
        for translated in translate(event):
            session.record(translated)

    hooks.register(commonmemo_record)
    return hooks


def prompt_with_context(context: MemoryContext, prompt: str, *, max_chars: int = 4000) -> str:
    """`prompt`, preceded by the rendered memory context when there is any."""
    rendered = context.render(max_chars=max_chars)
    if not rendered:
        return prompt
    return f"# Memory\n\n{rendered}\n\n# Request\n\n{prompt}"


async def run_with_memory(
    session: Session,
    runner: Any,
    project: Any,
    agent_name: str,
    prompt: str,
    *,
    run_session: Optional[Any] = None,
    hooks: Optional[Any] = None,
    episodes: int = 3,
    recent: int = 10,
) -> Any:
    """One CommonADK turn with all four tiers wired in: fetch context, run
    with the memory hook attached, return CommonADK's `Trace` unchanged.

    `run_session` is CommonADK's own `RunSession` (its per-SDK conversation
    state); pass the same one on every turn of a conversation, exactly as
    you would without memory.
    """
    context = await session.context(prompt, episodes=episodes, recent=recent)
    return await runner.run(
        project,
        agent_name,
        prompt_with_context(context, prompt),
        session=run_session,
        hooks=attach(session, hooks),
    )


__all__ = ["translate", "attach", "prompt_with_context", "run_with_memory"]
