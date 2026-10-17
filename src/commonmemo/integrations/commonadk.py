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

How each CommonADK event lands in the Log:

- every event is kept verbatim as an `ExternalEvent(source="commonadk.<type>")`
  -- token counts, cost, transfers and errors included, nothing dropped;
- `run_started.prompt` additionally becomes a `Message(role="user")` and
  `run_finished.final_text` a `Message(role="assistant")`;
- `tool_call` becomes a `ToolInvoked` (instead of an `ExternalEvent`), so a
  CommonADK tool call and an `@mem.track`-ed call read the same downstream.

Event ids are derived from CommonADK's `run_id` and `seq`, so a hook that
fires twice for the same event never records it twice.

And into the run (`run_with_memory`), hybrid:

- *push* -- in front of the prompt, the only place released CommonADK can
  take it. A runner that accepts `context=` / `turn_context=` (proposed in
  CommonADK issue #30) gets stable memory as `context=` (instructions) and
  per-turn memory as `turn_context=` instead -- detected from its signature;
- *pull* -- the memory tools added to the agent, and tool memory's notes
  written into the agent's own tools' descriptions (`prepare_project`).

CommonMemo changes nothing in CommonADK; see docs/design.md, "CommonADK
integration".

CommonADK is imported lazily, only when a function here is called --
`import commonmemo` never needs it installed.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Optional

from ..context import MemoryContext, MemoryPolicy
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


def prompt_with_context(context: MemoryContext, prompt: str, *, max_chars: int = 8000) -> str:
    """`prompt`, preceded by all of the rendered memory when there is any --
    the delivery for a CommonADK without `context=` / `turn_context=`."""
    rendered = context.render(max_chars=max_chars)
    if not rendered:
        return prompt
    return f"{rendered}\n\n# Request\n\n{prompt}"


def _annotated(tool_spec: Any, notes: list[str]) -> Any:
    """A copy of a CommonADK `ToolSpec` whose description -- and the
    function's own docstring, which five of the six adapters read -- ends
    with tool memory's notes. The function's behaviour is unchanged."""
    func = tool_spec.func
    doc = (tool_spec.description or "").rstrip() + "\n\nNotes from past calls of this tool:\n" + "\n".join(
        f"- {n}" for n in notes
    )
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    wrapper.__doc__ = doc
    return tool_spec.model_copy(update={"func": wrapper, "description": doc})


def prepare_project(
    session: Session,
    project: Any,
    agent_name: str,
    *,
    memory_tools: bool = True,
    writes: bool = True,
    tool_notes: bool = True,
) -> Any:
    """A copy of `project` with memory built into `agent_name`:

    - the memory tools (`memory_recall`, and unless `writes=False` the write
      tools) added to the agent's tools -- the *pull* half of hybrid memory;
    - tool memory's notes appended to each of the agent's own tools'
      descriptions, where the model reads them exactly when deciding how to
      call that tool (and where they stay stable, cache-friendly text).

    `project` itself is never mutated. Needs commonadk installed.
    """
    try:
        from commonadk.models import ToolSpec
    except ImportError as exc:  # pragma: no cover -- exercised only without commonadk
        raise ImportError(_INSTALL_HINT) from exc
    if agent_name not in project.agents:
        return project  # let CommonADK's build() report the unknown agent
    spec = project.agents[agent_name]
    tools = list(spec.tools)
    if tool_notes:
        notes = session.memory.tool_notes(session.agent or agent_name)
        tools = [_annotated(t, notes[t.name]) if t.name in notes else t for t in tools]
    if memory_tools:
        taken = {t.name for t in tools}
        tools += [ToolSpec.from_function(f) for f in session.memory.agent_tools(session, writes=writes)
                  if f.__name__ not in taken]
    agents = dict(project.agents)
    agents[agent_name] = spec.model_copy(update={"tools": tools})
    return project.model_copy(update={"agents": agents})


def _takes_split_context(runner: Any) -> bool:
    try:
        params = inspect.signature(runner.run).parameters
    except (TypeError, ValueError):
        return False
    return "context" in params and "turn_context" in params


async def run_with_memory(
    session: Session,
    runner: Any,
    project: Any,
    agent_name: str,
    prompt: str,
    *,
    run_session: Optional[Any] = None,
    hooks: Optional[Any] = None,
    policy: Optional[MemoryPolicy] = None,
    memory_tools: bool = True,
    writes: bool = True,
) -> Any:
    """One CommonADK turn with hybrid memory wired in, returning CommonADK's
    `Trace` unchanged:

    1. *push* -- the session's context, in front of the prompt; or, when
       the runner accepts `context=` / `turn_context=` (CommonADK issue
       #30), split: stable memory to the instructions, per-turn memory to
       the prompt;
    2. *pull* -- memory tools on the agent, and tool notes in its tools'
       descriptions (`prepare_project`);
    3. *capture* -- every event of the run recorded into the session.

    `run_session` is CommonADK's own `RunSession` (its per-SDK conversation
    state); pass the same one on every turn of a conversation.
    """
    from dataclasses import replace

    policy = replace(policy or session.memory.policy, tool_notes_in_text=False)
    context = await session.context(prompt, policy=policy)
    if run_session is not None and getattr(run_session, "user_id", "absent") is None:
        run_session.user_id = session.user_id
    built = prepare_project(session, project, agent_name, memory_tools=memory_tools, writes=writes)
    registry = attach(session, hooks)
    if _takes_split_context(runner):
        return await runner.run(
            built, agent_name, prompt, session=run_session, hooks=registry,
            context=context.stable_text() or None, turn_context=context.turn_text() or None,
        )
    return await runner.run(
        built, agent_name, prompt_with_context(context, prompt), session=run_session, hooks=registry
    )


__all__ = ["translate", "attach", "prompt_with_context", "prepare_project", "run_with_memory"]
