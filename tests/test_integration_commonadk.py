"""CommonADK integration, against CommonADK's real event model and hook
registry. No agent SDK is needed: a stand-in runner fires the same events a
real runner does, through the same `HookRegistry`."""

import asyncio

import pytest

pytest.importorskip("commonadk")

from commonadk.runners.events import RunFinished, RunStarted, ToolCall, Transfer  # noqa: E402
from commonadk.runners.hooks import HookRegistry  # noqa: E402

from commonmemo import ExternalEvent, Memory, Message, ToolInvoked  # noqa: E402
from commonmemo.integrations.commonadk import attach, run_with_memory, translate  # noqa: E402


def lookup(item_id: int) -> str:
    """Look up an item by id."""
    return f"item {item_id}"


def make_project():
    from commonadk.models import AgentConfig, AgentSpec, Project, ProjectConfig, ToolSpec

    return Project(
        config=ProjectConfig(name="p", default_model="anthropic/claude-haiku-4-5"),
        agents={
            "coordinator": AgentSpec(
                config=AgentConfig(name="coordinator"),
                instructions="You coordinate.",
                tools=[ToolSpec.from_function(lookup)],
            )
        },
    )


class FakeRunner:
    """Fires a realistic event sequence through whatever hooks it is given,
    and records what it was asked to run. Has CommonADK's split-context
    signature (`context=`, `turn_context=`)."""

    def __init__(self):
        self.calls = []

    async def run(self, project, agent_name, prompt, *, session=None, hooks=None, context=None, turn_context=None):
        self.calls.append(dict(project=project, prompt=prompt, context=context, turn_context=turn_context))
        events = [
            RunStarted(run_id="r1", target="openai", agent_name=agent_name, prompt=prompt),
            ToolCall(run_id="r1", agent_name=agent_name, tool_name="lookup", arguments={"id": 7},
                     result_summary="found", duration_ms=3.0),
            Transfer(run_id="r1", from_agent=agent_name, to_agent="writer", transfer_kind="delegate"),
            RunFinished(run_id="r1", final_text="Done: item 7 found.", total_tokens=120),
        ]
        for e in events:
            hooks.fire(e)
        return "trace"


class LegacyRunner(FakeRunner):
    """A CommonADK runner from before `context=` / `turn_context=`."""

    async def run(self, project, agent_name, prompt, *, session=None, hooks=None):
        return await super().run(project, agent_name, prompt, session=session, hooks=hooks)


def test_translate_maps_tool_calls_and_keeps_everything_else_verbatim():
    tool = translate(ToolCall(run_id="r", agent_name="a", tool_name="t", arguments={"x": 1}, error="boom"))
    assert len(tool) == 1 and isinstance(tool[0], ToolInvoked)
    assert (tool[0].name, tool[0].agent, tool[0].error) == ("t", "a", "boom")

    finished = translate(RunFinished(run_id="r", final_text="bye", total_tokens=5, total_cost_usd=None))
    assert isinstance(finished[0], ExternalEvent) and finished[0].source == "commonadk.run_finished"
    assert finished[0].data["total_tokens"] == 5 and finished[0].data["total_cost_usd"] is None
    assert isinstance(finished[1], Message) and finished[1].role == "assistant"
    # Every translated event carries the CommonADK run as its run_id.
    assert {e.run_id for e in tool + finished} == {"commonadk:r"}


def test_run_with_memory_splits_stable_and_turn_memory_and_records():
    mem = Memory()
    runner = FakeRunner()
    project = make_project()

    async def main():
        async with mem.session("s0", user_id="u1", agent="coordinator") as old:
            old.message("user", "where is item 7?")
        async with mem.session(user_id="u1", agent="coordinator") as s:
            s.remember("name", "Ada")
            s.never("suggest phone calls")
            await run_with_memory(s, runner, project, "coordinator", "find item 7")
        return s

    s = asyncio.run(main())
    (call,) = runner.calls
    # Stable memory to the instructions, per-turn memory to the prompt; the prompt itself untouched.
    assert "name: Ada" in call["context"] and "Never: suggest phone calls" in call["context"]
    assert "item 7" in call["turn_context"] and "name: Ada" not in call["turn_context"]
    assert call["prompt"] == "find item 7"
    # The pull half: memory tools on the agent, the caller's project untouched.
    names = [t.name for t in call["project"].agents["coordinator"].tools]
    assert names[0] == "lookup" and "memory_recall" in names and "memory_remember" in names
    assert [t.name for t in project.agents["coordinator"].tools] == ["lookup"]
    # Capture.
    events = mem.store.events(session_id=s.session_id)
    assert [e.kind for e in events].count("tool_invoked") == 1
    assert "transfer" in {e.source.split(".")[1] for e in events if isinstance(e, ExternalEvent)}
    assert [m.content for m in events if isinstance(m, Message)] == ["find item 7", "Done: item 7 found."]
    (recap,) = s.recaps
    assert "lookup" in recap.summary


def test_run_with_memory_falls_back_to_the_prompt_on_older_commonadk():
    mem = Memory()
    runner = LegacyRunner()

    async def main():
        async with mem.session(user_id="u1", agent="coordinator") as s:
            s.remember("name", "Ada")
            await run_with_memory(s, runner, make_project(), "coordinator", "find item 7")

    asyncio.run(main())
    (call,) = runner.calls
    assert "name: Ada" in call["prompt"] and call["prompt"].endswith("find item 7")


def test_prepare_project_writes_tool_notes_into_the_tools_own_docs():
    from commonmemo import ToolInvoked
    from commonmemo.integrations.commonadk import prepare_project

    mem = Memory()
    with mem.session(user_id="u1", agent="coordinator") as s:
        s.record(ToolInvoked(name="lookup", arguments={"item_id": "7/10/2026"}, error="ValueError: bad id '7/10'"))
        s.record(ToolInvoked(name="lookup", arguments={"item_id": 7}, result="item 7"))
    with mem.session(user_id="u2", agent="coordinator") as s2:
        built = prepare_project(s2, make_project(), "coordinator", memory_tools=False)
    (tool,) = built.agents["coordinator"].tools
    assert "was fixed by: item_id: dmy_or_mdy_date -> int" in tool.description
    assert tool.func.__doc__ == tool.description  # what five of six adapters read
    assert tool.func(item_id=3) == "item 3"  # behaviour unchanged
    assert "u1" not in tool.description and "7/10" not in tool.description  # no user data crosses users


def test_attach_adds_to_an_existing_registry_and_deduplicates():
    mem = Memory()
    seen = []
    hooks = HookRegistry()
    hooks.register(seen.append)
    with mem.session(user_id="u1") as s:
        attach(s, hooks)
        event = ToolCall(run_id="r9", agent_name="a", tool_name="t")
        hooks.fire(event)
        hooks.fire(event)  # same CommonADK event twice -> recorded once
    assert seen == [event, event]
    assert len(mem.store.events(kinds=["tool_invoked"])) == 1
    assert hooks.errors == []
