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


class FakeRunner:
    """Fires a realistic event sequence through whatever hooks it is given."""

    def __init__(self):
        self.prompts = []

    async def run(self, project, agent_name, prompt, *, session=None, hooks=None):
        self.prompts.append(prompt)
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


def test_translate_maps_tool_calls_and_keeps_everything_else_verbatim():
    tool = translate(ToolCall(run_id="r", agent_name="a", tool_name="t", arguments={"x": 1}, error="boom"))
    assert len(tool) == 1 and isinstance(tool[0], ToolInvoked)
    assert (tool[0].name, tool[0].agent, tool[0].error) == ("t", "a", "boom")

    finished = translate(RunFinished(run_id="r", final_text="bye", total_tokens=5, total_cost_usd=None))
    assert isinstance(finished[0], ExternalEvent) and finished[0].source == "commonadk.run_finished"
    assert finished[0].data["total_tokens"] == 5 and finished[0].data["total_cost_usd"] is None
    assert isinstance(finished[1], Message) and finished[1].role == "assistant"


def test_run_with_memory_records_and_injects_context():
    mem = Memory()
    runner = FakeRunner()

    async def main():
        async with mem.session(user_id="u1", agent="coordinator") as s:
            s.remember("name", "Ada")
            await run_with_memory(s, runner, object(), "coordinator", "find item 7")
        return s

    s = asyncio.run(main())
    events = mem.store.events(session_id=s.session_id)
    kinds = [e.kind for e in events]
    assert kinds.count("tool_invoked") == 1
    assert "transfer" in {e.source.split(".")[1] for e in events if isinstance(e, ExternalEvent)}
    assert [m.content for m in events if isinstance(m, Message)] == [runner.prompts[0], "Done: item 7 found."]
    # The fact recorded before the run reached the agent's prompt.
    assert "name: Ada" in runner.prompts[0] and runner.prompts[0].endswith("find item 7")
    (episode,) = s.episodes
    assert "lookup" in episode.summary


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
