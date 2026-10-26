"""Memory under Google ADK (or any CommonADK target), through CommonADK.

CommonADK turns Google ADK, OpenAI Agents, Claude Agent SDK, CrewAI, AutoGen and LangGraph
into one event stream, so `run_with_memory` gives all of them the same memory. This file shows
the whole loop and runs offline: a stand-in runner fires the events a real ADK run fires.
CommonADK itself is never changed.

    pip install "commonmemo[commonadk]"
    python examples/google_adk.py              # offline, no keys
    python examples/google_adk.py --live       # a real Google ADK run (needs google-adk and a model key; not run in CI)

Look for: the second turn's prompt starts with what the user said in the first, and the
`lookup` tool's description carries the note learned from its failure.
"""

import asyncio
import sys

from commonadk.models import AgentConfig, AgentSpec, Project, ProjectConfig, ToolSpec
from commonadk.runners.events import RunFinished, RunStarted, ToolCall

from commonmemo import Memory
from commonmemo.integrations.commonadk import run_with_memory


def lookup(item_id: int) -> str:
    """Look up an item by id."""
    return f"item {item_id}"


def make_project() -> Project:
    return Project(
        config=ProjectConfig(name="shop", default_model="anthropic/claude-haiku-4-5"),
        agents={"coordinator": AgentSpec(config=AgentConfig(name="coordinator"), instructions="You help with orders.",
                                         tools=[ToolSpec.from_function(lookup)])},
    )


class OfflineRunner:
    """Fires the events a real run fires, through the hooks CommonMemo attached."""

    async def run(self, project, agent_name, prompt, *, session=None, hooks=None, **_):
        print(f"--- prompt the model would see ---\n{prompt}\n")
        print(f"--- lookup's description ---\n{project.agents[agent_name].tools[0].description}\n")
        for e in (
            RunStarted(run_id="r1", target="google-adk", agent_name=agent_name, prompt=prompt),
            ToolCall(run_id="r1", agent_name=agent_name, tool_name="lookup", arguments={"item_id": "7/10"},
                     error="ValueError: bad id '7/10'"),
            ToolCall(run_id="r1", agent_name=agent_name, tool_name="lookup", arguments={"item_id": 7},
                     result_summary="item 7"),
            RunFinished(run_id="r1", final_text="Item 7 is in stock."),
        ):
            hooks.fire(e)


async def main(live: bool) -> None:
    if live:
        from commonadk.runners import get_runner

        runner = get_runner("google-adk")
    else:
        runner = OfflineRunner()
    mem = Memory()  # in memory; use "sqlite:///memory.db" to keep it
    project = make_project()
    for prompt in ("I'm Ada and I only buy vegan items. Is item 7 in stock?", "What did I say I only buy?"):
        async with mem.session(user_id="ada", agent="coordinator") as s:
            await run_with_memory(s, runner, project, "coordinator", prompt)


if __name__ == "__main__":
    asyncio.run(main("--live" in sys.argv))
