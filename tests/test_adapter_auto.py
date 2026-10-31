"""commonmemo.instrument (#116): defaults get patched, explicit choices win, uninstrument restores."""

import pytest

import commonmemo
from commonmemo import Memory
from commonmemo.adapters import auto


@pytest.fixture
def mem(tmp_path):
    return Memory(f"sqlite:///{tmp_path}/m.db", project="shop")


@pytest.fixture(autouse=True)
def clean():
    yield
    commonmemo.uninstrument()


def test_unknown_framework_rejected(mem):
    with pytest.raises(ValueError):
        commonmemo.instrument(mem, ["nope"])


def test_missing_framework_warns_only_when_named(mem, monkeypatch):
    monkeypatch.setitem(auto.HOOKS, "ghost", ("not_installed_pkg_xyz", lambda *a: None))
    assert "ghost" not in commonmemo.instrument(mem)
    with pytest.warns(UserWarning, match="not installed"):
        assert "ghost" not in commonmemo.instrument(mem, ["ghost"])


def test_broken_hook_warns_and_skips(mem, monkeypatch):
    def boom(*a):
        raise RuntimeError("moved")
    monkeypatch.setitem(auto.HOOKS, "adk", ("json", boom))
    with pytest.warns(UserWarning, match="could not hook adk"):
        assert commonmemo.instrument(mem, ["adk"]) == []


def test_resolvers():
    assert auto._value(None, "x") == "x"
    assert auto._value("u", "x") == "u"
    assert auto._value(lambda: "v", "x") == "v"


def test_adk(mem):
    pytest.importorskip("google.adk")
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from commonmemo.adapters.adk import CommonMemoMemoryService

    original = Runner.__init__
    assert "adk" in commonmemo.instrument(mem, ["adk"])
    kw = dict(app_name="a", agent=Agent(name="x", model="gemini-2.0-flash"), session_service=InMemorySessionService())
    assert isinstance(Runner(**kw).memory_service, CommonMemoMemoryService)
    own = CommonMemoMemoryService(mem)
    assert Runner(memory_service=own, **kw).memory_service is own
    commonmemo.uninstrument()
    assert Runner.__init__ is original
    assert Runner(**kw).memory_service is None


def test_langgraph(mem):
    pytest.importorskip("langgraph")
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph
    from langgraph.store.memory import InMemoryStore
    from commonmemo.adapters.langgraph import CommonMemoStore

    class S(TypedDict):
        x: int

    def graph():
        g = StateGraph(S)
        g.add_node("n", lambda s: s)
        g.add_edge(START, "n")
        g.add_edge("n", END)
        return g

    commonmemo.instrument(mem, ["langgraph"], user_id="u1")
    assert isinstance(graph().compile().store, CommonMemoStore)
    own = InMemoryStore()
    assert graph().compile(store=own).store is own
    commonmemo.uninstrument()
    assert graph().compile().store is None


def test_openai_agents(mem, monkeypatch):
    pytest.importorskip("agents")
    from agents import Agent, Runner
    from commonmemo.adapters.openai_agents import CommonMemoSession

    seen = []

    class Stop(Exception):
        pass

    from agents.run import AgentRunner

    def spy(self, *a, **kw):
        seen.append(kw.get("session"))
        raise Stop

    original_run_sync = Runner.__dict__["run_sync"]
    assert "openai_agents" in commonmemo.instrument(mem, ["openai_agents"], user_id=lambda: "u7", session_id="s9")
    monkeypatch.setattr(AgentRunner, "run_sync", spy)
    agent = Agent(name="a")
    with pytest.raises(Stop):
        Runner.run_sync(agent, "hi")
    assert isinstance(seen[0], CommonMemoSession)
    assert (seen[0].session_id, seen[0].user_id) == ("s9", "u7")
    with pytest.raises(Stop):
        Runner.run_sync(agent, "hi", session="mine")
    assert seen[1] == "mine"
    commonmemo.uninstrument()
    assert Runner.__dict__["run_sync"] is original_run_sync


def test_autogen(mem):
    pytest.importorskip("autogen_agentchat")
    pytest.importorskip("autogen_ext")
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.memory import ListMemory
    from autogen_ext.models.replay import ReplayChatCompletionClient
    from commonmemo.adapters.autogen import CommonMemoMemory

    def agent(**kw):
        return AssistantAgent("helper", model_client=ReplayChatCompletionClient(["ok"]), **kw)

    original = AssistantAgent.__init__
    assert "autogen" in commonmemo.instrument(mem, ["autogen"], user_id="ana")
    (hooked,) = agent()._memory
    assert isinstance(hooked, CommonMemoMemory) and hooked.user_id == "ana"
    own = ListMemory()
    assert agent(memory=[own])._memory == [own]
    commonmemo.uninstrument()
    assert AssistantAgent.__init__ is original
    assert not agent()._memory


def test_claude_agent_sdk(mem):
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ClaudeAgentOptions, InMemorySessionStore
    from commonmemo.adapters.claude_agent_sdk import CommonMemoSessionStore

    original = ClaudeAgentOptions.__init__
    assert "claude_agent_sdk" in commonmemo.instrument(mem, ["claude_agent_sdk"], user_id="ana")
    hooked = ClaudeAgentOptions().session_store
    assert isinstance(hooked, CommonMemoSessionStore) and hooked.user_id == "ana"
    own = InMemorySessionStore()
    assert ClaudeAgentOptions(session_store=own).session_store is own
    commonmemo.uninstrument()
    assert ClaudeAgentOptions.__init__ is original
    assert ClaudeAgentOptions().session_store is None
