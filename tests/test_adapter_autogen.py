"""The AutoGen adapter (#112), against the installed SDK's Memory interface and ListMemory."""

import asyncio

import pytest

pytest.importorskip("autogen_core")

from autogen_core.memory import ListMemory, Memory as AutoGenMemory, MemoryContent, MemoryMimeType  # noqa: E402
from autogen_core.model_context import UnboundedChatCompletionContext  # noqa: E402
from autogen_core.models import SystemMessage, UserMessage  # noqa: E402

from commonmemo import Memory  # noqa: E402
from commonmemo.adapters.autogen import CommonMemoMemory  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def text(s, **meta):
    return MemoryContent(content=s, mime_type=MemoryMimeType.TEXT, metadata=meta or None)


@pytest.fixture
def mem(tmp_path):
    return Memory(f"sqlite:///{tmp_path}/m.db", project="shop")


@pytest.fixture
def am(mem):
    return CommonMemoMemory(mem, user_id="ana", agent="helper")


def test_is_an_autogen_memory(am):
    assert isinstance(am, AutoGenMemory)


def test_add_then_query(am):
    run(am.add(text("Ana is allergic to peanuts", key="allergy")))
    found = run(am.query("what is Ana allergic to"))
    assert any("peanuts" in r.content for r in found.results)
    assert all(r.mime_type == MemoryMimeType.TEXT and "kind" in r.metadata for r in found.results)


def test_adding_twice_is_one_fact(mem, am):
    run(am.add(text("Ana likes oat bars")))
    run(am.add(text("Ana likes oat bars")))
    hits = [r for r in run(am.query("oat bars")).results if "oat bars" in r.content and r.metadata["kind"] == "user"]
    assert len(hits) == 1


def test_dict_content_and_refusal(am):
    run(am.add(MemoryContent(content={"diet": "vegan"}, mime_type=MemoryMimeType.JSON)))
    assert any("vegan" in r.content for r in run(am.query("diet")).results)
    with pytest.raises(TypeError):
        run(am.add(MemoryContent(content=b"\x00", mime_type=MemoryMimeType.BINARY)))


def test_users_are_separate(mem, am):
    run(am.add(text("Ana's code word is kumquat")))
    other = CommonMemoMemory(mem, user_id="bo", agent="helper")
    assert not any("kumquat" in r.content for r in run(other.query("code word")).results)


def test_update_context_injects_a_system_message(am):
    run(am.add(text("Ana is allergic to peanuts")))
    ctx = UnboundedChatCompletionContext()
    run(ctx.add_message(UserMessage(content="Any snack ideas given Ana's allergy?", source="user")))
    result = run(am.update_context(ctx))
    messages = run(ctx.get_messages())
    assert isinstance(messages[-1], SystemMessage) and "peanuts" in messages[-1].content
    assert result.memories.results


def test_update_context_matches_listmemory_shape(am):
    """ListMemory is the SDK's reference: one system message appended, results returned."""
    ref = ListMemory()
    run(ref.add(text("Ana is allergic to peanuts")))
    run(am.add(text("Ana is allergic to peanuts")))
    for memory in (ref, am):
        ctx = UnboundedChatCompletionContext()
        run(ctx.add_message(UserMessage(content="peanuts?", source="user")))
        run(memory.update_context(ctx))
        msgs = run(ctx.get_messages())
        assert len(msgs) == 2 and isinstance(msgs[1], SystemMessage)


def test_inject_off_and_empty_context(mem):
    quiet = CommonMemoMemory(mem, user_id="ana", inject=False)
    run(quiet.add(text("Ana is allergic to peanuts")))
    ctx = UnboundedChatCompletionContext()
    run(ctx.add_message(UserMessage(content="peanuts?", source="user")))
    assert run(quiet.update_context(ctx)).memories.results == []
    assert len(run(ctx.get_messages())) == 1
    assert run(CommonMemoMemory(mem, user_id="ana").update_context(UnboundedChatCompletionContext())).memories.results == []


def test_clear_forgets_the_user(mem, am):
    run(am.add(text("Ana's code word is kumquat")))
    run(am.clear())
    assert not any("kumquat" in r.content for r in run(am.query("code word")).results)
    with pytest.raises(ValueError):
        run(CommonMemoMemory(mem).clear())


def test_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    run(CommonMemoMemory(Memory(url, project="shop"), user_id="ana").add(text("Ana is allergic to peanuts")))
    again = CommonMemoMemory(Memory(url, project="shop"), user_id="ana")
    assert any("peanuts" in r.content for r in run(again.query("allergic")).results)


def test_assistant_agent_accepts_it(am):
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.replay import ReplayChatCompletionClient
    agent = AssistantAgent("helper", model_client=ReplayChatCompletionClient(["ok"]), memory=[am])
    assert agent is not None
