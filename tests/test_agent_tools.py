"""The pull half of hybrid memory: tools agents call."""

import asyncio
import inspect

from commonmemo import Memory


def tools_by_name(mem, s, **kw):
    return {f.__name__: f for f in mem.agent_tools(s, **kw)}


def test_tools_are_plain_typed_documented_sync_functions():
    mem = Memory()
    with mem.session(user_id="u1") as s:
        tools = mem.agent_tools(s)
    assert [t.__name__ for t in tools] == [
        "memory_recall", "memory_remember", "memory_note_fact", "memory_avoid", "memory_save_progress"]
    for t in tools:
        assert not inspect.iscoroutinefunction(t)  # not every framework awaits tools
        assert inspect.getdoc(t)
        hints = inspect.get_annotations(t, eval_str=True)  # as CommonADK's ToolSpec reads them
        assert hints.pop("return") is str and all(p in hints for p in inspect.signature(t).parameters)


def test_read_only_tools():
    mem = Memory()
    with mem.session(user_id="u1") as s:
        assert list(tools_by_name(mem, s, writes=False)) == ["memory_recall"]


def test_agent_writes_are_marked_and_take_effect():
    mem = Memory(project="shop")
    with mem.session(user_id="u1", agent="support") as s:
        t = tools_by_name(mem, s)
        assert t["memory_remember"]("preferred_contact", "email") == "Saved: preferred_contact = email"
        t["memory_note_fact"]("refund_window", "30 days")
        t["memory_avoid"]("suggest phone calls", "prefers email")
        t["memory_save_progress"]("drafted reply", '{"draft": 1}')
        events = s.events()
    assert {e.kind: getattr(e, "source", None) for e in events}["user_fact"] == "agent"
    assert mem.world_facts() == []  # an agent's world fact is a proposal, not yet everyone's
    assert [(w.key, w.source, w.status) for w in mem.pending_world_facts()] == [("refund_window", "agent", "proposed")]
    assert mem.negatives(["user:u1"])[0].source == "agent"
    assert s.resume_point().checkpoint.data == {"draft": 1}
    with mem.session(user_id="u1", agent="support") as s:
        text = asyncio.run(s.context("x")).stable_text()
    assert "preferred_contact: email" in text and "Never: suggest phone calls" in text


def test_recall_searches_wider_than_the_push_and_works_inside_an_event_loop():
    mem = Memory()
    with mem.session(user_id="u1") as s:
        s.message("user", "my ThinkPad laptop will not charge")
        s.remember("laptop", "ThinkPad X1")

    async def main():
        async with mem.session(user_id="u1") as s:
            return tools_by_name(mem, s)["memory_recall"]("laptop")  # sync tool, called from async code

    out = asyncio.run(main())
    assert "[user] laptop: ThinkPad X1" in out and "[past session]" in out


def test_recall_with_nothing_and_remember_without_a_user():
    mem = Memory()
    with mem.session() as s:
        t = tools_by_name(mem, s)
        assert t["memory_recall"]("anything") == "Nothing in memory matches that."
        assert t["memory_remember"]("k", "v").startswith("Not saved")
