"""The push says when it left things out, if the agent can recall (#23)."""

from __future__ import annotations

import asyncio

from commonmemo import Memory, MemoryPolicy
from commonmemo.agent_tools import build_agent_tools
from commonmemo.context import TURN_HEADER


def ctx_for(s, query: str = ""):
    return asyncio.run(s.context(query))


def with_facts(mem: Memory, n: int) -> None:
    with mem.session(user_id="ada") as s:
        for i in range(n):
            s.remember(f"fact_{i:02d}", f"value {i}")


def test_cut_facts_are_counted():
    mem = Memory()
    with_facts(mem, 25)
    with mem.session(user_id="ada") as s:
        build_agent_tools(s)
        ctx = ctx_for(s)
        assert len(ctx.facts) == 20
        assert ctx.more_line() == ("More in memory than shown here: 5 facts about the user. "
                                   "Call memory_recall with what you need.")
        turn = ctx.turn_text()
        assert turn.startswith(TURN_HEADER) and turn.endswith(ctx.more_line())  # inside the data framing


def test_silent_when_nothing_was_cut():
    mem = Memory()
    with_facts(mem, 3)
    with mem.session(user_id="ada") as s:
        build_agent_tools(s)
        ctx = ctx_for(s)
        assert ctx.more_line() == "" and ctx.turn_text() == ""


def test_silent_without_a_recall_tool():
    mem = Memory()
    with_facts(mem, 25)
    with mem.session(user_id="ada") as s:  # no agent tools: nothing to point at
        ctx = ctx_for(s)
        assert ctx.more["facts"] == 5 and ctx.more_line() == "" and "More in memory" not in ctx.render()


def test_facts_stated_in_this_session_count_too():
    mem = Memory()
    with mem.session(user_id="ada") as s:
        build_agent_tools(s)
        for i in range(22):
            s.remember(f"fact_{i:02d}", f"value {i}")
        assert "2 facts about the user" in ctx_for(s).more_line()


def test_recaps_past_the_limit_that_nearly_qualify():
    mem = Memory(policy=MemoryPolicy(recaps=1))
    for day in range(3):
        with mem.session(user_id="ada") as s:
            s.message("user", f"the charger firmware rollback on unit {day}")
    with mem.session(user_id="ada") as s:  # unrelated: past sessions that don't match are not counted
        s.message("user", "lunch order for friday")
    with mem.session(user_id="ada") as s:
        build_agent_tools(s)
        ctx = ctx_for(s, "charger firmware rollback")
        assert len(ctx.recaps) == 1
        assert "2 past sessions" in ctx.more_line()


def test_other_kinds_and_plurals():
    mem = Memory(policy=MemoryPolicy(max_negatives=1, max_world=1))
    mem.learn("refund_window", "30 days")
    mem.learn("shipping", "2 days")
    mem.learn("returns", "free")
    with mem.session(user_id="ada") as s:
        s.never("call me")
        s.never("send faxes")
        build_agent_tools(s)
        line = ctx_for(s).more_line()
    assert "1 standing constraint" in line and "1 standing constraints" not in line and "2 known facts" in line


def test_entries_cut_for_length_and_the_line_is_never_cut():
    mem = Memory(policy=MemoryPolicy(max_chars_stable=200, max_chars_turn=150))
    with_facts(mem, 15)
    with mem.session(user_id="ada") as s:
        build_agent_tools(s)
        ctx = ctx_for(s)
        line = ctx.more_line()
        assert "entries cut for length" in line
        assert ctx.turn_text().endswith(line)
