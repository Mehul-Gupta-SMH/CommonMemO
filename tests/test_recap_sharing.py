"""A user's recaps are shared across their agents (harnesses) (#34)."""

from __future__ import annotations

import asyncio
from typing import Optional

from commonmemo import Memory, MemoryPolicy
from commonmemo.agent_tools import build_agent_tools
from commonmemo.context import MemoryContext
from commonmemo.engines.base import Recap

TOPIC = "the search_flights date parsing breaks on slashes"
QUERY = "search_flights date parsing"


def recall(session, query: str) -> str:
    return {t.__name__: t for t in build_agent_tools(session)}["memory_recall"](query)


def worked_in(mem: Memory, agent: str, *, user: Optional[str] = "ada", text: str = TOPIC) -> None:
    with mem.session(user_id=user, agent=agent) as s:
        s.message("user", text)


def test_a_users_recap_from_another_agent_is_pushed():
    mem = Memory()
    worked_in(mem, "codex")
    with mem.session(user_id="ada", agent="claude-code") as s:
        ctx = asyncio.run(s.context(QUERY))
        assert [r.agent for r in ctx.recaps] == ["codex"]
        assert "slashes" in ctx.turn_text() and "codex" in ctx.turn_text()
        assert "slashes" in recall(s, "search_flights")


def test_recap_line_names_another_agent_once():
    ctx = MemoryContext(agent="claude-code")

    def recap(summary: str, agent: str) -> Recap:
        return Recap(id="r", user_id="ada", agent=agent, session_id="s", summary=summary, started_at=0, ended_at=0)

    assert ctx.recap_line(recap("Fixed the parser.", "codex")) == "Fixed the parser. (in codex)"
    assert ctx.recap_line(recap("Fixed the parser.", "claude-code")) == "Fixed the parser."
    assert ctx.recap_line(recap("Session with agent codex: fixed it.", "codex")) == "Session with agent codex: fixed it."


def test_per_agent_when_turned_off():
    mem = Memory(policy=MemoryPolicy(recaps_across_agents=False))
    worked_in(mem, "codex")
    with mem.session(user_id="ada", agent="claude-code") as s:
        assert asyncio.run(s.context(QUERY)).recaps == []
        assert "slashes" not in recall(s, "search_flights")  # recall follows the policy too


def test_without_a_user_recaps_stay_per_agent():
    mem = Memory()
    worked_in(mem, "codex", user=None)
    with mem.session(agent="claude-code") as s:
        assert asyncio.run(s.context(QUERY)).recaps == []
        assert "slashes" not in recall(s, "search_flights")


def test_another_users_recaps_never_appear():
    mem = Memory()
    worked_in(mem, "codex", user="bob", text="bob's secret project: " + TOPIC)
    with mem.session(user_id="ada", agent="claude-code") as s:
        assert asyncio.run(s.context(QUERY)).recaps == []
        assert "secret" not in recall(s, "search_flights date parsing secret project")
