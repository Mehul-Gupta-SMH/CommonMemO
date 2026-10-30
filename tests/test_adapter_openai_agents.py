"""The OpenAI Agents adapter (#110), against the installed SDK's Session protocol and its own SQLiteSession."""

import asyncio

import pytest

pytest.importorskip("agents")

from agents.memory import Session, SQLiteSession  # noqa: E402

from commonmemo import Memory  # noqa: E402
from commonmemo.adapters.openai_agents import CommonMemoSession  # noqa: E402
from commonmemo.agent_tools import recall_entries  # noqa: E402

ITEMS = [
    {"role": "user", "content": "My daughter Maya is allergic to peanuts, so no peanut snacks."},
    {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{\"q\": \"snacks\"}"},
    {"type": "function_call_output", "call_id": "c1", "output": "oat bars"},
    {"role": "assistant", "content": [{"type": "output_text", "text": "Oat bars are safe for Maya."}]},
]


@pytest.fixture
def mem(tmp_path):
    return Memory(f"sqlite:///{tmp_path}/m.db", project="shop")


def run(coro):
    return asyncio.run(coro)


def test_is_an_openai_agents_session(mem):
    assert isinstance(CommonMemoSession(mem, "s1", user_id="ada"), Session)


def test_items_come_back_exactly_as_added(mem):
    s = CommonMemoSession(mem, "s1", user_id="ada")
    run(s.add_items(ITEMS))
    assert run(s.get_items()) == ITEMS
    assert run(s.get_items(limit=2)) == ITEMS[-2:]
    assert run(s.get_items(limit=0)) == []


def test_it_behaves_like_the_sdks_own_sqlite_session(mem):
    ours, theirs = CommonMemoSession(mem, "s1", user_id="ada"), SQLiteSession("s1")
    for session in (ours, theirs):
        run(session.add_items(ITEMS[:2]))
        run(session.add_items(ITEMS[2:]))
    assert run(ours.get_items()) == run(theirs.get_items())
    assert run(ours.pop_item()) == run(theirs.pop_item())
    assert run(ours.get_items(limit=2)) == run(theirs.get_items(limit=2))
    for session in (ours, theirs):
        run(session.clear_session())
    assert run(ours.get_items()) == run(theirs.get_items()) == []
    assert run(ours.pop_item()) is run(theirs.pop_item()) is None


def test_a_later_session_object_sees_the_same_items(mem):
    run(CommonMemoSession(mem, "s1", user_id="ada").add_items(ITEMS))
    again = CommonMemoSession(mem, "s1", user_id="ada")
    assert run(again.get_items()) == ITEMS


def test_pop_and_clear_append_to_the_log_and_delete_nothing(mem):
    s = CommonMemoSession(mem, "s1", user_id="ada")
    run(s.add_items(ITEMS))
    before = len(mem.store.events())
    run(s.pop_item())
    run(s.clear_session())
    assert len(mem.store.events()) == before + 2
    run(s.add_items(ITEMS[:1]))
    assert run(s.get_items()) == ITEMS[:1]


def test_what_the_user_said_is_searchable_from_another_session(mem):
    run(CommonMemoSession(mem, "s1", user_id="ada").add_items(ITEMS))
    other = mem.session("s2", user_id="ada")
    lines, _, _ = recall_entries(other, "which snacks is Maya allergic to?")
    assert any(label == "said" and "peanut" in text for label, text in lines)


def test_sessions_are_separate(mem):
    run(CommonMemoSession(mem, "s1", user_id="ada").add_items(ITEMS))
    assert run(CommonMemoSession(mem, "s2", user_id="ada").get_items()) == []


def test_another_user_cannot_resume_a_session(mem):
    run(CommonMemoSession(mem, "s1", user_id="ada").add_items(ITEMS))
    with pytest.raises(PermissionError):
        CommonMemoSession(mem, "s1", user_id="grace")
