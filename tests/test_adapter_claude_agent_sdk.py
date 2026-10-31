"""The Claude Agent SDK adapter (#113), against the installed SDK's SessionStore protocol and its InMemorySessionStore."""

import asyncio

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import InMemorySessionStore  # noqa: E402

from commonmemo import Memory  # noqa: E402
from commonmemo.adapters.claude_agent_sdk import CommonMemoSessionStore  # noqa: E402
from commonmemo.agent_tools import recall_entries  # noqa: E402

MAIN = {"project_key": "proj", "session_id": "s1"}
SUB = {**MAIN, "subpath": "subagents/agent-1"}
ENTRIES = [
    {"type": "user", "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z",
     "message": {"role": "user", "content": "My daughter Maya is allergic to peanuts, so no peanut snacks."}},
    {"type": "assistant", "uuid": "u2", "timestamp": "2026-01-01T00:00:01Z",
     "message": {"role": "assistant", "content": [{"type": "text", "text": "Oat bars are safe for Maya."}]}},
    {"type": "summary", "summary": "snack talk"},
]


@pytest.fixture
def mem(tmp_path):
    return Memory(f"sqlite:///{tmp_path}/m.db", project="shop")


@pytest.fixture
def store(mem):
    return CommonMemoSessionStore(mem, user_id="ada")


def run(coro):
    return asyncio.run(coro)


def both(stores, call):
    return [run(call(s)) for s in stores]


def test_has_every_method_the_sdks_own_store_has(store):
    for name in ("append", "load", "list_sessions", "list_session_summaries", "delete", "list_subkeys"):
        assert callable(getattr(store, name)) and callable(getattr(InMemorySessionStore, name))


def test_entries_come_back_exactly_as_appended(store):
    assert run(store.load(MAIN)) is None
    run(store.append(MAIN, ENTRIES[:2]))
    run(store.append(MAIN, ENTRIES[2:]))
    assert run(store.load(MAIN)) == ENTRIES


def test_it_behaves_like_the_sdks_in_memory_store(store):
    ours, theirs = store, InMemorySessionStore()
    for s in (ours, theirs):
        run(s.append(MAIN, ENTRIES))
        run(s.append(SUB, ENTRIES[:1]))
        run(s.append({**MAIN, "session_id": "s2"}, ENTRIES[:1]))
        run(s.append({"project_key": "other", "session_id": "s3"}, ENTRIES[:1]))
    for key in (MAIN, SUB, {**MAIN, "session_id": "nope"}):
        a, b = both((ours, theirs), lambda s: s.load(key))
        assert a == b
    a, b = both((ours, theirs), lambda s: s.list_sessions("proj"))
    assert sorted(x["session_id"] for x in a) == sorted(x["session_id"] for x in b) == ["s1", "s2"]
    a, b = both((ours, theirs), lambda s: s.list_subkeys(MAIN))
    assert a == b == ["subagents/agent-1"]
    a, b = both((ours, theirs), lambda s: s.list_session_summaries("proj"))
    def strip(rows):
        return sorted(({k: v for k, v in r.items() if k != "mtime"} for r in rows), key=lambda r: r["session_id"])

    assert strip(a) == strip(b) and len(a) == 2
    for s in (ours, theirs):
        run(s.delete(MAIN))  # cascades to the subagent transcript
    for key in (MAIN, SUB):
        a, b = both((ours, theirs), lambda s: s.load(key))
        assert a is b is None
    a, b = both((ours, theirs), lambda s: s.list_sessions("proj"))
    assert [x["session_id"] for x in a] == [x["session_id"] for x in b] == ["s2"]
    assert run(ours.list_subkeys(MAIN)) == run(theirs.list_subkeys(MAIN)) == []


def test_appending_an_entry_twice_stores_it_once(store):
    run(store.append(MAIN, ENTRIES))
    run(store.append(MAIN, ENTRIES[:2]))
    run(store.append(MAIN, [ENTRIES[0], ENTRIES[0]]))
    assert run(store.load(MAIN)) == ENTRIES


def test_entries_without_a_uuid_are_always_appended(store):
    run(store.append(MAIN, ENTRIES[2:]))
    run(store.append(MAIN, ENTRIES[2:]))
    assert run(store.load(MAIN)) == ENTRIES[2:] * 2


def test_mtime_moves_forward_on_every_append(store):
    run(store.append(MAIN, ENTRIES[:1]))
    first = run(store.list_sessions("proj"))[0]["mtime"]
    run(store.append(MAIN, ENTRIES[1:2]))
    assert run(store.list_sessions("proj"))[0]["mtime"] > first


def test_a_later_store_object_sees_the_same_transcript(mem, store):
    run(store.append(MAIN, ENTRIES))
    again = CommonMemoSessionStore(mem, user_id="ada")
    assert run(again.load(MAIN)) == ENTRIES
    assert [x["session_id"] for x in run(again.list_sessions("proj"))] == ["s1"]


def test_two_objects_on_one_log_see_each_others_writes(mem):
    a, b = CommonMemoSessionStore(mem), CommonMemoSessionStore(mem)
    run(a.append(MAIN, ENTRIES[:2]))
    run(b.append(MAIN, ENTRIES[:3]))  # b skips what a stored
    assert run(a.load(MAIN)) == run(b.load(MAIN)) == ENTRIES


def test_delete_is_in_the_log_and_removes_nothing_from_it(mem, store):
    run(store.append(MAIN, ENTRIES))
    before = len(mem.store.events())
    run(store.delete(MAIN))
    assert len(mem.store.events()) == before + 1
    run(store.delete(MAIN))  # nothing left: not recorded again
    assert len(mem.store.events()) == before + 1
    run(store.append(MAIN, ENTRIES[:1]))
    assert run(store.load(MAIN)) == ENTRIES[:1]


def test_stores_with_different_ids_are_separate(mem):
    run(CommonMemoSessionStore(mem, session_id="a").append(MAIN, ENTRIES))
    assert run(CommonMemoSessionStore(mem, session_id="b").load(MAIN)) is None


def test_what_was_said_is_searchable_from_another_session(mem, store):
    run(store.append(MAIN, ENTRIES))
    other = mem.session("s2", user_id="ada")
    lines, _, _ = recall_entries(other, "which snacks is Maya allergic to?")
    assert any(label == "said" and "peanut" in text for label, text in lines)


def test_subagent_transcripts_are_not_recorded_as_conversation(mem, store):
    run(store.append(SUB, ENTRIES))
    other = mem.session("s2", user_id="ada")
    lines, _, _ = recall_entries(other, "peanut snacks")
    assert not any("peanut" in text for _, text in lines)
