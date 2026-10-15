import time

import pytest

from commonmemo import Message, SQLiteStore, ToolInvoked
from commonmemo.stores import SessionRecord, store_from_url


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


def test_append_is_idempotent_by_id(store):
    e = Message(role="user", content="hi", session_id="s1")
    store.append(e)
    store.append(e)
    assert store.events() == [e]


def test_filters_and_limit_keep_oldest_first(store):
    events = [Message(role="user", content=str(i), session_id="s1", ts=float(i)) for i in range(5)]
    events.append(ToolInvoked(name="t", session_id="s2", user_id="u2", ts=10.0))
    for e in events:
        store.append(e)
    assert [e.content for e in store.events(session_id="s1", limit=2)] == ["3", "4"]
    assert store.events(kinds=["tool_invoked"]) == [events[-1]]
    assert store.events(user_id="u2") == [events[-1]]
    assert len(store.events(since=3.0)) == 3


def test_open_resume_and_state(store):
    rec = store.open_session(SessionRecord("s1", "u1", "a", opened_at=1.0))
    store.save_state("s1", {"step": 2})
    store.close_session("s1", 5.0)
    resumed = store.open_session(SessionRecord("s1", "other", "b", opened_at=9.0))
    assert resumed.user_id == "u1" and resumed.agent == "a"
    assert resumed.state == {"step": 2}
    assert resumed.is_open
    assert rec.session_id == "s1"


def test_expired_session_is_invisible_and_pruned(store):
    store.open_session(SessionRecord("old", None, None, opened_at=1.0, expires_at=time.time() - 1))
    assert store.get_session("old") is None
    assert store.prune(older_than=0) == 1


def test_forget_user_removes_events_inside_their_sessions(store):
    store.open_session(SessionRecord("s1", "u1", None, opened_at=1.0))
    store.append(Message(role="user", content="mine", session_id="s1"))  # no user_id on event
    store.append(Message(role="user", content="also mine", user_id="u1"))
    store.append(Message(role="user", content="theirs", user_id="u2"))
    assert store.forget_user("u1") == 3
    assert [e.content for e in store.events()] == ["theirs"]


def test_store_from_url(tmp_path):
    assert store_from_url("sqlite://:memory:").path == ":memory:"
    assert store_from_url(f"sqlite:///{tmp_path}/m.db").path == f"{tmp_path}/m.db"
    with pytest.raises(ValueError, match="no scheme"):
        store_from_url("mem.db")
    with pytest.raises(ValueError, match="unknown store scheme"):
        store_from_url("redis://x")
