"""The `Store` contract (#95): one suite, run against every store.

SQLite (and Postgres, when the suite runs in that mode: the SQL stores map to it) and `MemoryStore`,
which is plain Python. A new store, SQL or not, is correct when this passes for it: add it to
`STORES`.
"""

import time

import pytest

from commonmemo import Message, SQLiteStore, ToolInvoked
from commonmemo.events import WorldFact
from commonmemo.stores import MemoryStore, SessionRecord, store_from_url

STORES = {"sql": lambda: SQLiteStore(":memory:"), "memory": MemoryStore}


@pytest.fixture(params=sorted(STORES))
def store(request):
    s = STORES[request.param]()
    yield s
    s.close()


def msg(content="x", **kw):
    return Message(role="user", content=content, **kw)


# -- the Log ------------------------------------------------------------------


def test_append_is_idempotent_by_id(store):
    e = Message(role="user", content="hi", session_id="s1")
    assert store.append(e) is True
    assert store.append(e) is False  # a repeated id is a no-op, never an overwrite
    assert store.events() == [e]


def test_append_many_is_in_order_and_reports_each(store):
    a, b = msg("a"), msg("b")
    assert store.append_many([a, b, a]) == [True, True, False]
    assert [e.content for e in store.events()] == ["a", "b"]
    assert store.append_many([]) == []


def test_filters_and_limit_keep_oldest_first(store):
    events = [Message(role="user", content=str(i), session_id="s1", ts=float(i)) for i in range(5)]
    events.append(ToolInvoked(name="t", session_id="s2", user_id="u2", ts=10.0))
    for e in events:
        store.append(e)
    assert [e.content for e in store.events(session_id="s1", limit=2)] == ["3", "4"]
    assert store.events(kinds=["tool_invoked"]) == [events[-1]]
    assert store.events(user_id="u2") == [events[-1]]
    assert len(store.events(since=3.0)) == 3
    assert [e.content for e in store.events(kinds=["message"], since=2.0, limit=2)] == ["3", "4"]
    assert store.events(kinds=[]) == []


def test_run_id_filters(store):
    store.append(msg("a", run_id="r1"))
    store.append(msg("b", run_id="r2"))
    store.append(msg("c"))
    assert [e.content for e in store.events(run_id="r1")] == ["a"]


def test_get_events_returns_known_ids_oldest_first(store):
    a, b, c = msg("a"), msg("b"), msg("c")
    for e in (a, b, c):
        store.append(e)
    assert [e.content for e in store.get_events([c.id, "nope", a.id, a.id])] == ["a", "c"]
    assert store.get_events([]) == []


def test_events_are_copies_not_the_store_s_own(store):
    e = msg("original", session_id="s1")
    store.append(e)
    got = store.events()[0]
    assert got == e and got is not e


def test_positions_only_grow_and_events_after_reads_in_log_order(store):
    for i in range(5):
        store.append(msg(str(i), ts=float(i)))
    store.append(msg("2", ts=2.0))  # same content, new id
    after = store.events_after(0)
    seqs = [n for n, _ in after]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) == 6
    assert [e.content for _, e in store.events_after(seqs[2])] == ["3", "4", "2"]
    assert [n for n, _ in store.events_after(0, limit=2)] == seqs[:2]
    assert store.events_after(seqs[-1]) == []
    store.append(msg("later"))
    assert store.events_after(seqs[-1])[0][0] > seqs[-1]


def test_position_by_id_by_time_or_latest(store):
    events = [msg(str(i), ts=float(i * 10)) for i in range(3)]
    for e in events:
        store.append(e)
    seqs = [n for n, _ in store.events_after(0)]
    assert store.position() == seqs[-1]
    assert store.position(event_id=events[1].id) == seqs[1]
    assert store.position(event_id="nope") == 0
    assert store.position(at=15.0) == seqs[1] and store.position(at=-1.0) == 0
    assert SQLiteStore(":memory:").position() == 0


def test_event_counts_bucket_by_kind_source_and_scope(store):
    day = 86400.0
    store.append(WorldFact(key="a", value="1", scope="project:x", source="agent", ts=day * 5 + 1))
    store.append(WorldFact(key="b", value="2", scope="project:x", source="agent", ts=day * 5 + 2))
    store.append(WorldFact(key="c", value="3", scope="global", source="tool", ts=day * 5 + 3))
    store.append(msg("m", ts=day * 6 + 1))
    by = {(c.bucket, c.kind, c.source, c.scope): c.n for c in store.event_counts(bucket=day)}
    assert by[(day * 5, "world_fact", "agent", "project:x")] == 2 and by[(day * 5, "world_fact", "tool", "global")] == 1
    assert by[(day * 6, "message", "", "")] == 1  # a kind without a source or scope counts under ""
    assert sum(c.n for c in store.event_counts(bucket=day * 7)) == 4
    assert [c.n for c in store.event_counts(since=day * 6)] == [1]
    assert store.event_counts(until=day * 5) == []


def test_prune_keeps_daily_counts_of_what_it_deletes(store):
    day = 86400.0
    for i in range(3):
        store.append(msg(str(i), ts=day * 2 + i))
    store.append(msg("new", ts=day * 9))
    assert store.prune(older_than=day * 3) == 3
    assert [e.content for e in store.events()] == ["new"]
    assert sum(c.n for c in store.event_counts(bucket=day)) == 4  # the pruned still count, per day
    assert store.prune(older_than=day * 3) == 0


# -- claims -------------------------------------------------------------------


def test_a_claim_is_taken_once_and_the_event_goes_with_it(store):
    first = msg("claimed")
    assert store.claim("k", session_id="s1", owner="w1", lease_until=100.0, event=first) is None
    other = msg("never written")
    held = store.claim("k", session_id="s1", owner="w2", lease_until=200.0, event=other)
    assert (held.owner, held.status, held.lease_until) == ("w1", "held", 100.0)
    assert [e.content for e in store.events()] == ["claimed"]  # a lost race appends nothing
    assert store.get_claim("k") == held and store.get_claim("none") is None


def test_settle_frees_or_finishes_a_claim_and_appends_atomically(store):
    store.claim("k", session_id="s1", owner="w1", lease_until=100.0, event=msg("start"))
    store.settle("k", session_id="s1", status="free", event=msg("released"))
    assert store.get_claim("k").status == "free"
    won = store.claim("k", session_id="s1", owner="w2", lease_until=300.0, event=msg("again"))
    assert won is None and store.get_claim("k").owner == "w2"
    store.settle("k", session_id="s1", status="done", event=msg("finished"))
    assert store.get_claim("k").status == "done"
    assert store.claim("k", session_id="s1", owner="w3", lease_until=9.0, event=msg("no")).status == "done"
    store.settle("unclaimed", session_id=None, status="done", event=msg("created"))  # settling creates it
    assert store.get_claim("unclaimed").status == "done"
    assert [e.content for e in store.events()] == ["start", "released", "again", "finished", "created"]


# -- Working memory -----------------------------------------------------------


def test_open_resume_and_state(store):
    rec = store.open_session(SessionRecord("s1", "u1", "a", opened_at=1.0))
    store.save_state("s1", {"step": 2})
    store.close_session("s1", 5.0)
    assert store.get_session("s1").closed_at == 5.0
    resumed = store.open_session(SessionRecord("s1", "other", "b", opened_at=9.0))
    assert resumed.user_id == "u1" and resumed.agent == "a"  # the record that was there wins
    assert resumed.state == {"step": 2}
    assert resumed.is_open and store.get_session("s1").is_open
    assert rec.session_id == "s1"


def test_state_is_a_copy(store):
    store.open_session(SessionRecord("s1", None, None, opened_at=1.0))
    state = {"n": 1}
    store.save_state("s1", state)
    state["n"] = 99
    got = store.get_session("s1")
    assert got.state == {"n": 1}
    got.state["n"] = 7
    assert store.get_session("s1").state == {"n": 1}
    assert store.get_session("nope") is None


def test_expired_session_is_invisible_and_pruned(store):
    store.open_session(SessionRecord("old", None, None, opened_at=1.0, expires_at=time.time() - 1))
    assert store.get_session("old") is None
    assert store.prune(older_than=0) == 1
    store.open_session(SessionRecord("old", "u", None, opened_at=2.0))  # a new one under the same id
    assert store.get_session("old").user_id == "u"


def test_session_ids_belong_to_a_user(store):
    store.open_session(SessionRecord("s1", "u1", None, opened_at=1.0))
    store.open_session(SessionRecord("s2", "u1", None, opened_at=2.0))
    store.open_session(SessionRecord("s3", "u2", None, opened_at=3.0))
    assert sorted(store.session_ids("u1")) == ["s1", "s2"] and store.session_ids("nobody") == []


# -- lifecycle ----------------------------------------------------------------


def test_forget_user_removes_events_inside_their_sessions(store):
    store.open_session(SessionRecord("s1", "u1", None, opened_at=1.0))
    store.append(Message(role="user", content="mine", session_id="s1"))  # no user_id on event
    store.append(Message(role="user", content="also mine", user_id="u1"))
    store.append(Message(role="user", content="theirs", user_id="u2"))
    assert store.forget_user("u1") == 3  # two events and the session; the event inside it is the third... see below
    assert [e.content for e in store.events()] == ["theirs"]
    assert store.get_session("s1") is None and store.session_ids("u1") == []
    assert store.forget_user("u1") == 0


def test_forget_user_releases_their_claims(store):
    store.open_session(SessionRecord("s1", "u1", None, opened_at=1.0))
    store.claim("k", session_id="s1", owner="w", lease_until=1.0, event=msg("c", session_id="s1"))
    store.claim("other", session_id="s9", owner="w", lease_until=1.0, event=msg("o", session_id="s9"))
    store.forget_user("u1")
    assert store.get_claim("k") is None and store.get_claim("other") is not None


def test_store_from_url(tmp_path):
    assert store_from_url("sqlite://:memory:").path == ":memory:"
    assert store_from_url(f"sqlite:///{tmp_path}/m.db").path == f"{tmp_path}/m.db"
    assert isinstance(store_from_url("memory://"), MemoryStore)
    with pytest.raises(ValueError, match="no scheme"):
        store_from_url("mem.db")
    with pytest.raises(ValueError, match="unknown store scheme"):
        store_from_url("redis://x")
