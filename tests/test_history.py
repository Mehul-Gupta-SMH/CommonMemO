"""Memory history (#27 phase 1): run ids, metrics, snapshots, as_of, diff."""

import asyncio
import sqlite3
import time

import pytest

from commonmemo import Memory
from commonmemo.agent_tools import build_agent_tools
from commonmemo.events import Message, ToolInvoked, UserFact, WorldFact
from commonmemo.stores import SQLiteStore

DAY = 86400.0


# -- run_id --------------------------------------------------------------------


def test_events_written_inside_a_run_carry_its_id():
    mem = Memory(project="shop")
    with mem.run("deploy-42") as run:
        mem.learn("refund_window", "30 days")
        with mem.session(user_id="u1") as s:
            s.message("user", "hi")
    with mem.session(user_id="u1", run_id="import-7") as s:
        s.remember("tz", "CET")
        with mem.run("ignored"):  # the session's own run id wins
            s.message("user", "again")
    s2 = mem.session(user_id="u1")
    s2.record(Message(role="user", content="tagged", run_id="mine"))  # an explicit id is kept

    assert run.run_id == "deploy-42"
    assert mem.history().by_run("deploy-42") == {"world_fact": 1, "message": 1}
    assert mem.history().by_run("import-7") == {"user_fact": 1, "message": 1}
    assert mem.history().by_run("mine") == {"message": 1}
    assert mem.history().by_run("ignored") == {}


def test_a_run_without_an_id_gets_one_and_async_works():
    mem = Memory()

    async def main():
        async with mem.run() as run:
            mem.learn("k", "v")
        return run.run_id

    run_id = asyncio.run(main())
    assert run_id and mem.history().by_run(run_id) == {"world_fact": 1}


@pytest.mark.sqlite_only  # edits the SQLite file directly
def test_a_store_file_from_before_run_id_gets_the_column(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, "
                  "ts REAL NOT NULL, kind TEXT NOT NULL, session_id TEXT, user_id TEXT, agent TEXT, body TEXT NOT NULL)")
        c.execute("INSERT INTO events (id, ts, kind, body) VALUES ('e0', 1.0, 'message', "
                  "'{\"kind\": \"message\", \"id\": \"e0\", \"ts\": 1.0, \"role\": \"user\", \"content\": \"old\"}')")
    store = SQLiteStore(path)
    store.append(Message(role="user", content="new", run_id="r"))
    assert [e.content for e in store.events()] == ["old", "new"]
    assert [e.content for e in store.events(run_id="r")] == ["new"]


# -- metrics -------------------------------------------------------------------


def test_velocity_counts_by_kind_source_and_scope_per_bucket():
    mem = Memory(project="shop")
    t0 = 1_700_000_000.0
    mem.record(WorldFact(key="a", value="1", scope="project:shop", ts=t0))
    mem.record(WorldFact(key="b", value="1", scope="project:shop", ts=t0 + 10))
    mem.record(WorldFact(key="c", value="1", scope="global", ts=t0 + 2 * 3600))
    mem.record(Message(role="user", content="x", ts=t0 + 20))
    rows = mem.history().velocity(since=t0 - 1, bucket=3600, kinds=["world_fact"])
    assert [(r.scope, r.source, r.n) for r in rows] == [("project:shop", "explicit", 2), ("global", "explicit", 1)]
    assert rows[1].bucket - rows[0].bucket == 2 * 3600


def test_pruned_events_still_count_from_the_daily_rollups(tmp_path):
    mem = Memory(f"sqlite:///{tmp_path}/m.db")
    old = time.time() - 40 * DAY
    for i in range(3):
        mem.record(UserFact(key=f"k{i}", value="v", user_id="u1", ts=old + i))
    mem.record(UserFact(key="new", value="v", user_id="u1"))
    before = sum(r.n for r in mem.history().velocity(kinds=["user_fact"]))
    assert mem.prune(older_than_days=30) == 3
    assert len(mem.store.events(kinds=["user_fact"])) == 1
    assert sum(r.n for r in mem.history().velocity(kinds=["user_fact"])) == before == 4
    mem.prune(older_than_days=30)  # pruning again adds nothing twice
    assert sum(r.n for r in mem.history().velocity(kinds=["user_fact"])) == 4


def test_source_mix():
    mem = Memory(project="shop")
    mem.learn("a", "1")
    with mem.session(user_id="u1") as s:
        tools = {t.__name__: t for t in build_agent_tools(s)}
        tools["memory_note_fact"]("b", "2")
        tools["memory_note_fact"]("c", "3")
        s.remember("tz", "CET")
    mix = mem.history().source_mix()
    assert mix == {"agent": 0.5, "explicit": 0.5}


def test_churn_finds_keys_that_keep_changing_and_flip_flops():
    mem = Memory(project="shop")
    with mem.session(user_id="u1") as s:
        for v in ("email", "phone", "email", "sms"):
            s.remember("contact", v)
        s.remember("tz", "CET")
    mem.learn("hours", "9-5")
    mem.learn("hours", "8-4")
    with mem.session(user_id="u2") as s:  # proposals change nothing, so no churn
        tool = {t.__name__: t for t in build_agent_tools(s)}["memory_note_fact"]
        for v in ("1", "2", "3"):
            tool("hours", v)
    churn = mem.history().churn(min_changes=1)
    assert [(c.kind, c.key, c.changes, c.flip_flops) for c in churn] == [
        ("user_fact", "u1/contact", 3, 1),
        ("world_fact", "project:shop/hours", 1, 0),
    ]


def test_review_flow():
    mem = Memory(project="shop")
    with mem.session(user_id="u1") as s:
        tool = {t.__name__: t for t in build_agent_tools(s)}["memory_note_fact"]
        for k in ("a", "b", "c"):
            tool(k, "v")
    a, b, _ = mem.pending_world_facts()
    mem.confirm_fact(a.event_id)
    mem.reject_fact(b.event_id)
    flow = mem.history().review_flow()
    assert (flow.proposed, flow.confirmed, flow.rejected, flow.pending_now) == (3, 1, 1, 1)
    assert flow.rejection_rate == 0.5 and flow.median_review_seconds is not None


def test_staleness_and_growth():
    mem = Memory(project="shop")
    now = time.time()
    mem.record(WorldFact(key="old", value="v", scope="project:shop", ts=now - 100 * DAY))
    mem.record(WorldFact(key="week", value="v", scope="project:shop", ts=now - 3 * DAY))
    mem.record(WorldFact(key="price", value="9", scope="project:shop", ts=now - 2 * DAY, valid_until=now - DAY))
    mem.learn("fresh", "v")
    with mem.session(user_id="u1", agent="a") as s:
        s.remember("tz", "CET")
        s.remember("tz", "PST")
        s.remember("contact", "email")
        s.record(ToolInvoked(name="search", arguments={"q": "x"}, error="timeout after 30s"))
        s.record(ToolInvoked(name="search", arguments={"q": "x"}, error="order 4411 not found"))
    st = mem.history().staleness(now=now)
    assert st.world_facts == 4 and st.expired == 1
    assert st.ages == {"<1d": 1, "1-7d": 2, "7-30d": 0, "30-90d": 0, ">90d": 1}
    g = mem.history().growth()
    assert g.world_facts == 4 and g.profile_keys == {"u1": 2} and g.error_signatures == {"search": 2}


# -- snapshots, as_of, diff ----------------------------------------------------


def test_as_of_a_snapshot_a_position_or_a_time_and_live_memory_is_untouched():
    mem = Memory(project="shop")
    mem.learn("refund_window", "14 days")
    seq_after_first = mem.store.position()
    mem.snapshot("before v3")
    t_between = time.time()
    time.sleep(0.01)
    mem.learn("refund_window", "30 days")

    for point in ("before v3", seq_after_first, t_between):
        view = mem.as_of(point)
        assert {w.key: w.value for w in view.world_facts()} == {"refund_window": "14 days"}
    assert {w.key: w.value for w in mem.world_facts()} == {"refund_window": "30 days"}
    assert {w.key: w.value for w in mem.as_of(None).world_facts()} == {"refund_window": "30 days"}
    with pytest.raises(KeyError):
        mem.as_of("no such snapshot")


def test_diff_lists_what_changed_since_a_snapshot():
    mem = Memory(project="shop")
    mem.learn("refund_window", "14 days")
    mem.learn("support_hours", "9-5")
    mem.snapshot("s1")
    mem.learn("refund_window", "30 days")
    mem.retract_fact("support_hours")
    mem.learn("gift_wrap", "free")
    mem.never("suggest phone calls", scope="project:shop")
    with mem.session(user_id="u1") as s:
        s.record(ToolInvoked(name="search", arguments={}))
    d = mem.diff("s1")
    got = {(c.kind, c.key.split("#")[0], c.change) for c in d.changes}
    assert ("world_fact", "project:shop/refund_window", "changed") in got
    assert ("world_fact", "project:shop/support_hours", "removed") in got
    assert ("world_fact", "project:shop/gift_wrap", "added") in got
    assert ("tool", "search", "added") in got
    assert any(k == "negative" and ch == "added" for k, _, ch in got)
    assert d.summary()["world_fact"] == {"added": 1, "changed": 1, "removed": 1}
    assert not mem.diff("s1", "s1")


def test_forgotten_data_is_gone_from_every_past_view():
    mem = Memory(project="shop")
    with mem.session(user_id="u1", agent="a") as s:
        s.feedback(-1, "too long")
        s.never("call me")
    mem.snapshot("with u1")
    assert mem.as_of("with u1").feedback_summary().count == 1
    asyncio.run(mem.forget_user("u1"))
    view = mem.as_of("with u1")
    assert view.feedback_summary().count == 0
    assert view.negatives(["user:u1"]) == []
