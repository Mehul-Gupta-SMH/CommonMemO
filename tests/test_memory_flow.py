"""End to end through all four tiers (Log, Working, Recaps, UserProfile) with the local engine."""

import asyncio

import pytest

from commonmemo import LocalEngine, Memory, MemoryEvent
from commonmemo.engines import engine_from_name


class TicketOpened(MemoryEvent):
    ticket: str


def run(coro):
    return asyncio.run(coro)


def test_four_tiers_end_to_end(tmp_path):
    mem = Memory(f"sqlite:///{tmp_path}/mem.db")

    @mem.track
    def lookup_order(order_id: str) -> str:
        return f"order {order_id}: shipped"

    async def first_session():
        async with mem.session(user_id="u1", agent="support") as s:
            s.message("user", "Where is my refund for order 42?")
            lookup_order("42")
            s.record(TicketOpened(ticket="T-9"))
            s.state["ticket"] = "T-9"
            s.remember("preferred_contact", "email")
            s.message("assistant", "Your refund for order 42 was issued today.")
        return s

    s1 = run(first_session())

    # Recaps: one recap, with provenance back to every event.
    (recap,) = s1.recaps
    assert "refund" in recap.summary and "lookup_order" in recap.summary
    assert set(recap.source_event_ids) == {e.id for e in mem.store.events(session_id=s1.session_id)}

    async def second_session():
        async with mem.session(user_id="u1", agent="support") as s:
            return await s.context("refund status")

    ctx = run(second_session())
    assert [(f.key, f.value) for f in ctx.facts] == [("preferred_contact", "email")]
    assert [e.session_id for e in ctx.recaps] == [s1.session_id]
    text = ctx.render()
    assert "preferred_contact: email" in text and "refund" in text
    run(mem.aclose())


def test_same_path_is_the_whole_setup(tmp_path):
    mem = Memory(f"sqlite:///{tmp_path}/mem.db")
    assert isinstance(mem.engine, LocalEngine) and mem.engine.path == mem.store.path
    run(mem.aclose())


def test_newer_fact_supersedes_and_history_is_kept():
    mem = Memory()

    async def main():
        for value in ["imperial", "metric"]:
            async with mem.session(user_id="u1") as s:
                s.remember("units", value)
        return await mem.engine.user_profile("u1"), await mem.engine.fact_history("u1", "units")

    facts, history = run(main())
    assert [(f.key, f.value) for f in facts] == [("units", "metric")]
    assert [h[0] for h in history] == ["imperial", "metric"]
    assert history[0][2] is not None and history[1][2] is None


def test_session_state_survives_resume_across_handles(tmp_path):
    url = f"sqlite:///{tmp_path}/mem.db"
    a = Memory(url)
    with a.session("shared", user_id="u1", agent="claude-code") as s:
        s.state["plan"] = ["read", "edit"]
        s.message("user", "start")
    run(a.aclose())

    b = Memory(url)  # another process / harness, same store
    with b.session("shared", user_id="u1", agent="codex") as s:
        assert s.state["plan"] == ["read", "edit"]
        assert s.agent == "codex"  # who is working now...
        assert b.store.get_session("shared").agent == "claude-code"  # ...but the owner record is not rewritten
        s.message("user", "continue")
    assert [e.agent for e in b.store.events(session_id="shared", kinds=["message"])] == ["claude-code", "codex"]
    # Re-consolidation on the second close replaces the recap, never duplicates it.
    eps = run(b.engine.search_recaps("", user_id="u1"))
    assert len(eps) == 1 and len(eps[0].source_event_ids) == 2
    run(b.aclose())


def test_resuming_another_users_session_is_refused():
    mem = Memory()
    with mem.session("s1", user_id="u1"):
        pass
    with pytest.raises(PermissionError):
        mem.session("s1", user_id="u2")


def test_closed_session_rejects_writes_and_remember_needs_a_user():
    mem = Memory()
    s = mem.session()
    with pytest.raises(ValueError, match="needs a user"):
        s.remember("k", "v")
    run(s.aclose())
    with pytest.raises(RuntimeError, match="is closed"):
        s.message("user", "late")


def test_forget_user_clears_every_tier():
    mem = Memory()
    with mem.session(user_id="u1") as s:
        s.message("user", "hello")
        s.remember("name", "Ada")
    with mem.session(user_id="u2") as s:
        s.message("user", "other")

    assert run(mem.forget_user("u1")) > 0
    assert mem.store.events(user_id="u1") == []
    assert run(mem.engine.user_profile("u1")) == []
    assert run(mem.engine.search_recaps("", user_id="u1")) == []
    assert len(run(mem.engine.search_recaps("", user_id="u2"))) == 1


def test_plain_with_inside_a_running_loop_is_a_clear_error():
    mem = Memory()

    async def main():
        with mem.session():
            pass

    with pytest.raises(RuntimeError, match="async with"):
        run(main())


def test_empty_session_produces_no_recap():
    mem = Memory()
    with mem.session(user_id="u1") as s:
        pass
    assert s.recaps == []


def test_engine_registry_errors_are_distinct(monkeypatch):
    from commonmemo import engines

    monkeypatch.setitem(engines._PLANNED, "future", "a future engine")
    with pytest.raises(NotImplementedError, match="planned"):
        engine_from_name("future")
    with pytest.raises(ValueError, match="unknown engine"):
        engine_from_name("nope")
