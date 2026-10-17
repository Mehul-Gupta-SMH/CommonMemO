"""Reins (#27 phase 2): revert by time and by cause, freeze, budgets, alerts."""

import asyncio
import logging
import time

import pytest

from commonmemo import AlertPolicy, Budget, Memory
from commonmemo.agent_tools import build_agent_tools
from commonmemo.events import WorldFact


def world(mem):
    return {w.key: w.value for w in mem.world_facts()}


def negatives(mem, scope="project:shop"):
    return sorted(n.statement for n in mem.negatives([scope]))


def note(session, key, value):
    return {t.__name__: t for t in build_agent_tools(session)}["memory_note_fact"](key, value)


# -- revert by time ------------------------------------------------------------


def test_revert_to_a_snapshot_restores_changed_added_and_removed_facts_and_negatives():
    mem = Memory(project="shop")
    mem.learn("refund_window", "14 days")
    mem.learn("support_hours", "9-5")
    mem.never("promise delivery dates", scope="project:shop")
    mem.snapshot("good")
    mem.learn("refund_window", "90 days")          # changed
    mem.retract_fact("support_hours")              # removed
    mem.learn("gift_wrap", "free")                 # added
    mem.never("mention competitors", scope="project:shop")
    mem.revoke_negative(next(n.event_id for n in mem.negatives(["project:shop"])
                             if n.statement == "promise delivery dates"))

    result = mem.revert(to="good", reason="bad deploy")
    assert world(mem) == {"refund_window": "14 days", "support_hours": "9-5"}
    assert negatives(mem) == ["promise delivery dates"]
    assert result.summary() == {"world_fact": 2, "world_fact_retracted": 1, "negative": 1, "negative_revoked": 1}
    assert all(e.run_id == result.run_id for e in result.events)
    assert mem.history().by_run(result.run_id)["reverted"] == 1
    assert not mem.diff("good")  # memory matches the snapshot again


def test_a_dry_run_plans_without_writing():
    mem = Memory(project="shop")
    mem.learn("a", "1")
    mem.snapshot("s")
    mem.learn("a", "2")
    before = mem.store.position()
    plan = mem.revert(to="s", dry_run=True)
    assert plan.dry_run and [e.value for e in plan.events] == ["1"]
    assert mem.store.position() == before and world(mem) == {"a": "2"}


def test_scope_limits_a_revert():
    mem = Memory(project="shop")
    mem.learn("a", "1")
    mem.learn("b", "1", scope="global")
    mem.snapshot("s")
    mem.learn("a", "2")
    mem.learn("b", "2", scope="global")
    mem.revert(to="s", scope="global")
    assert world(mem) == {"a": "2", "b": "1"}


# -- revert by cause -----------------------------------------------------------


def test_revert_a_run_keeps_what_others_changed_later():
    mem = Memory(project="shop", world_fact_approval=False)
    mem.learn("refund_window", "30 days")
    with mem.run("bad-run"), mem.session(user_id="u1") as s:
        note(s, "shipping_fee", "free")
        note(s, "warranty", "5 years")
    mem.learn("support_hours", "9-6")          # someone else, afterwards
    mem.learn("warranty", "2 years")           # an operator corrects one of them later

    mem.revert(run_id="bad-run")
    # The run's own write that nobody touched since is gone; the operator's later
    # correction of the other one survives, and so does the unrelated change.
    assert world(mem) == {"refund_window": "30 days", "support_hours": "9-6", "warranty": "2 years"}


def test_revert_a_run_rejects_the_proposals_it_left_waiting():
    mem = Memory(project="shop")
    with mem.run("spam"), mem.session(user_id="mallory") as s:
        for i in range(5):
            note(s, f"k{i}", "v")
    with mem.session(user_id="ada") as s:
        note(s, "warranty", "2 years")
    assert len(mem.pending_world_facts()) == 6
    mem.revert(run_id="spam")
    assert [w.key for w in mem.pending_world_facts()] == ["warranty"]


def test_revert_by_source():
    mem = Memory(project="shop", world_fact_approval=False)
    mem.learn("a", "operator")
    with mem.session(user_id="u1") as s:
        note(s, "b", "agent")
    mem.revert(source="agent")
    assert world(mem) == {"a": "operator"}


def test_a_revert_can_be_reverted_and_rebuilds_agree(tmp_path):
    mem = Memory(f"sqlite:///{tmp_path}/m.db", project="shop")
    mem.learn("a", "1")
    mem.snapshot("s")
    mem.learn("a", "2")
    first = mem.revert(to="s")
    assert world(mem) == {"a": "1"}
    mem.revert(run_id=first.run_id)
    assert world(mem) == {"a": "2"}
    live = world(mem)
    mem.rebuild_knowledge()
    assert world(mem) == live


def test_a_revert_never_brings_back_forgotten_data():
    mem = Memory(project="shop")
    with mem.session(user_id="u1") as s:
        s.never("call me")
    mem.snapshot("with u1")
    asyncio.run(mem.forget_user("u1"))
    mem.revert(to="with u1")
    assert mem.negatives(["user:u1"]) == []


def test_bad_revert_requests_fail_loudly():
    mem = Memory()
    with pytest.raises(ValueError):
        mem.revert()  # neither a point nor a cause
    with pytest.raises(ValueError):
        mem.revert(to=0, run_id="r")  # both
    with pytest.raises(ValueError):
        mem.revert(to=0, kinds=["tool"])  # not revertible
    with pytest.raises(KeyError):
        mem.revert(to="no such snapshot")


# -- freeze and budgets --------------------------------------------------------


def test_freeze_turns_writes_into_proposals_until_unfrozen():
    mem = Memory(project="shop", world_fact_approval=False)
    freeze = mem.freeze(scope="project:shop", reason="incident")
    mem.learn("a", "1")
    with mem.session(user_id="u1") as s:
        note(s, "b", "2")
    mem.learn("g", "1", scope="global")  # another scope is not frozen
    assert world(mem) == {"g": "1"}
    assert sorted(w.key for w in mem.pending_world_facts()) == ["a", "b"]
    assert [f.reason for f in mem.freezes()] == ["incident"]
    mem.unfreeze(freeze.id)
    mem.learn("c", "3")
    assert "c" in world(mem) and mem.freezes() == []


def test_a_source_freeze_only_holds_that_source():
    mem = Memory(project="shop", world_fact_approval=False)
    mem.freeze(source="agent")
    mem.learn("op", "1")
    with mem.session(user_id="u1") as s:
        note(s, "ag", "1")
    assert world(mem) == {"op": "1"} and [w.key for w in mem.pending_world_facts()] == ["ag"]


def test_budget_turns_writes_past_the_cap_into_proposals():
    mem = Memory(project="shop", world_fact_approval=False, budgets={"agent": Budget(writes=2, per=3600)})
    with mem.session(user_id="u1") as s:
        for i in range(4):
            note(s, f"k{i}", "v")
    assert sorted(world(mem)) == ["k0", "k1"]
    assert sorted(w.key for w in mem.pending_world_facts()) == ["k2", "k3"]
    assert Memory(budgets={"agent": 5}).budgets["agent"] == Budget(writes=5, per=3600.0)


# -- alerts --------------------------------------------------------------------


def _spike(mem, n, source="agent", ts=None):
    for i in range(n):
        mem.record(WorldFact(key=f"k{i}", value="v", scope="project:shop", source=source,
                             proposed=True, ts=ts if ts is not None else time.time()))


def test_velocity_spike_alerts_are_logged_and_sent_by_default(caplog):
    seen = []
    mem = Memory(project="shop", alerts=AlertPolicy(on_alert=seen.append, velocity_min=20, check_every=0))
    week_ago = time.time() - 7 * 86400
    _spike(mem, 3, ts=week_ago)                        # a quiet baseline
    with caplog.at_level(logging.WARNING, logger="commonmemo.alerts"):
        _spike(mem, 25)
    assert any(a.kind == "velocity" and a.subject == "agent" for a in seen)
    assert "velocity" in caplog.text
    assert mem.freezes() == []                         # auto-freeze is opt-in


def test_auto_freeze_is_opt_in():
    mem = Memory(project="shop", alerts=AlertPolicy(auto_freeze=True, velocity_min=20, check_every=0))
    _spike(mem, 25)
    assert [(f.scope, f.source) for f in mem.freezes()] == [(None, "agent")]


def test_alerts_can_be_turned_off_and_checked_by_hand():
    seen = []
    mem = Memory(project="shop", alerts=AlertPolicy(enabled=False, on_alert=seen.append, velocity_min=20))
    _spike(mem, 25)
    assert seen == []
    assert [a.kind for a in mem.alerts()] == ["velocity"]


def test_churn_and_backlog_alerts():
    mem = Memory(project="shop", alerts=AlertPolicy(enabled=False, backlog=3, flip_flops=2))
    with mem.session(user_id="u1") as s:
        for v in ("a", "b", "a", "b", "a"):
            s.remember("contact", v)
        for i in range(4):
            note(s, f"k{i}", "v")
    kinds = {a.kind for a in mem.alerts()}
    assert kinds == {"churn", "review_backlog"}
