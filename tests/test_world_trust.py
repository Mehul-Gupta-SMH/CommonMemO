"""World facts reach every user, so who may change them matters (#24)."""

import asyncio

import pytest

from commonmemo import Memory
from commonmemo.agent_tools import build_agent_tools
from commonmemo.events import Feedback, WorldFact, describe


def note_fact(mem, session, key, value):
    tool = {t.__name__: t for t in build_agent_tools(session)}["memory_note_fact"]
    return tool(key, value)


def world_text(session, query="refund"):
    return asyncio.run(session.context(query)).stable_text()


def test_an_agent_fact_from_one_user_never_reaches_another_until_approved():
    mem = Memory(project="shop")
    mem.learn("refund_window", "30 days")
    with mem.session(user_id="mallory", agent="support") as s:
        reply = note_fact(mem, s, "refund_window", "90 days")
        assert reply.startswith("Proposed fact")
        mine = world_text(s)
        assert "90 days" in mine and "unconfirmed" in mine  # the proposer sees it, marked
    with mem.session(user_id="ada", agent="support") as s:
        theirs = world_text(s)
    assert "30 days" in theirs and "90 days" not in theirs  # everyone else does not
    assert {w.key: w.value for w in mem.world_facts()} == {"refund_window": "30 days"}

    (pending,) = mem.pending_world_facts()
    assert (pending.value, pending.user_id, pending.status) == ("90 days", "mallory", "proposed")


def test_confirming_makes_it_current_for_everyone_and_rejecting_does_not():
    mem = Memory(project="shop")
    with mem.session(user_id="u1") as s:
        note_fact(mem, s, "support_hours", "9-5")
        note_fact(mem, s, "refund_window", "90 days")
    hours, refund = mem.pending_world_facts()
    mem.confirm_fact(hours.event_id)
    mem.reject_fact(refund.event_id, reason="not our policy")
    assert {w.key: (w.value, w.source) for w in mem.world_facts()} == {"support_hours": ("9-5", "confirmed")}
    assert mem.pending_world_facts() == []
    with pytest.raises(ValueError):
        mem.confirm_fact(refund.event_id)  # already settled


def test_confirm_can_correct_the_value():
    mem = Memory(project="shop")
    with mem.session(user_id="u1") as s:
        note_fact(mem, s, "support_hours", "9 to 5ish")
    (p,) = mem.pending_world_facts()
    mem.confirm_fact(p.event_id, value="09:00-17:00 CET")
    assert mem.world_facts()[0].value == "09:00-17:00 CET"


def test_a_less_trusted_source_cannot_overwrite_a_more_trusted_fact():
    mem = Memory(project="shop", world_fact_approval=False)  # agent writes apply directly...
    mem.learn("refund_window", "30 days")  # ...but an operator's fact still outranks them
    with mem.session(user_id="u1") as s:
        assert note_fact(mem, s, "refund_window", "90 days").startswith("Saved fact")
        note_fact(mem, s, "gift_wrap", "free")
    assert {w.key: w.value for w in mem.world_facts()} == {"refund_window": "30 days", "gift_wrap": "free"}
    (disputed,) = mem.pending_world_facts()
    assert (disputed.value, disputed.status) == ("90 days", "disputed")
    mem.learn("refund_window", "45 days")  # an equal-ranked source supersedes as before
    assert {w.key: w.value for w in mem.world_facts()}["refund_window"] == "45 days"


def test_retraction_ends_a_fact_without_replacing_it():
    mem = Memory(project="shop")
    mem.learn("refund_window", "30 days")
    mem.retract_fact("refund_window", reason="refunds discontinued")
    assert mem.world_facts() == []
    assert [v for v, _, end in mem.knowledge.world_history("project:shop", "refund_window")] == ["30 days"]


def test_the_push_shows_each_facts_age_and_source():
    mem = Memory(project="shop")
    mem.learn("refund_window", "30 days")
    with mem.session(user_id="u1") as s:
        line = next(l for l in world_text(s).splitlines() if "refund_window" in l)
    assert "as of 20" in line and "operator" in line


def test_trust_survives_a_rebuild_and_forgetting_the_proposer(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    mem = Memory(url, project="shop")
    with mem.session(user_id="u1") as s:
        note_fact(mem, s, "support_hours", "9-5")
        note_fact(mem, s, "refund_window", "90 days")
    hours, _ = mem.pending_world_facts()
    mem.confirm_fact(hours.event_id)
    before = (mem.world_facts(), mem.pending_world_facts())
    mem.rebuild_knowledge()
    assert (mem.world_facts(), mem.pending_world_facts()) == before

    asyncio.run(mem.forget_user("u1"))
    mem.rebuild_knowledge()
    # The approval carries its own copy of the fact, so it outlives the
    # conversation that proposed it; the unapproved proposal goes with the user.
    assert {w.key: w.value for w in mem.world_facts()} == {"support_hours": "9-5"}
    assert mem.pending_world_facts() == []


def test_an_older_projection_file_rebuilds_itself(tmp_path):
    import sqlite3

    path = tmp_path / "m.db"
    url = f"sqlite:///{path}"
    mem = Memory(url, project="shop")
    mem.learn("refund_window", "30 days")
    with sqlite3.connect(path) as c:  # as if written by an older release
        c.execute("UPDATE knowledge_meta SET value = 1 WHERE key = 'schema'")
        c.execute("DROP TABLE world_facts")
        c.execute("CREATE TABLE world_facts (scope TEXT, key TEXT, value TEXT)")
    assert {w.key for w in Memory(url, project="shop").world_facts()} == {"refund_window"}


def test_proposals_and_review_actions_are_not_fed_to_engines_as_statements():
    proposal = WorldFact(key="refund_window", value="90 days", source="agent", proposed=True)
    assert describe(proposal) is None
    assert describe(WorldFact(key="refund_window", value="30 days")) == "Note: refund_window is 30 days."


def test_corrections_are_described_as_corrections():
    correction = Feedback(feedback_kind="correction", wrong="Pune", right="Bangalore", rating=-1.0)
    assert describe(correction) == "No, not 'Pune' but 'Bangalore'."
