"""World facts: the key registry, lifetimes, pinning and bulk import (#24)."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from commonmemo import Memory
from commonmemo.agent_tools import build_agent_tools
from commonmemo.facts import FactKey, parse_duration
from commonmemo.mcp import tool_schema

KEYS = [
    FactKey("refund_window", "How long customers have to return an order", lifetime="90d", pinned=True),
    FactKey("plan_tier", "The customer's plan", allowed=("free", "pro")),
    FactKey("support_hours", "When the support desk answers"),
]


def tools(s):
    return {t.__name__: t for t in build_agent_tools(s)}


def test_durations():
    assert parse_duration("90d") == 90 * 86400 and parse_duration("12h") == 43200 and parse_duration(30) == 30.0
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_agents_are_told_the_registered_keys():
    mem = Memory(fact_keys=KEYS)
    with mem.session(user_id="ada") as s:
        note = tools(s)["memory_note_fact"]
    assert "refund_window: How long customers have to return an order" in note.__doc__
    assert "plan_tier: The customer's plan (one of: free, pro)" in note.__doc__
    schema = tool_schema(note)  # what an MCP client sees
    assert "refund_window" in schema["description"] and schema["inputSchema"]["required"] == ["key", "value"]
    with Memory().session(user_id="bob") as s:  # no registry: the description is unchanged
        assert "Use one of these keys" not in tools(s)["memory_note_fact"].__doc__


def test_unregistered_near_duplicate_and_disallowed_writes_stay_proposals_even_without_approval():
    mem = Memory(fact_keys=KEYS, world_fact_approval=False)
    with mem.session(user_id="ada") as s:
        note = tools(s)["memory_note_fact"]
        assert note("refund_window", "30 days").startswith("Saved fact")  # registered: written directly
        reply = note("refund_period", "60 days")
        assert "Proposed" in reply and "did you mean 'refund_window'" in reply
        assert "not an allowed value" in note("plan_tier", "enterprise")
        assert "unregistered key" in note("favourite_colour", "blue")
    assert {w.key: w.value for w in mem.world_facts()} == {"refund_window": "30 days"}
    notes = {w.key: w.note for w in mem.pending_world_facts()}
    assert "did you mean 'refund_window'" in notes["refund_period"] and "enterprise" in notes["plan_tier"]


def test_operators_may_use_any_key_but_not_a_disallowed_value():
    mem = Memory(fact_keys=KEYS)
    mem.learn("office_city", "Pune")  # unregistered keys are fine for operators
    with pytest.raises(ValueError, match="not an allowed value"):
        mem.learn("plan_tier", "enterprise")
    assert {w.key for w in mem.world_facts()} == {"office_city"}


def test_registered_lifetime_applies_when_the_write_gives_none():
    mem = Memory(fact_keys=KEYS)
    mem.learn("refund_window", "30 days")
    mem.learn("support_hours", "9-5", ttl=60)
    facts = {w.key: w for w in mem.world_facts()}
    assert facts["refund_window"].valid_until == pytest.approx(time.time() + 90 * 86400, abs=5)
    assert facts["support_hours"].valid_until == pytest.approx(time.time() + 60, abs=5)


def test_pinned_facts_are_stable_and_the_rest_ranked_by_the_request():
    mem = Memory(fact_keys=KEYS)
    mem.learn("refund_window", "30 days")
    mem.learn("support_hours", "9am to 5pm IST")
    mem.learn("office_city", "Pune")
    with mem.session(user_id="ada") as s:
        build_agent_tools(s)
        ctx = asyncio.run(s.context("when does the support desk answer?"))
        assert [w.key for w in ctx.world] == ["refund_window"]  # pinned: stable half
        assert [w.key for w in ctx.world_turn] == ["support_hours"]  # matched via its description
        assert "Facts for this request" in ctx.turn_text() and "support_hours" not in ctx.stable_text()
        assert "1 known fact" in ctx.more_line()  # office_city: left to recall
        assert "Pune" in tools(s)["memory_recall"]("office city")


def test_without_a_registry_nothing_changes():
    mem = Memory()
    mem.learn("refund_window", "30 days")
    mem.learn("office_city", "Pune")
    with mem.session(user_id="ada") as s:
        ctx = asyncio.run(s.context("anything"))
    assert [w.key for w in ctx.world] == ["office_city", "refund_window"] and ctx.world_turn == []


def test_bulk_import_is_idempotent(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({
        "scope": "project:shop",
        "keys": [{"key": "refund_window", "description": "Return period", "lifetime": "90d", "pinned": True}],
        "facts": {"refund_window": "30 days", "support_hours": "9-5"},
    }))
    mem = Memory(project="shop")
    first = mem.load_facts(path)
    assert (first.keys, first.written, first.unchanged) == (1, 2, 0)
    again = mem.load_facts(path)
    assert (again.written, again.unchanged) == (0, 2)  # no churn on every start
    facts = {w.key: w for w in mem.world_facts()}
    assert facts["refund_window"].source == "import" and facts["refund_window"].scope == "project:shop"
    assert mem.fact_keys.pinned("refund_window")
    assert len(mem.store.events(kinds=["world_fact"])) == 2  # the second load wrote nothing


def test_bulk_import_from_yaml_and_bad_input(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "facts.yaml"
    path.write_text(yaml.safe_dump({"facts": [{"key": "price_basic", "value": "$10", "ttl": "1h"}]}))
    mem = Memory()
    assert mem.load_facts(path).written == 1
    assert mem.world_facts()[0].valid_until == pytest.approx(time.time() + 3600, abs=5)
    with pytest.raises(ValueError, match="unknown fields"):
        mem.load_facts({"keys": [{"key": "x", "pinnned": True}]})
    with pytest.raises(ValueError, match="unknown sections"):
        mem.load_facts({"fact": {}})


def test_registry_survives_rebuild_and_pending_notes_are_in_the_log():
    mem = Memory(fact_keys=KEYS, world_fact_approval=False)
    with mem.session(user_id="ada") as s:
        tools(s)["memory_note_fact"]("refund_period", "60 days")
    mem.knowledge.rebuild(mem.store)  # the decision and its note are in the event, not the registry
    [pending] = mem.pending_world_facts()
    assert pending.key == "refund_period" and "refund_window" in pending.note
