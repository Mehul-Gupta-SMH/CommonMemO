"""Recent facts rank above older ones, unless the fact is durable (#170)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time

from commonmemo import Memory, MemoryPolicy, ModelExtractor
from commonmemo.events import UserFact
from commonmemo.extraction import parse

DAY = 86400.0


def run(coro):
    return asyncio.run(coro)


def seed(mem, user, key, value, age_days, lifespan="state"):
    """A fact stated `age_days` ago, written the way the extractor writes it, then consolidated at close."""
    e = dataclasses.replace(UserFact(key=key, value=value, source="extracted", lifespan=lifespan),
                            ts=time.time() - age_days * DAY)
    with mem.session(f"s-{key}", user_id=user, agent="a") as s:
        s.record(e)


def picked(mem, user, query, limit=1):
    facts = mem._visible_facts(user, run(mem.engine.user_profile(user)))
    return [f.key for f in run(mem._pick_facts(query, facts, limit, user))]


def test_parse_reads_lifespan_and_drops_unknown_labels():
    text = json.dumps({"facts": [
        {"key": "name", "value": "Ada", "lifespan": "Durable"},
        {"key": "city", "value": "Lisbon", "lifespan": "state"},
        {"key": "trip", "value": "Rome in May", "lifespan": "someday"},
        {"key": "plain", "value": "no label"},
    ]})
    assert [f.lifespan for f in parse(text)] == ["durable", "state", "", ""]


def test_a_newer_state_fact_outranks_an_older_one_that_matches_equally():
    mem = Memory(policy=MemoryPolicy(recency_half_life="30d"))
    seed(mem, "ada", "city_before", "Ada lives in Berlin city", 400)
    seed(mem, "ada", "city_now", "Ada lives in Lisbon city", 2)
    assert picked(mem, "ada", "which city does Ada live in") == ["city_now"]


def test_off_by_default_so_order_is_unchanged():
    mem = Memory()
    seed(mem, "ada", "city_before", "Ada lives in Berlin city", 400)
    seed(mem, "ada", "city_now", "Ada lives in Lisbon city", 2)
    assert picked(mem, "ada", "which city does Ada live in") == ["city_before"]  # key order, a tie


def test_a_durable_fact_does_not_decay():
    mem = Memory(policy=MemoryPolicy(recency_half_life="30d"))
    seed(mem, "ada", "allergy", "Ada is allergic to penicillin medicine", 3000, "durable")
    seed(mem, "ada", "medicine_note", "Ada takes vitamin medicine today", 1, "state")
    # Equal match; the year-old durable fact must not lose to the fresh state fact because of age.
    assert picked(mem, "ada", "medicine") == ["allergy"]


def test_unlabelled_facts_do_not_decay():
    mem = Memory(policy=MemoryPolicy(recency_half_life="30d"))
    seed(mem, "ada", "a_old", "tea preference is green tea", 3000, "")
    seed(mem, "ada", "b_new", "tea preference is black tea", 1, "")
    assert picked(mem, "ada", "tea preference") == ["a_old"]


def test_an_old_state_fact_alone_is_still_found():
    mem = Memory(policy=MemoryPolicy(recency_half_life="30d"))
    seed(mem, "ada", "employer", "Ada works at Initech company", 2000)
    seed(mem, "ada", "pet", "A dog called Rex lives nearby", 1)
    assert picked(mem, "ada", "which company employs Ada") == ["employer"]


def test_events_age_faster_than_states():
    mem = Memory(policy=MemoryPolicy(recency_half_life="30d"))
    seed(mem, "ada", "e1", "Ada visited the museum", 40, "event")
    seed(mem, "ada", "s1", "Ada visited the museum", 40, "state")
    facts = mem._visible_facts("ada", run(mem.engine.user_profile("ada")))
    w = mem.fact_weight("ada", facts)
    by = {f.key: w(f) for f in facts}
    assert by["e1"] < by["s1"] < 1.0


def test_the_date_is_shown_when_recency_is_on():
    mem = Memory(policy=MemoryPolicy(recency_half_life="30d"))
    seed(mem, "ada", "city", "Ada lives in Lisbon", 2)
    with mem.session("t", user_id="ada") as s:
        ctx = run(s.context("where does Ada live"))
    assert "(stated " in ctx.render()
    off = Memory()
    seed(off, "ada", "city", "Ada lives in Lisbon", 2)
    with off.session("t", user_id="ada") as s:
        assert "(stated " not in run(s.context("where")).render()


def test_extracted_lifespan_is_written_and_old_events_replay():
    reply = json.dumps({"facts": [{"key": "home_city", "value": "Ada moved to Lisbon", "lifespan": "state"}]})
    mem = Memory(extractor=ModelExtractor(lambda p: reply))
    with mem.session("s1", user_id="ada", agent="a") as s:
        s.message("user", "I moved to Lisbon")
    (e,) = mem.store.events(user_id="ada", kinds=[UserFact.kind])
    assert e.lifespan == "state"
    # An event stored before the field existed has no lifespan and loads as "".
    payload = json.loads(e.model_dump_json() if hasattr(e, "model_dump_json") else json.dumps(e.payload))
    payload.pop("lifespan", None)
    assert UserFact(**{k: v for k, v in payload.items() if k in ("key", "value", "source")}).lifespan == ""


def test_the_flag_reaches_the_policy_and_the_hook_setup():
    import argparse

    from commonmemo.cli import policy_from
    from commonmemo.hooks import hook_config

    args = argparse.Namespace(store="sqlite:///.commonmemo/memory.db", engine="local", project=None, user="ada",
                              agent=None, session=None, read_only=False, recall_tool=False, no_environment=False,
                              messages=4, recall_messages=6, message_dates="off", recency_half_life="90d")
    assert policy_from(args).recency_half_life == "90d"
    assert "--recency-half-life 90d" in hook_config("claude-code", args)["hooks"]["SessionStart"][0]["hooks"][0]["command"]
