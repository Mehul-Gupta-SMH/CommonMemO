"""Turn traces: what each push gave the agent, and why (#25)."""

from __future__ import annotations

import asyncio

from commonmemo import Memory, MemoryPolicy


def run(coro):
    return asyncio.run(coro)


def test_a_turn_trace_records_what_was_pushed_by_reference(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    mem = Memory(url)
    mem.learn("refund_window", "30 days")
    with mem.session(user_id="ada") as s:
        s.remember("preferred_contact", "email")
        s.never("call me")
    with mem.session("s2", user_id="ada", agent="support") as s:
        run(s.push_turn("how will you contact me about my refund?"))
    run(mem.aclose())

    mem = Memory(url)  # read back from the file: the trace survives the round trip
    [trace] = mem.traces(session_id="s2")
    assert (trace.reason, trace.user_id, trace.agent) == ("turn", "ada", "support")
    kinds = {(i["kind"], i.get("key")) for i in trace.items}
    assert {("fact", "preferred_contact"), ("world", "refund_window"), ("negative", None)} <= kinds
    assert trace.text is None  # references only, by default
    assert "preferred_contact" in "\n".join(trace.lines())
    run(mem.aclose())


def test_why_traces_an_item_back_to_its_source_events():
    mem = Memory()
    mem.learn("refund_window", "30 days")
    with mem.session("s1", user_id="ada") as s:
        s.remember("preferred_contact", "email")
    with mem.session("s2", user_id="ada") as s:
        run(s.push_turn("hello"))
    fact = mem.why("preferred_contact", session_id="s2")
    assert fact is not None and fact.item["kind"] == "fact"
    [source] = fact.sources
    assert (source.kind, source.session_id, source.value) == ("user_fact", "s1", "email")
    world = mem.why("refund_window", session_id="s2")
    assert world.item["source"] == "explicit" and world.sources[0].kind == "world_fact"
    assert "pushed in a turn push to session s2" in world.lines()[0]
    assert mem.why("never_pushed", session_id="s2") is None


def test_what_was_left_out_is_in_the_trace():
    mem = Memory()
    with mem.session(user_id="ada") as s:
        for i in range(23):
            s.remember(f"fact_{i:02d}", f"value {i}")
    with mem.session("s2", user_id="ada") as s:
        run(s.push_turn("hello"))
    [trace] = mem.traces(session_id="s2")
    assert trace.more == {"facts": 3}
    assert "left out: 3 facts" in "\n".join(trace.lines())


def test_full_text_only_when_opted_in_and_forget_user_removes_traces():
    mem = Memory(policy=MemoryPolicy(trace_text=True))
    with mem.session(user_id="ada") as s:
        s.remember("home_address", "12 Baker Street")
    with mem.session("s2", user_id="ada") as s:
        run(s.push("", reason="start", part="all"))
    [trace] = mem.traces(session_id="s2")
    assert "12 Baker Street" in trace.text and trace.chars == len(trace.text)
    run(mem.forget_user("ada"))
    assert mem.traces(user_id="ada") == [] and mem.traces(session_id="s2") == []


def test_refresh_traces_carry_recap_scores():
    mem = Memory()
    with mem.session(user_id="ada") as s:
        s.message("user", "the charger firmware rollback bricked unit 7; flash it over serial instead")
    with mem.session("s2", user_id="ada") as s:
        run(s.push_turn("Ship release 4.2"))
        run(s.refresh("deploy: charger firmware rollback started on unit 7"))
    turn, refresh = mem.traces(session_id="s2")
    assert refresh.reason == "refresh"
    [recap] = refresh.items
    assert recap["kind"] == "recap" and recap["score"] is not None and recap["events"]
