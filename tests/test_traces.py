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
        run(s.push("how will you contact me about my refund?", part="all"))  # both halves delivered
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
        run(s.push("hello", part="all"))
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


def test_a_trace_lists_only_what_its_push_delivered():
    """#49: a turn push delivers the turn half; its trace must not claim the
    stable half's facts and constraints too (found by the live run, #48)."""
    from commonmemo.hooks import handle

    mem = Memory()
    with mem.session("seed", user_id="mal") as s:
        s.remember("note", "hello")
    handle({"session_id": "y", "hook_event_name": "SessionStart"}, mem, harness="claude-code", user="mal")
    out = handle({"session_id": "y", "hook_event_name": "UserPromptSubmit", "prompt": "Give me a commit tip"},
                 mem, harness="claude-code", user="mal")
    start, turn = mem.traces(session_id="claude-code-y")
    assert ("fact", "note") in {(i["kind"], i.get("key")) for i in start.items}  # the start push delivered it
    assert out is None and turn.chars == 0 and turn.items == ()                   # the turn push delivered nothing
    with mem.session("s", user_id="mal") as s:
        stable = run(s.push("x", reason="start", part="stable"))
    [trace] = mem.traces(session_id="s")
    assert stable.facts and all(i["kind"] in ("fact", "negative", "world", "lesson") for i in trace.items)


def test_a_recall_is_traced_with_what_it_returned():
    """#50: what the agent pulled is in the Log, like what was pushed."""
    from commonmemo.agent_tools import build_agent_tools

    mem = Memory()
    with mem.session("s", user_id="ada") as s:
        s.remember("parking_spot", "level 3, bay 17")
        recall = {t.__name__: t for t in build_agent_tools(s)}["memory_recall"]
        assert "level 3, bay 17" in recall("parking spot")
        read_only = {t.__name__: t for t in build_agent_tools(s, writes=False)}["memory_recall"]
        read_only("parking spot")
    [trace] = mem.traces(session_id="s")  # one: the read-only recall recorded nothing
    assert trace.reason == "recall" and trace.query is None  # the query is kept only with trace_text
    assert [(i["kind"], i.get("key")) for i in trace.items] == [("fact", "parking_spot")]
    assert mem.why("parking_spot", session_id="s").trace.reason == "recall"

    mem = Memory(policy=MemoryPolicy(trace_text=True))
    with mem.session("s", user_id="ada") as s:
        {t.__name__: t for t in build_agent_tools(s)}["memory_recall"]("anything")
    [trace] = mem.traces(session_id="s")
    assert trace.query == "anything" and trace.text == "Nothing in memory matches that."
