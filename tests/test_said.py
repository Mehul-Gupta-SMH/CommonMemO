"""Windows of a user's past messages in the push and in recall (#62)."""

from __future__ import annotations

import argparse
import asyncio

from commonmemo import Memory, MemoryPolicy
from commonmemo.agent_tools import build_agent_tools
from commonmemo.hooks import handle, hook_config
from commonmemo.said import WINDOW


def run(coro):
    return asyncio.run(coro)


def talk(mem, sid, lines, user="ada", agent="chat"):
    with mem.session(sid, user_id=user, agent=agent) as s:
        for role, text in lines:
            s.message(role, text)


def mem_with(**policy):
    mem = Memory(policy=MemoryPolicy(**policy))
    talk(mem, "s1", [("user", "My sister Lena arrives on Friday 14 November, I need the guest room ready."),
                     ("assistant", "Noted. Anything else?"),
                     ("user", "The guest room key is under the blue pot by the door.")])
    return mem


def turn(mem, query, user="ada", sid="s2", **kw):
    with mem.session(sid, user_id=user, agent=kw.pop("agent", "chat")) as s:
        return run(s.context(query)), s


def test_off_by_default():
    mem = Memory()
    talk(mem, "s1", [("user", "Hello."), ("user", "The guest room key is under the blue pot."),
                     ("assistant", "Noted.")])  # past the first message, so the recap digest doesn't hold it
    ctx, _ = turn(mem, "where is the guest room key?")
    assert ctx.said == [] and "blue pot" not in ctx.turn_text()


def test_the_turn_gets_the_windows_the_request_is_about():
    mem = mem_with(messages=2)
    ctx, _ = turn(mem, "Where is the guest room key?")
    text = ctx.turn_text()
    assert "## What was said in earlier conversations" in text and "[session s1," in text
    assert "user: The guest room key is under the blue pot by the door." in text
    assert "blue pot" not in ctx.stable_text()  # a request-shaped half, not the stable one
    assert ctx.said[0].score > 0 and len(ctx.said[0].lines) == 3


def test_nothing_unrelated_is_pushed():
    mem = mem_with(messages=2)
    ctx, _ = turn(mem, "What is the capital of Peru?")
    assert ctx.said == []


def test_only_this_users_other_sessions():
    mem = mem_with(messages=3)
    talk(mem, "b1", [("user", "Bob's guest room key is under the red mat.")], user="bob")
    ctx, _ = turn(mem, "guest room key", user="ada", sid="s2")
    assert {w.session_id for w in ctx.said} == {"s1"}
    # the current session is already in the conversation
    with mem.session("s3", user_id="ada") as s:
        s.message("user", "The garage key is on the hook.")
        assert all(w.session_id != "s3" for w in run(s.context("garage key")).said)
    # sessions without a user have no past messages to search
    anon = Memory(policy=MemoryPolicy(messages=3))
    with anon.session("a1", agent="chat") as s:
        s.message("user", "The guest room key is under the blue pot.")
    with anon.session("a2", agent="chat") as s:
        assert run(s.context("guest room key")).said == []


def test_recaps_across_agents_false_keeps_each_agents_messages_apart():
    mem = Memory(policy=MemoryPolicy(messages=3, recaps_across_agents=False))
    talk(mem, "s1", [("user", "The guest room key is under the blue pot.")], agent="codex")
    other, _ = turn(mem, "guest room key", agent="claude-code")
    same, _ = turn(mem, "guest room key", agent="codex", sid="s3")
    assert other.said == [] and len(same.said) == 1


def test_recall_returns_them_only_when_asked_to():
    for n, expected in ((0, False), (3, True)):
        mem = mem_with(recall_messages=n)
        with mem.session("s2", user_id="ada", agent="chat") as s:
            out = {t.__name__: t for t in build_agent_tools(s)}["memory_recall"]("where is the guest room key")
        assert ("[said] [session s1," in out and "blue pot" in out) is expected


def test_forgotten_text_is_redacted_and_forget_user_removes_the_messages():
    mem = Memory(policy=MemoryPolicy(messages=3))
    with mem.session("s1", user_id="ada") as s:
        s.remember("room_code", "4471")
        s.message("user", "The guest room code is 4471 and the key is under the blue pot.")
        s.forget("room_code")
    ctx, _ = turn(mem, "guest room code key")
    assert ctx.said and "4471" not in ctx.turn_text() and "[forgotten]" in ctx.turn_text()
    run(mem.forget_user("ada"))
    assert turn(mem, "guest room code key")[0].said == []


def test_the_trace_names_the_events_and_why_finds_them():
    mem = mem_with(messages=2)
    with mem.session("s2", user_id="ada", agent="chat") as s:
        ctx = run(s.push_turn("where is the guest room key"))
        [item] = [i for i in mem.traces(session_id="s2")[-1].items if i["kind"] == "said"]
        found = mem.why(item["id"], session_id="s2")
    assert item["session"] == "s1" and len(item["events"]) == 3 and ctx.said
    assert found is not None and [e.id for e in found.sources] == item["events"]


def test_a_stable_part_trace_does_not_claim_them():
    from commonmemo.traces import trace_items

    mem = mem_with(messages=2)
    ctx, _ = turn(mem, "guest room key")
    assert any(i["kind"] == "said" for i in trace_items(ctx, "turn"))
    assert not any(i["kind"] == "said" for i in trace_items(ctx, "stable"))


def test_what_was_left_out_is_counted_and_pointed_at_recall():
    mem = Memory(policy=MemoryPolicy(messages=1))
    for n in range(3):
        talk(mem, f"s{n}", [("user", f"The guest room key number {n} is under the blue pot.")])
    with mem.session("s9", user_id="ada") as s:
        s.recall_tool = True
        ctx = run(s.context("guest room key"))
    assert len(ctx.said) == 1 and ctx.more["messages"] == 2
    assert "2 earlier exchanges" in ctx.turn_text()


def test_windows_are_bounded():
    mem = Memory(policy=MemoryPolicy(messages=10))
    talk(mem, "s1", [("user", f"the garage door code is {n} " + "word " * 200) for n in range(WINDOW * 2)])
    ctx, _ = turn(mem, "garage door code")
    assert len(ctx.said) == 2 and all(len(w.lines) == WINDOW for w in ctx.said)
    assert all(len(line) <= 410 for w in ctx.said for line in w.lines)


def test_the_hooks_push_them_and_the_setup_carries_the_flags():
    mem = mem_with(messages=2)
    out = handle({"hook_event_name": "UserPromptSubmit", "session_id": "h1", "cwd": ".",
                  "prompt": "Where is the guest room key?"}, mem, harness="claude-code", user="ada")
    assert "blue pot" in out["hookSpecificOutput"]["additionalContext"]
    args = argparse.Namespace(store="sqlite:///.commonmemo/memory.db", engine="local", project=None, user="ada",
                              agent=None, session=None, read_only=False, recall_tool=False, no_environment=False,
                              messages=4, recall_messages=6)
    command = hook_config("claude-code", args)["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "--messages 4" in command and "--recall-messages 6" in command
