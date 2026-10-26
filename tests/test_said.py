"""Windows of a user's past messages in the push and in recall (#62)."""

from __future__ import annotations

import argparse
import asyncio

import pytest

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


def test_on_by_default_and_off_with_zero():
    lines = [("user", "Hello."), ("user", "The guest room key is under the blue pot."),
             ("assistant", "Noted.")]  # past the first message, so the recap digest doesn't hold it
    on = Memory()
    talk(on, "s1", lines)
    ctx, _ = turn(on, "where is the guest room key?")
    assert len(ctx.said) == 1 and "blue pot" in ctx.turn_text()
    with on.session("s3", user_id="ada", agent="chat") as s:
        out = {t.__name__: t for t in build_agent_tools(s)}["memory_recall"]("where is the guest room key")
    assert "[said] [session s1," in out and "blue pot" in out
    off = Memory(policy=MemoryPolicy(messages=0, recall_messages=0))
    talk(off, "s1", lines)
    ctx, _ = turn(off, "where is the guest room key?")
    assert ctx.said == [] and "blue pot" not in ctx.turn_text()


def test_the_command_line_and_the_default_policy_agree():
    from commonmemo.cli import WINDOWS, memory_options

    assert str(MemoryPolicy().messages) == WINDOWS == str(MemoryPolicy().recall_messages)
    args = memory_options(argparse.ArgumentParser(), agent=None).parse_args([])
    assert (args.messages, args.recall_messages) == (MemoryPolicy().messages, MemoryPolicy().recall_messages)
    assert memory_options(argparse.ArgumentParser(), agent=None).parse_args(["--messages", "0"]).messages == 0


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
    assert item["session"] == "s1" and ctx.said and "events" not in item  # the window id names them (#83)
    assert found is not None and [e.id for e in found.sources] == list(ctx.said[0].event_ids) and len(found.sources) == 3


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
    # the default needs no flag; turning them off does
    args.messages, args.recall_messages = 8, 0
    command = hook_config("claude-code", args)["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "--messages" not in command and "--recall-messages 0" in command


# -- the index (#65) ------------------------------------------------------------------------


def rows(mem, table):
    return mem.knowledge._conn.execute(f"SELECT * FROM {table}").fetchall()


def test_windows_fill_as_messages_arrive_and_a_rebuild_gives_the_same_index():
    mem = Memory(policy=MemoryPolicy(messages=3))
    talk(mem, "s1", [("user", f"message {i} about the garage code") for i in range(WINDOW + 3)])
    before = sorted((r["id"], r["n"], r["event_ids"]) for r in rows(mem, "message_windows"))
    terms_before = len(rows(mem, "window_terms"))
    assert [n for _, n, _ in before] == [WINDOW, 3]
    mem.knowledge.rebuild(mem.store)
    assert sorted((r["id"], r["n"], r["event_ids"]) for r in rows(mem, "message_windows")) == before
    assert len(rows(mem, "window_terms")) == terms_before
    # a session continued later (same id) fills the open window first
    talk(mem, "s1", [("user", "one more about the garage code")] * 4)
    assert sorted(r["n"] for r in rows(mem, "message_windows")) == [1, WINDOW, WINDOW]


def test_the_index_ranks_what_a_scan_of_every_window_would():
    from commonmemo.ranking import KeywordRanker
    from commonmemo.said import window_line

    mem = Memory(policy=MemoryPolicy(messages=5))
    topics = ["garage code", "sister Lena visit", "blue pot key", "flight to Lisbon", "dentist on Tuesday"]
    for s in range(8):
        talk(mem, f"s{s}", [("user", f"note {s}.{i}: {topics[(s + i) % 5]} and some filler words") for i in range(9)])
    query = "when does my sister Lena visit and where is the key"
    ctx, _ = turn(mem, query, sid="now")
    # brute force: every window of every session, ranked by the same ranker
    windows = []
    for s in range(8):
        msgs = mem.store.events(session_id=f"s{s}", kinds=["message"])
        for i in range(0, len(msgs), WINDOW):
            windows.append("\n".join(window_line(m.role, m.content) for m in msgs[i: i + WINDOW]))
    scores = run(KeywordRanker().score(query, windows))
    best = sorted(set(scores), reverse=True)[0]
    assert ctx.said and ctx.said[0].score == round(best, 3)
    assert {w.text() for w in ctx.said} <= set(windows)


def test_prune_and_forget_user_take_the_index_with_them():
    mem = mem_with(messages=2)
    assert rows(mem, "message_windows") and rows(mem, "window_terms")
    mem.prune(older_than_days=-1)  # everything is older than tomorrow
    assert rows(mem, "message_windows") == [] and rows(mem, "window_terms") == []
    assert turn(mem, "guest room key")[0].said == []
    mem = mem_with(messages=2)
    run(mem.forget_user("ada"))
    assert rows(mem, "message_windows") == [] and rows(mem, "window_terms") == [] and rows(mem, "window_users") == []


def test_the_scratch_copies_history_replays_into_keep_no_windows():
    from commonmemo.knowledge import Knowledge

    mem = mem_with(messages=2)
    scratch = Knowledge(":memory:", windows=False)
    scratch.catch_up(mem.store)
    assert scratch._conn.execute("SELECT COUNT(*) AS n FROM message_windows").fetchone()["n"] == 0
    anon = Memory()
    with anon.session("a1") as s:
        s.message("user", "no user, so nothing is indexed")
    assert rows(anon, "message_windows") == []


def test_an_embedding_ranker_finds_windows_that_share_no_word_with_the_request():
    from commonmemo.ranking import EmbeddingRanker

    def embed(texts):  # a toy concept embedder: "sibling" words land together
        return [[1.0 if any(w in t.lower() for w in ("sister", "brother", "sibling")) else 0.0, 0.1] for t in texts]

    mem = Memory(policy=MemoryPolicy(messages=2), ranker=EmbeddingRanker(embed, threshold=0.5))
    talk(mem, "s1", [("user", "My sister arrives on Friday."), ("assistant", "Noted.")])
    talk(mem, "s2", [("user", "The dentist is on Tuesday."), ("assistant", "Noted.")])
    ctx, _ = turn(mem, "When does my sibling get here?")  # shares no word with the message
    assert [w.session_id for w in ctx.said] == ["s1"]


def test_a_search_reads_only_its_candidates_from_the_log(monkeypatch):
    mem = Memory(policy=MemoryPolicy(messages=2))
    for s in range(30):
        talk(mem, f"s{s}", [("user", f"filler {s} about nothing in particular"), ("assistant", "ok")])
    talk(mem, "needle", [("user", "the zebra enclosure code is 4471")])
    asked = []
    original = mem.store.get_events
    monkeypatch.setattr(mem.store, "get_events", lambda ids: asked.append(len(list(ids))) or original(ids))
    ctx, _ = turn(mem, "zebra enclosure code")
    assert [w.session_id for w in ctx.said] == ["needle"] and max(asked) <= 2  # not the 61 messages in the Log


def test_index_rows_carry_small_integers_not_ids(tmp_path):
    mem = mem_with(messages=2)
    talk(mem, "b1", [("user", "Bob's guest room key is under the red mat.")], user="bob")
    assert {r["user_id"]: r["uid"] for r in rows(mem, "window_users")} == {"ada": 1, "bob": 2}
    assert sorted(r["wid"] for r in rows(mem, "message_windows")) == [1, 2]
    assert all(isinstance(r["uid"], int) and isinstance(r["wid"], int) for r in rows(mem, "window_terms"))
    assert {r["uid"] for r in rows(mem, "window_terms")} == {1, 2}
    # forgetting one user leaves the other's index alone
    run(mem.forget_user("ada"))
    assert {r["uid"] for r in rows(mem, "window_terms")} == {2}
    assert [w.session_id for w in turn(mem, "guest room key", user="bob", sid="s9")[0].said] == ["b1"]


@pytest.mark.sqlite_only
def test_an_index_in_the_old_shape_is_rebuilt_from_the_log(tmp_path):
    db = tmp_path / "memory.db"
    mem = Memory(f"sqlite:///{db}", policy=MemoryPolicy(messages=2))
    talk(mem, "s1", [("user", "The guest room key is under the blue pot by the door.")])
    run(mem.aclose())
    import sqlite3

    raw = sqlite3.connect(db)  # put the index back as it was before #76 (schema 6) and break it
    raw.executescript("DROP TABLE window_terms; DROP TABLE window_users; "
                      "CREATE TABLE window_terms (user_id TEXT NOT NULL, term TEXT NOT NULL, window_id TEXT NOT NULL, "
                      "PRIMARY KEY (user_id, term, window_id)); UPDATE knowledge_meta SET value = 6 WHERE key = 'schema'")
    raw.commit()
    raw.close()
    mem = Memory(f"sqlite:///{db}", policy=MemoryPolicy(messages=2))
    ctx, _ = turn(mem, "where is the guest room key?")
    assert [w.session_id for w in ctx.said] == ["s1"] and "uid" in {c for c in mem.knowledge._conn.columns("window_terms")}


# -- stored vectors (#77) -------------------------------------------------------------------


class Embedder:
    """A toy concept embedder that records every text it is asked to embed."""

    CONCEPTS = (("sister", "brother", "sibling"), ("dentist", "doctor"), ("key", "pot"), ("flight", "plane", "lisbon"))

    def __init__(self):
        self.seen, self.calls = [], 0

    def __call__(self, texts):
        self.calls += 1
        self.seen += list(texts)
        return [[float(any(w in t.lower() for w in group)) + 0.01 for group in self.CONCEPTS] for t in texts]


def vec_mem(embed, path=None, **policy):
    from commonmemo.ranking import EmbeddingRanker

    options = dict(policy=MemoryPolicy(messages=3, **policy), ranker=EmbeddingRanker(embed, threshold=0.5))
    return Memory(f"sqlite:///{path}", **options) if path else Memory(**options)


def vec_talk(mem):
    talk(mem, "s1", [("user", "My sister arrives on Friday."), ("assistant", "Noted.")])
    talk(mem, "s2", [("user", "The dentist is on Tuesday."), ("assistant", "Noted.")])
    talk(mem, "s3", [("user", "I booked a flight to Lisbon."), ("assistant", "Great.")])


def test_vectors_match_ranking_the_windows_with_the_ranker():
    from commonmemo.ranking import EmbeddingRanker
    from commonmemo.said import _windows

    embed = Embedder()
    mem = vec_mem(embed)
    vec_talk(mem)
    query = "When does my sibling get here?"
    ctx, _ = turn(mem, query)
    rows = mem.knowledge.recent_windows("ada", exclude_session="now")
    windows = _windows(mem, rows, user_id="ada")
    direct = run(EmbeddingRanker(Embedder(), threshold=0.5).score(query, [w.said for w in windows]))
    expected = sorted(((round(s, 3), w.session_id) for s, w in zip(direct, windows) if s >= 0.5), reverse=True)
    assert [(w.score, w.session_id) for w in ctx.said] == expected and expected[0][1] == "s1"


def test_a_second_process_embeds_only_the_request(tmp_path):
    db = tmp_path / "memory.db"
    first = Embedder()
    mem = vec_mem(first, db)
    vec_talk(mem)
    turn(mem, "When does my sibling get here?", sid="now")
    assert len(first.seen) > 3  # the windows were embedded once
    run(mem.aclose())
    second = Embedder()  # a new process: a new ranker with an empty cache, the same file
    mem = vec_mem(second, db)
    ctx, _ = turn(mem, "When does my sibling get here?", sid="now")
    assert [w.session_id for w in ctx.said] == ["s1"]
    assert sorted(second.seen) == sorted(["When does my sibling get here?", "The quick brown fox jumps over the lazy dog."])


def test_a_windows_older_than_the_newest_are_found_by_their_words(monkeypatch):
    import commonmemo.said as said

    monkeypatch.setattr(said, "VECTOR_SCAN", 1)
    mem = vec_mem(Embedder())
    vec_talk(mem)  # three windows; only the newest is scanned, the sister window is found through its words
    ctx, _ = turn(mem, "When does my sister arrive?", sid="now")
    assert [w.session_id for w in ctx.said] == ["s1"]


def test_another_embedder_drops_the_stored_vectors(tmp_path):
    db = tmp_path / "memory.db"
    mem = vec_mem(Embedder(), db)
    vec_talk(mem)
    turn(mem, "When does my sibling get here?", sid="now")
    stored = len(rows(mem, "window_vectors"))
    run(mem.aclose())
    other = Embedder()  # the same dimensions, a different model: other directions for the same text
    other.__class__ = type("Other", (Embedder,), {"__call__": lambda self, texts: (
        setattr(self, "seen", self.seen + list(texts)) or [[float(len(t) % 7), 1.0, 0.0, 2.0] for t in texts])})
    mem = vec_mem(other, db)
    turn(mem, "When does my sibling get here?", sid="now")
    assert stored == 3 and len(other.seen) > 3  # all of it embedded again


def test_a_forgotten_value_never_reaches_the_embedder_or_a_stored_vector():
    embed = Embedder()
    mem = vec_mem(embed)
    with mem.session("s1", user_id="ada") as s:
        s.remember("code", "4471")
        s.message("user", "My sister's gate code is 4471.")
        s.forget("code")
    turn(mem, "When does my sibling get here?", sid="now")
    assert embed.seen and not any("4471" in t for t in embed.seen)
    # a value forgotten after its vector was stored makes the vector stale: it is made again, redacted
    mem = vec_mem(embed)
    with mem.session("s1", user_id="ada") as s:
        s.message("user", "My sister's gate code is 9912.")
    turn(mem, "When does my sibling get here?", sid="now")
    before = len(embed.seen)
    with mem.session("s5", user_id="ada") as s:
        s.remember("code2", "9912")
        s.forget("code2")
    turn(mem, "When does my sibling get here?", sid="s6")
    assert len(embed.seen) > before and not any("9912" in t for t in embed.seen[before:])


def test_prune_and_forget_user_take_the_vectors_with_them():
    mem = vec_mem(Embedder())
    vec_talk(mem)
    turn(mem, "When does my sibling get here?", sid="now")
    assert rows(mem, "window_vectors")
    mem.prune(older_than_days=-1)
    assert rows(mem, "window_vectors") == []
    mem = vec_mem(Embedder())
    vec_talk(mem)
    turn(mem, "When does my sibling get here?", sid="now")
    run(mem.forget_user("ada"))
    assert rows(mem, "window_vectors") == []


def test_at_most_so_many_windows_are_embedded_by_one_search(monkeypatch):
    import commonmemo.said as said

    monkeypatch.setattr(said, "EMBED_PER_SEARCH", 2)
    embed = Embedder()
    mem = vec_mem(embed)
    vec_talk(mem)
    turn(mem, "When does my sibling get here?", sid="now")
    assert len(rows(mem, "window_vectors")) == 2  # the newest two; the next search does the rest
    turn(mem, "When does my sibling get here?", sid="now7")
    assert len(rows(mem, "window_vectors")) == 3


def test_numpy_and_plain_python_agree(monkeypatch):
    import builtins

    from commonmemo.said import _cosines

    assert _cosines([1.0], []) == []
    q, vs = [1.0, 0.0, 0.5], [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    with_numpy = _cosines(q, vs)
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "numpy" else real(name, *a, **k))
    plain = _cosines(q, vs)
    assert [round(x, 5) for x in with_numpy] == [round(x, 5) for x in plain] == [1.0, 0.0, 0.0, 0.0]


def test_no_windows_means_no_embedding_call():
    embed = Embedder()
    mem = vec_mem(embed)
    turn(mem, "When does my sibling get here?", sid="now")
    assert embed.seen == []


def test_a_recap_item_names_a_few_events_and_how_many_there_are():
    from commonmemo.traces import RECAP_EVENTS

    mem = Memory()
    with mem.session("s1", user_id="ada", agent="chat") as s:
        for i in range(RECAP_EVENTS * 3):
            s.message("user", f"the guest room key number {i} is in the hall")
    with mem.session("s2", user_id="ada", agent="chat") as s:
        ctx = run(s.push_turn("where is the guest room key"))
        [item] = [i for i in mem.traces(session_id="s2")[-1].items if i["kind"] == "recap"]
        found = mem.why(item["id"], session_id="s2")
    assert len(item["events"]) == RECAP_EVENTS and item["n"] >= RECAP_EVENTS * 3 and ctx.recaps
    assert found is not None and any("more events of session s1" in line for line in found.lines())
