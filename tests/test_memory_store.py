"""Memory on the plain-Python store (#95): the library does not need SQL for the Log and Working memory."""

import asyncio
import threading

from commonmemo import Memory, Message
from commonmemo.stores import MemoryStore


def test_a_conversation_end_to_end_on_the_memory_store():
    mem = Memory("memory://")
    assert isinstance(mem.store, MemoryStore)
    with mem.session("s1", user_id="ada", agent="a") as s:
        s.message("user", "the guest room key is under the blue pot")
        s.message("assistant", "Noted.")
        s.remember("preferred_contact", "email")
    with mem.session("s2", user_id="ada", agent="a") as s:
        ctx = asyncio.run(s.push_turn("where is the guest room key"))
    assert "blue pot" in ctx.turn_text() and "email" in ctx.stable_text()
    assert asyncio.run(mem.forget_user("ada")) > 0
    assert mem.store.events(user_id="ada") == [] and mem.store.session_ids("ada") == []


def test_one_claim_wins_a_race_between_threads():
    store = MemoryStore()
    wins, barrier = [], threading.Barrier(8)

    def go(n):
        barrier.wait()
        if store.claim("step", session_id="s", owner=f"w{n}", lease_until=1.0, event=Message(role="user", content=str(n))) is None:
            wins.append(n)

    threads = [threading.Thread(target=go, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(store.events()) == 1  # one owner, one event, whatever the interleaving


def test_concurrent_appends_get_distinct_ordered_positions():
    store = MemoryStore()

    def add(n):
        for i in range(50):
            store.append(Message(role="user", content=f"{n}-{i}"))

    threads = [threading.Thread(target=add, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [n for n, _ in store.events_after(0)]
    assert seqs == list(range(1, 201))
