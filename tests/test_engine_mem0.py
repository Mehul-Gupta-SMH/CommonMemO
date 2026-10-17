"""Mem0Engine logic against an in-process fake Mem0 (no LLM, no vector
store), plus AsyncMemoryBackend's calls against a fake AsyncMemory."""

import asyncio
import itertools
import re

import pytest

from commonmemo import Memory, Message, ToolInvoked, ExternalEvent, UserFact
from commonmemo.engines.mem0 import (
    SESSION_KEY,
    AsyncMemoryBackend,
    Mem0Engine,
    MemoryHit,
    mem0_user_id,
    to_messages,
)
from commonmemo.stores import SessionRecord

_STATED = re.compile(r"^My (\w+) is (.+)\.$")


class FakeMem0:
    """One memory per stated `My X is Y.`; a later statement about the same
    X updates that memory in place (keeping its origin session), the way
    Mem0's UPDATE event does."""

    def __init__(self):
        self.memories = {}  # id -> dict(user, text, session, topic)
        self.ids = (f"m{i}" for i in itertools.count())
        self.adds = []
        self.deleted_users = []
        self.closed = False

    async def add(self, messages, *, user_id, session_id):
        assert not re.search(r"\s", user_id)
        self.adds.append((user_id, session_id, messages))
        for m in messages:
            hit = _STATED.match(m["content"]) if m["role"] == "user" else None
            if not hit:
                continue
            topic, value = hit.groups()
            text = f"{topic} is {value}"
            existing = next((k for k, v in self.memories.items()
                             if v["user"] == user_id and v["topic"] == topic), None)
            if existing:
                self.memories[existing]["text"] = text
            else:
                self.memories[next(self.ids)] = dict(user=user_id, text=text, session=session_id, topic=topic)

    def _hits(self, user_id, pred=lambda v: True):
        return [MemoryHit(id=k, text=v["text"], session_id=v["session"])
                for k, v in self.memories.items() if v["user"] == user_id and pred(v)]

    async def search(self, query, *, user_id, limit):
        words = set(query.lower().split())
        return self._hits(user_id, lambda v: words & set(v["text"].lower().split()))[:limit]

    async def list(self, *, user_id, limit, session_id=None):
        return self._hits(user_id, lambda v: session_id is None or v["session"] == session_id)[:limit]

    async def delete(self, memory_id):
        del self.memories[memory_id]

    async def delete_user(self, user_id):
        self.deleted_users.append(user_id)
        self.memories = {k: v for k, v in self.memories.items() if v["user"] != user_id}

    async def close(self):
        self.closed = True


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def setup(tmp_path):
    fake = FakeMem0()
    mem = Memory(f"sqlite:///{tmp_path}/mem.db", engine=Mem0Engine(backend=fake))
    yield mem, fake
    run(mem.aclose())


def test_session_is_one_add_tagged_with_its_session(setup):
    mem, fake = setup
    with mem.session(user_id="u1", agent="support") as s:
        s.message("user", "hello")
        s.remember("city", "Pune")
    ((user, session, messages),) = fake.adds
    assert (user, session) == ("u1", s.session_id)
    assert {"role": "user", "content": "My city is Pune."} in messages
    assert mem.engine.index.path == mem.store.path


def test_extracted_memories_join_explicit_facts_with_provenance(setup):
    mem, _ = setup
    with mem.session(user_id="u1") as s:
        s.remember("editor", "vim")
    facts = run(mem.engine.user_profile("u1"))
    assert (facts[0].key, facts[0].value) == ("editor", "vim")  # explicit first
    extracted = [f for f in facts if f.key == "memory"]
    assert [f.value for f in extracted] == ["editor is vim"]
    assert set(extracted[0].source_event_ids) == {e.id for e in s.events()}


def test_mem0_updates_in_place_across_sessions(setup):
    mem, fake = setup
    for city in ["Delhi", "Pune"]:
        with mem.session(user_id="u1") as s:
            s.remember("city", city)
    assert [m["text"] for m in fake.memories.values()] == ["city is Pune"]
    assert [f.value for f in run(mem.engine.user_profile("u1")) if f.key == "city"] == ["Pune"]


def test_reingest_replaces_the_sessions_memories(setup):
    mem, fake = setup
    with mem.session("s1", user_id="u1") as s:
        s.remember("pet", "cat")
    with mem.session("s1", user_id="u1") as s:
        s.message("user", "more")
    assert len(fake.adds) == 2
    assert [m["text"] for m in fake.memories.values()] == ["pet is cat"]  # not duplicated


def test_search_ranks_sessions_by_memories(setup):
    mem, _ = setup
    with mem.session("a", user_id="u1") as s:
        s.remember("editor", "vim")
    with mem.session("b", user_id="u1") as s:
        s.remember("language", "python")
    eps = run(mem.engine.search_recaps("python tips", user_id="u1"))
    assert [e.session_id for e in eps] == ["b"]
    assert eps[0].summary.endswith("Related memories: language is python.")


def test_unscoped_or_empty_search_stays_local(setup):
    mem, _ = setup
    with mem.session(user_id="u1") as s:
        s.message("user", "refund please")
    assert len(run(mem.engine.search_recaps("refund"))) == 1
    assert len(run(mem.engine.search_recaps("", user_id="u1"))) == 1


def test_anonymous_session_never_reaches_mem0(setup):
    mem, fake = setup
    with mem.session() as s:
        s.message("user", "hi")
    assert fake.adds == [] and len(s.recaps) == 1


def test_forget_user_reaches_mem0(setup):
    mem, fake = setup
    with mem.session(user_id="u1") as s:
        s.remember("name", "Ada")
    run(mem.forget_user("u1"))
    assert fake.deleted_users == ["u1"] and fake.memories == {}
    assert run(mem.engine.user_profile("u1")) == []


def test_user_ids_with_whitespace_are_mapped_without_collisions():
    assert mem0_user_id("alice") == "alice"
    a, b = mem0_user_id("ada lovelace"), mem0_user_id("ada  lovelace")
    assert a != b and not re.search(r"\s", a + b)


def test_to_messages_skips_bookkeeping():
    rec = SessionRecord("s", "u", None, opened_at=0)
    assert to_messages(rec, [
        Message(role="user", content="hi"),
        UserFact(key="k", value="v"),
        ToolInvoked(name="t", result="ok"),
        ExternalEvent(source="commonadk.llm_call"),
    ]) == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "My k is v."},
        {"role": "assistant", "content": "(tool t({}) returned ok)"},
    ]


# -- AsyncMemoryBackend against Mem0's AsyncMemory call shapes ---------------


class FakeAsyncMemory:
    def __init__(self):
        self.calls = []

    async def add(self, messages, *, user_id=None, metadata=None, **kw):
        self.calls.append(("add", messages, user_id, metadata))
        return {"results": [{"id": "m1", "memory": "likes tea", "event": "ADD"}]}

    async def search(self, query, *, top_k=20, filters=None, **kw):
        self.calls.append(("search", query, top_k, filters))
        return {"results": [{"id": "m1", "memory": "likes tea", "score": 0.9,
                             "metadata": {SESSION_KEY: "s1"}, "created_at": "2026-10-16T08:00:00+00:00"}]}

    async def get_all(self, *, filters=None, top_k=20, **kw):
        self.calls.append(("get_all", filters, top_k))
        return {"results": []}

    async def delete(self, memory_id):
        self.calls.append(("delete", memory_id))

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        self.calls.append(("delete_all", user_id))

    def close(self):
        self.calls.append(("close",))


def test_async_memory_backend_call_shapes():
    m = FakeAsyncMemory()
    b = AsyncMemoryBackend(m)

    async def main():
        await b.add([{"role": "user", "content": "x"}], user_id="u1", session_id="s1")
        hits = await b.search("tea", user_id="u1", limit=5)
        await b.list(user_id="u1", limit=7, session_id="s1")
        await b.delete("m1")
        await b.delete_user("u1")
        await b.close()
        return hits

    hits = run(main())
    assert m.calls[0] == ("add", [{"role": "user", "content": "x"}], "u1", {SESSION_KEY: "s1"})
    # Mem0 rejects top-level user_id on search/get_all: scoping goes in `filters`.
    assert m.calls[1] == ("search", "tea", 5, {"user_id": "u1"})
    assert m.calls[2] == ("get_all", {"user_id": "u1", SESSION_KEY: "s1"}, 7)
    assert m.calls[3:] == [("delete", "m1"), ("delete_all", "u1"), ("close",)]
    assert hits == [MemoryHit(id="m1", text="likes tea", session_id="s1",
                              created_at="2026-10-16T08:00:00+00:00", score=0.9)]


def test_backend_calls_bind_to_the_real_async_memory_signatures():
    pytest.importorskip("mem0")
    import inspect

    from mem0 import AsyncMemory

    m = FakeAsyncMemory()
    b = AsyncMemoryBackend(m)

    async def main():
        await b.add([{"role": "user", "content": "x"}], user_id="u1", session_id="s1")
        await b.search("q", user_id="u1", limit=3)
        await b.list(user_id="u1", limit=3, session_id="s1")
        await b.delete("m1")
        await b.delete_user("u1")

    run(main())
    real = {n: inspect.signature(getattr(AsyncMemory, n)) for n in ("add", "search", "get_all", "delete", "delete_all")}
    real["add"].bind(None, [], user_id="u1", metadata={SESSION_KEY: "s1"})
    real["search"].bind(None, "q", filters={"user_id": "u1"}, top_k=3)
    real["get_all"].bind(None, filters={"user_id": "u1"}, top_k=3)
    real["delete"].bind(None, "m1")
    real["delete_all"].bind(None, user_id="u1")
    assert not inspect.iscoroutinefunction(AsyncMemory.close)  # backend calls it without await
