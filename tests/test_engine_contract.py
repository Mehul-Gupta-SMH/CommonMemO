"""The `Engine` contract (#99): one suite, run against every engine that can run offline.

`LocalEngine` (SQL) and `ListEngine`, a few lines of plain Python defined here, so the suite itself is shown
to pass on something that is not SQL; and `GraphitiEngine` and `Mem0Engine` on the fakes their own tests
use (`FakeGraph`, `FakeMem0`), which exercises each engine's own code (what it keeps locally, how it maps
sessions, users and provenance onto the backend, forgetting, re-ingest) without a model. What the fakes cannot
say anything about is Graphiti's or Mem0's real extraction and search quality: that is a live run
(`scripts/live_engines.py`, #3). A new engine is correct when this passes for it: add it to `ENGINES`.
"""

import asyncio

import pytest

from commonmemo import Message, ToolInvoked, UserFact
from commonmemo.engines import LocalEngine
from commonmemo.engines.base import Engine, Fact, Recap
from commonmemo.engines.local import content_words, summarize
from commonmemo.engines.graphiti import GraphitiEngine
from commonmemo.engines.mem0 import Mem0Engine
from commonmemo.stores import MemoryStore, SessionRecord
from test_engine_graphiti import FakeGraph  # noqa: E402 -- the fakes the engines' own tests use
from test_engine_mem0 import FakeMem0  # noqa: E402


class ListEngine(Engine):
    """The contract with lists and dicts: no SQL, no persistence, word overlap for relevance."""

    name = "list"

    def __init__(self) -> None:
        self.recaps: dict[str, Recap] = {}
        self.facts: list[dict] = []

    async def ingest(self, session, events):
        if not events:
            return []
        recap = Recap(f"ep_{session.session_id}", session.session_id, session.user_id, session.agent,
                      summarize(session, events), min(e.ts for e in events), max(e.ts for e in events),
                      tuple(e.id for e in events))
        self.recaps[session.session_id] = recap  # a re-ingest replaces
        self.facts = [f for f in self.facts if f["session"] != session.session_id]
        for e in sorted((e for e in events if isinstance(e, UserFact)), key=lambda e: e.ts):
            user = e.user_id or session.user_id
            if user is None:
                continue
            for f in self.facts:
                if f["user"] == user and f["key"] == e.key and f["to"] is None:
                    f["to"] = e.ts
            self.facts.append({"user": user, "key": e.key, "value": e.value, "from": e.ts, "to": None,
                               "sources": (e.id,), "session": session.session_id})
        return [recap]

    async def search_recaps(self, query, *, user_id=None, agent=None, limit=5):
        wanted = content_words(query)
        found = []
        for r in self.recaps.values():
            if (user_id is not None and r.user_id != user_id) or (agent is not None and r.agent != agent):
                continue
            overlap = len(wanted & content_words(r.summary))
            if wanted and not overlap:
                continue
            found.append((overlap / len(wanted) if wanted else 0.0, r.ended_at, r))
        found.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [Recap(**{**r.__dict__, "score": s}) for s, _, r in found[:limit]]

    async def user_profile(self, user_id):
        return [Fact(f["user"], f["key"], f["value"], f["from"], f["sources"])
                for f in sorted(self.facts, key=lambda f: f["key"]) if f["user"] == user_id and f["to"] is None]

    async def forget_user(self, user_id):
        gone = [s for s, r in self.recaps.items() if r.user_id == user_id]
        n = len(gone) + sum(1 for f in self.facts if f["user"] == user_id)
        for s in gone:
            del self.recaps[s]
        self.facts = [f for f in self.facts if f["user"] != user_id]
        return n


ENGINES = {
    "local": LocalEngine,
    "list": ListEngine,
    "graphiti": lambda: GraphitiEngine(backend=FakeGraph()),
    "mem0": lambda: Mem0Engine(backend=FakeMem0()),
}


@pytest.fixture(params=sorted(ENGINES))
def engine(request):
    e = ENGINES[request.param]()
    e.bind(MemoryStore())  # Memory does this once, before anything else
    yield e
    asyncio.run(e.aclose())


def run(coro):
    return asyncio.run(coro)


def session(sid, user="ada", agent="a"):
    return SessionRecord(sid, user, agent, opened_at=1.0)


def stated(engine, user, keys=("city", "pet")):
    """The user's explicit facts for the keys the tests state: the first one listed for each key. An engine may add
    extracted facts beside them (`Mem0Engine`, `GraphitiEngine`: that is what a model reading the conversation is
    for), sometimes under the same key, but after them and never in place of them: explicit facts win."""
    out: dict = {}
    for f in run(engine.user_profile(user)):
        if f.key in keys:
            out.setdefault(f.key, f)
    return out


def talk(sid, said, *, user="ada", ts=10.0, facts=()):
    events = [Message(role="user", content=said, session_id=sid, user_id=user, ts=ts),
              Message(role="assistant", content="Noted.", session_id=sid, user_id=user, ts=ts + 1)]
    events += [UserFact(key=k, value=v, session_id=sid, user_id=user, ts=ts + 2 + i) for i, (k, v) in enumerate(facts)]
    return events


def test_ingest_makes_a_recap_that_names_its_sources(engine):
    events = talk("s1", "the guest room key is under the blue pot", ts=10.0)
    (recap,) = run(engine.ingest(session("s1"), events))
    assert (recap.session_id, recap.user_id, recap.agent) == ("s1", "ada", "a")
    assert recap.summary and recap.started_at <= recap.ended_at
    assert set(recap.source_event_ids) == {e.id for e in events}  # provenance: every recap says where it came from
    assert run(engine.ingest(session("s2"), [])) == []  # nothing happened, nothing to recap


def test_ingesting_a_session_again_replaces_never_duplicates(engine):
    events = talk("s1", "the guest room key is under the blue pot", facts=[("pet", "dog")])
    run(engine.ingest(session("s1"), events))
    run(engine.ingest(session("s1"), events))
    assert len(run(engine.search_recaps("guest room key", user_id="ada"))) == 1
    assert {k: f.value for k, f in stated(engine, "ada").items()} == {"pet": "dog"}


def test_search_puts_the_best_match_first_and_scores_it(engine):
    run(engine.ingest(session("s1"), talk("s1", "the guest room key is under the blue pot", ts=10.0)))
    run(engine.ingest(session("s2"), talk("s2", "please renew the passport before december", ts=20.0)))
    best = run(engine.search_recaps("where is the guest room key"))
    assert best[0].session_id == "s1" and best[0].score is not None
    assert run(engine.search_recaps("passport renewal"))[0].session_id == "s2"


def test_search_filters_by_user_and_agent_and_limits(engine):
    run(engine.ingest(session("s1", "ada", "claude"), talk("s1", "shared topic about boats", user="ada", ts=10.0)))
    run(engine.ingest(session("s2", "bob", "codex"), talk("s2", "shared topic about boats", user="bob", ts=20.0)))
    run(engine.ingest(session("s3", "ada", "codex"), talk("s3", "shared topic about boats", user="ada", ts=30.0)))
    assert {r.session_id for r in run(engine.search_recaps("boats", user_id="ada"))} == {"s1", "s3"}
    assert [r.session_id for r in run(engine.search_recaps("boats", agent="codex", user_id="ada"))] == ["s3"]
    assert len(run(engine.search_recaps("boats", limit=2))) == 2
    assert run(engine.search_recaps("boats", user_id="nobody")) == []


def test_the_profile_holds_current_facts_only_with_their_sources(engine):
    first = talk("s1", "hello", ts=10.0, facts=[("city", "Pune"), ("pet", "dog")])
    run(engine.ingest(session("s1"), first))
    second = talk("s2", "I moved", ts=50.0, facts=[("city", "Oslo")])
    run(engine.ingest(session("s2"), second))
    profile = stated(engine, "ada")
    assert {k: f.value for k, f in profile.items()} == {"city": "Oslo", "pet": "dog"}  # the superseded value is gone
    fact_event = next(e for e in second if isinstance(e, UserFact))
    assert profile["city"].source_event_ids == (fact_event.id,) and profile["city"].user_id == "ada"
    assert stated(engine, "bob") == {}


def test_facts_belong_to_their_user(engine):
    run(engine.ingest(session("s1", "ada"), talk("s1", "hi", user="ada", facts=[("city", "Pune")])))
    run(engine.ingest(session("s2", "bob"), talk("s2", "hi", user="bob", facts=[("city", "Lima")])))
    assert stated(engine, "ada")["city"].value == "Pune" and stated(engine, "bob")["city"].value == "Lima"


def test_forget_user_removes_their_recaps_and_facts_and_nobody_else_s(engine):
    run(engine.ingest(session("s1", "ada"), talk("s1", "boats and more boats", user="ada", facts=[("city", "Pune")])))
    run(engine.ingest(session("s2", "bob"), talk("s2", "boats and more boats", user="bob", facts=[("city", "Lima")])))
    assert run(engine.forget_user("ada")) > 0
    assert run(engine.user_profile("ada")) == [] and run(engine.search_recaps("boats", user_id="ada")) == []
    assert stated(engine, "bob")["city"].value == "Lima"
    assert [r.session_id for r in run(engine.search_recaps("boats"))] == ["s2"]
    assert run(engine.forget_user("ada")) == 0


def test_tool_use_shows_in_the_recap_and_the_optional_parts_have_defaults(engine):
    events = talk("s1", "deploy the app", ts=10.0) + [ToolInvoked(name="deploy", session_id="s1", user_id="ada", ts=12.0)]
    (recap,) = run(engine.ingest(session("s1"), events))
    assert "deploy" in recap.summary
    assert engine.ranker() is None or hasattr(engine.ranker(), "score")
    assert isinstance(engine.name, str) and engine.name
