"""GraphitiEngine logic against an in-process fake graph (no database, no
LLM), plus GraphitiBackend's calls into graphiti-core against a fake client
when graphiti-core is installed."""

import asyncio
import inspect
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from commonmemo import Memory
from commonmemo.engines.graphiti import (
    SYSTEM,
    GraphFact,
    GraphitiEngine,
    group_id_for,
    transcript,
)

_FACT_LINE = re.compile(r"^user: my (\w+) is (.+)$")


class FakeGraph:
    """Stands in for Graphiti: one "extracted" fact per `user: my X is Y`
    line, keyword search over fact text, newer facts invalidating older ones
    with the same name in a group."""

    def __init__(self):
        self.episodes = {}  # uuid -> dict
        self.facts = []  # list of [group_id, GraphFact]
        self.removed = []
        self.deleted_groups = []
        self.closed = False

    async def add_episode(self, *, group_id, name, body, source_description, reference_time):
        # Like Graphiti: the graph assigns the episode uuid.
        uuid = f"ep-{len(self.episodes) + len(self.removed)}"
        assert re.match(r"^[A-Za-z0-9_-]+$", group_id)
        self.episodes[uuid] = dict(group_id=group_id, name=name, body=body, ref=reference_time)
        for line in body.splitlines():
            m = _FACT_LINE.match(line)
            if not m:
                continue
            key, value = m.groups()
            for i, (g, f) in enumerate(self.facts):
                if g == group_id and f.name == key and f.invalid_at is None:
                    self.facts[i] = [g, GraphFact(f.name, f.fact, f.episode_ids, f.valid_at, reference_time)]
            self.facts.append([group_id, GraphFact(key, f"user {key} is {value}", (uuid,), reference_time)])
        return uuid

    async def remove_episode(self, uuid):
        if uuid not in self.episodes:
            return False
        del self.episodes[uuid]
        self.facts = [[g, f] for g, f in self.facts if f.episode_ids != (uuid,)]
        self.removed.append(uuid)
        return True

    async def search(self, query, *, group_ids, limit):
        words = set(query.lower().split())
        hits = [
            f for g, f in self.facts
            if (group_ids is None or g in group_ids) and words & set(f.fact.lower().split())
        ]
        return hits[:limit]

    async def current_facts(self, group_id, *, limit):
        return [f for g, f in self.facts if g == group_id and f.invalid_at is None][:limit]

    async def delete_group(self, group_id):
        self.deleted_groups.append(group_id)
        self.facts = [[g, f] for g, f in self.facts if g != group_id]

    async def close(self):
        self.closed = True


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def setup(tmp_path):
    graph = FakeGraph()
    mem = Memory(f"sqlite:///{tmp_path}/mem.db", engine=GraphitiEngine(backend=graph))
    yield mem, graph
    run(mem.aclose())


def test_index_shares_the_store_file(setup):
    mem, _ = setup
    assert mem.engine.index.path == mem.store.path


def test_session_becomes_one_graph_episode_in_the_users_group(setup):
    mem, graph = setup
    with mem.session(user_id="u1", agent="support") as s:
        s.message("user", "my laptop will not boot")
        s.remember("city", "Pune")
        s.message("assistant", "Try holding the power button.")
    (ep,) = graph.episodes.values()
    assert ep["group_id"] == "u_u1"
    assert "user: my city is Pune" in ep["body"]
    assert "support: Try holding the power button." in ep["body"]
    assert ep["ref"].tzinfo is timezone.utc
    assert list(graph.episodes) == [mem.engine.index.external_id(s.session_id, SYSTEM)]


def test_reingest_replaces_the_graph_episode(setup):
    mem, graph = setup
    with mem.session("s1", user_id="u1") as s:
        s.message("user", "first")
    first = mem.engine.index.external_id("s1", SYSTEM)
    with mem.session("s1", user_id="u1") as s:
        s.message("user", "second")
    assert graph.removed == [first]
    (uid, ep), = graph.episodes.items()
    assert "first" in ep["body"] and "second" in ep["body"]
    assert mem.engine.index.external_id("s1", SYSTEM) == uid != first


def test_search_ranks_recaps_by_graph_facts_and_explains_why(setup):
    mem, _ = setup
    with mem.session("a", user_id="u1") as s:
        s.remember("editor", "vim")
    with mem.session("b", user_id="u1") as s:
        s.remember("language", "python")
    eps = run(mem.engine.search_recaps("which python version", user_id="u1"))
    assert [e.session_id for e in eps] == ["b"]
    assert "Related facts: user language is python." in eps[0].summary
    assert eps[0].score == 1.0
    assert eps[0].source_event_ids  # provenance survives the graph round trip


def test_search_is_scoped_to_the_user(setup):
    mem, _ = setup
    with mem.session(user_id="u1") as s:
        s.remember("language", "python")
    with mem.session(user_id="u2") as s:
        s.remember("language", "python")
    eps = run(mem.engine.search_recaps("python", user_id="u2"))
    assert len(eps) == 1 and eps[0].user_id == "u2"


def test_no_graph_hit_falls_back_to_local_keyword_search(setup):
    mem, _ = setup
    with mem.session(user_id="u1") as s:
        s.message("user", "refund for order 42")
    eps = run(mem.engine.search_recaps("refund", user_id="u1"))
    assert len(eps) == 1 and "refund" in eps[0].summary


def test_user_facts_explicit_first_then_extracted_with_provenance(setup):
    mem, graph = setup
    with mem.session(user_id="u1") as s:
        s.remember("units", "metric")
    # A fact only the graph knows (extracted from conversation, not stated).
    (uid,) = graph.episodes
    graph.facts.append(["u_u1", GraphFact("WORKS_AT", "user works at Acme", (uid,))])
    facts = run(mem.engine.user_profile("u1"))
    assert [(f.key, f.value) for f in facts][0] == ("units", "metric")
    extracted = {f.key: f for f in facts[1:]}
    assert extracted["WORKS_AT"].value == "user works at Acme"
    assert extracted["WORKS_AT"].source_event_ids  # traced through the episode


def test_invalidated_graph_facts_are_not_current(setup):
    mem, _ = setup
    for city in ["Delhi", "Pune"]:
        with mem.session(user_id="u1") as s:
            s.remember("city", city)
    extracted = [f.value for f in run(mem.engine.user_profile("u1")) if f.key == "city" and f.value.startswith("user")]
    assert extracted == ["user city is Pune"]


def test_forget_user_deletes_the_group_and_the_index(setup):
    mem, graph = setup
    with mem.session(user_id="u1") as s:
        s.remember("name", "Ada")
    run(mem.forget_user("u1"))
    assert graph.deleted_groups == ["u_u1"]
    assert run(mem.engine.user_profile("u1")) == []
    assert mem.engine.index.external_id(s.session_id, SYSTEM) is None


def test_group_ids_are_valid_and_collision_free():
    assert group_id_for("alice-01") == "u_alice-01"
    a, b = group_id_for("a@x.com"), group_id_for("a#x.com")
    assert a != b and all(re.match(r"^[A-Za-z0-9_-]+$", g) for g in (a, b))
    assert group_id_for(None) == "commonmemo_anonymous"


def test_transcript_skips_framework_bookkeeping():
    from commonmemo import ExternalEvent, Message, ToolInvoked
    from commonmemo.stores import SessionRecord

    rec = SessionRecord("s", "u", "bot", opened_at=0)
    text = transcript(rec, [
        Message(role="user", content="hi"),
        ToolInvoked(name="t", arguments={"x": 1}, error="boom"),
        ExternalEvent(source="commonadk.llm_call", data={"total_tokens": 9}),
    ])
    assert text == "user: hi\ntool: t({'x': 1}) failed with boom"


def test_aclose_closes_the_graph(setup):
    mem, graph = setup
    run(mem.aclose())
    assert graph.closed


# -- GraphitiBackend against graphiti-core's real types ----------------------


class FakeGraphitiClient:
    def __init__(self):
        self.calls = []
        self.driver = object()

    async def add_episode(self, **kwargs):
        self.calls.append(("add_episode", kwargs))
        return SimpleNamespace(episode=SimpleNamespace(uuid="graph-assigned"))

    async def remove_episode(self, uuid):
        from graphiti_core.errors import NodeNotFoundError

        raise NodeNotFoundError(uuid)

    async def search(self, query, group_ids=None, num_results=10):
        self.calls.append(("search", query, group_ids, num_results))
        return [SimpleNamespace(name="LIKES", fact="u likes tea", episodes=["e1"], valid_at=None,
                                invalid_at=None, expired_at=None)]

    async def close(self):
        self.calls.append(("close",))


def test_graphiti_backend_maps_onto_graphiti_core():
    pytest.importorskip("graphiti_core")
    from graphiti_core.nodes import EpisodeType

    from commonmemo.engines.graphiti import GraphitiBackend

    client = FakeGraphitiClient()
    backend = GraphitiBackend(client)
    when = datetime(2026, 10, 16, tzinfo=timezone.utc)

    async def main():
        assert await backend.add_episode(group_id="u_1", name="n", body="b",
                                         source_description="d", reference_time=when) == "graph-assigned"
        assert await backend.remove_episode("missing") is False
        facts = await backend.search("tea", group_ids=["u_1"], limit=3)
        await backend.close()
        return facts

    facts = run(main())
    _, kwargs = client.calls[0]
    # No `uuid`: in Graphiti that names an *existing* episode to reprocess.
    assert kwargs == dict(name="n", episode_body="b", source_description="d", reference_time=when,
                          source=EpisodeType.message, group_id="u_1")
    from graphiti_core import Graphiti
    inspect.signature(Graphiti.add_episode).bind(None, **kwargs)
    assert client.calls[1] == ("search", "tea", ["u_1"], 3)
    assert facts == [GraphFact("LIKES", "u likes tea", ("e1",))]
    assert client.calls[-1] == ("close",)


def test_engine_by_name_reads_connection_from_env(monkeypatch):
    pytest.importorskip("graphiti_core")
    for var in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="NEO4J_URI"):
        Memory(engine="graphiti")
