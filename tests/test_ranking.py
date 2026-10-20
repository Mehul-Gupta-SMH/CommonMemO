"""Ranking by meaning (#21)."""

from __future__ import annotations

import asyncio

from commonmemo import Memory, MemoryPolicy
from commonmemo.agent_tools import build_agent_tools
from commonmemo.engines.graphiti import GraphitiEngine
from commonmemo.engines.mem0 import Mem0Engine
from commonmemo.ranking import EmbeddingRanker, KeywordRanker, rank, terms
from evals.embedder import embed


def run(coro):
    return asyncio.run(coro)


def recall(s, query):
    return {t.__name__: t for t in build_agent_tools(s)}["memory_recall"](query)


def test_keyword_ranker_normalizes():
    assert terms("Returns shipped to the addresses") == {"return", "ship", "address"}
    assert terms("shipping ships") == {"ship"} and terms("called calls") == {"call"}
    scores = run(KeywordRanker().score("how do I return this", ["refund_window: return within 30 days", "colour: red"]))
    assert scores[0] > 0 and scores[1] == 0


def test_embedding_ranker_caches_and_accepts_sync_or_async_embedders():
    calls = []

    def counted(texts):
        calls.append(list(texts))
        return embed(texts)

    async def async_embed(texts):
        return embed(texts)

    ranker = EmbeddingRanker(counted)
    run(ranker.score("reach me", ["preferred_contact: email", "diet: vegan"]))
    run(ranker.score("reach me", ["preferred_contact: email", "diet: vegan"]))
    assert calls == [["reach me", "preferred_contact: email", "diet: vegan"]]  # embedded once
    a = run(EmbeddingRanker(async_embed).score("reach me", ["preferred_contact: email", "diet: vegan"]))
    assert a[0] > 0.3 > a[1]


def test_recall_finds_by_meaning_with_an_embedder():
    for ranker, found in ((KeywordRanker(), False), (EmbeddingRanker(embed), True)):
        mem = Memory(ranker=ranker)
        with mem.session(user_id="ada") as s:
            s.remember("preferred_contact", "email")
            s.remember("diet", "vegetarian")
        with mem.session(user_id="ada") as s:
            out = recall(s, "How do I reach them?")
        assert ("preferred_contact: email" in out) is found, (ranker, out)
        assert "vegetarian" not in out


def test_the_push_picks_related_facts_not_the_first_twenty():
    mem = Memory(ranker=EmbeddingRanker(embed))
    with mem.session(user_id="ada") as s:
        for i in range(48):
            s.remember(f"a_note_{i:02d}", f"misc {i}")
        s.remember("shoe_size", "42")
        s.remember("preferred_contact", "email")
    with mem.session(user_id="ada") as s:
        keys = [f.key for f in run(s.context("what size should I order?")).facts]
        assert len(keys) == 20 and keys[0] == "shoe_size"
        assert "preferred_contact" not in keys  # unrelated, and only 20 fit
        keys = [f.key for f in run(s.context("how will you contact me?")).facts]
        assert keys[0] == "preferred_contact"


def test_keyword_ranking_also_picks_related_facts():
    mem = Memory()
    with mem.session(user_id="ada") as s:
        for i in range(48):
            s.remember(f"a_note_{i:02d}", f"misc {i}")
        s.remember("shoe_size", "42")
    with mem.session(user_id="ada") as s:
        assert run(s.context("what shoe size do I wear?")).facts[0].key == "shoe_size"


def test_a_small_profile_is_never_ranked():
    calls = []
    mem = Memory(ranker=EmbeddingRanker(lambda texts: calls.append(texts) or embed(texts)))
    with mem.session(user_id="ada") as s:
        s.remember("preferred_contact", "email")
        run(s.context("anything"))
    assert calls == []  # everything fits: no embedding call per turn


def test_recall_threshold_can_be_set():
    mem = Memory(ranker=EmbeddingRanker(embed))
    mem.policy_for_recall.min_recall_score = 0.99
    with mem.session(user_id="ada") as s:
        s.remember("preferred_contact", "email")
        assert "Nothing in memory matches" in recall(s, "How do I reach them?")


def test_engines_with_embedders_supply_a_ranker():
    class Embedder:
        async def create_batch(self, texts):
            return embed(texts)

    class Graphiti:
        embedder = Embedder()

    class GraphBackend:
        graphiti = Graphiti()

    class Model:
        def embed(self, text, action):
            assert action == "search"
            return embed([text])[0]

    class Mem0Memory:
        embedding_model = Model()

    class Mem0Backend:
        memory = Mem0Memory()

    for engine in (GraphitiEngine(backend=GraphBackend()), Mem0Engine(backend=Mem0Backend())):
        ranker = engine.ranker()
        assert isinstance(ranker, EmbeddingRanker)
        related, _ = run(rank(ranker, "how do I reach them", ["preferred_contact: email", "diet: vegan"], str))
        assert related == ["preferred_contact: email"]
    assert isinstance(Memory().ranker, KeywordRanker)  # the local engine doesn't embed


def test_policy_field_defaults():
    assert MemoryPolicy().min_recall_score is None
