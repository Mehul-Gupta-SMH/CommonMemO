"""The opt-in local embedder and the hybrid ranker (#129)."""

from __future__ import annotations

import asyncio

import pytest

from commonmemo import Memory, MemoryPolicy
from commonmemo import embed as embed_mod
from commonmemo.ranking import HybridRanker
from tests.test_said import talk, turn


def toy(texts):  # "sibling" words land together; everything else is orthogonal noise
    return [[1.0 if any(w in t.lower() for w in ("sister", "brother", "sibling")) else 0.0, 0.1] for t in texts]


class FakeModel:
    """Stands in for a Model2Vec StaticModel: `encode(list) -> array-like rows`."""

    def encode(self, texts):
        class Row(list):
            def tolist(self):
                return list(self)
        return [Row(v) for v in toy(texts)]


def test_hybrid_blend_weights_shared_words_and_meaning_equally():
    scores = HybridRanker.blend("zebra code", ["zebra code 4471", "unrelated"], [0.0, 1.0])
    assert scores == [0.5, 0.5]  # all words and no meaning vs. all meaning and no words
    assert HybridRanker.blend("", ["x"], [0.4]) == [0.2]


def test_hybrid_ranker_scores_words_and_meaning():
    ranker = HybridRanker(toy)
    scores = asyncio.run(ranker.score("my sibling", ["My sister arrives", "my cat", "the dentist"]))
    assert scores[0] > scores[1] > scores[2] or scores[0] > scores[2]
    assert scores[0] == max(scores)


def test_windows_found_by_meaning_and_by_words_with_stored_vectors():
    mem = Memory(policy=MemoryPolicy(messages=2), ranker=HybridRanker(toy, threshold=0.3))
    talk(mem, "s1", [("user", "My sister arrives on Friday."), ("assistant", "Noted.")])
    talk(mem, "s2", [("user", "The dentist is on Tuesday."), ("assistant", "Noted.")])
    ctx, _ = turn(mem, "When does my sibling get here?", sid="q1")
    assert [w.session_id for w in ctx.said] == ["s1"]
    ctx, _ = turn(mem, "Which day is the dentist?", sid="q2")  # shared words, no meaning in common with the toy model
    assert "s2" in [w.session_id for w in ctx.said]


def test_local_ranker_with_a_given_model_needs_no_extra_or_network():
    ranker = embed_mod.local_ranker(model=FakeModel())
    assert isinstance(ranker, HybridRanker)
    assert asyncio.run(ranker.score("my sibling", ["my sister"]))[0] > 0.5


def test_unpinned_build_refuses_to_load_weights(monkeypatch):
    monkeypatch.setattr(embed_mod, "REVISION", "")
    with pytest.raises(embed_mod.EmbedUnavailable, match="not pinned"):
        embed_mod.load()


def test_missing_extra_and_missing_weights_fail_with_a_message(tmp_path):
    with pytest.raises(embed_mod.EmbedUnavailable):
        embed_mod.load(str(tmp_path), revision="r", sha256="0")


def test_a_checksum_mismatch_fails(tmp_path, monkeypatch):
    pytest.importorskip("model2vec")
    (tmp_path / "model.safetensors").write_bytes(b"not the weights")
    with pytest.raises(embed_mod.EmbedUnavailable, match="checksum"):
        embed_mod.load(str(tmp_path), revision="r", sha256="0" * 64)


@pytest.mark.skipif(not embed_mod.SHA256, reason="needs the pinned weights (commonmemo[embed] and network)")
def test_real_weights_load_and_rank():
    ranker = embed_mod.local_ranker()
    s = asyncio.run(ranker.score("my brother is visiting", ["my sibling arrives Friday", "the dentist on Tuesday"]))
    assert s[0] > s[1]
