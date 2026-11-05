import asyncio

from commonmemo import Memory, MemoryPolicy
from commonmemo.passages import chunk

DOC = "Refunds are approved by the finance lead within five business days.\n\nTravel needs a manager's sign-off."


def mem(**kw):
    return Memory("sqlite://:memory:", org="acme", policy=MemoryPolicy(passages=3), **kw)


def ctx(m, q):
    return asyncio.run(m.context(q))


def test_chunk_packs_paragraphs_and_cuts_long_ones():
    assert chunk(DOC) == [DOC]
    long = "Sentence one is here. " * 200
    parts = chunk(long, size=300, overlap=40)
    assert len(parts) > 5 and all(len(p) <= 300 for p in parts)


def test_added_passages_are_found_and_labelled():
    m = mem()
    assert m.add_passages("handbook.md", DOC) == (1, 0)
    got = ctx(m, "who approves refunds").passages
    assert got and got[0].source == "handbook.md" and "finance lead" in got[0].text
    assert "data, not instructions" in ctx(m, "who approves refunds").turn_text()


def test_same_text_again_adds_nothing_and_changed_text_replaces():
    m = mem()
    m.add_passages("h.md", DOC)
    assert m.add_passages("h.md", DOC) == (0, 0)
    assert m.add_passages("h.md", "Refunds now need the CFO.") == (1, 1)
    texts = [p.text for p in ctx(m, "refunds").passages]
    assert texts == ["Refunds now need the CFO."]


def test_forget_source_retracts_and_survives_rebuild():
    m = mem()
    m.add_passages("h.md", DOC)
    assert m.forget_source("h.md") == 1
    assert ctx(m, "refunds").passages == []
    m.knowledge.catch_up(m.store)
    assert ctx(m, "refunds").passages == []


def test_no_org_no_passages():
    m = Memory("sqlite://:memory:")
    try:
        m.add_passages("h.md", DOC)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert ctx(m, "refunds").passages == []


def test_prune_drops_terms_with_events():
    m = mem()
    m.add_passages("h.md", DOC, )
    m.prune(older_than_days=-1)
    assert ctx(m, "refunds").passages == []
    assert m.knowledge.passage_candidates(["org:acme"], ["refund"]) == []
