"""Ranking memory against a request, by meaning when an embedder is available (#21).

The push and `memory_recall` have to choose: which of a user's 50 profile
facts go in the push, which unpinned world facts the request is about, what
recall returns. They all ask one `Ranker`, so they all agree:

- **`KeywordRanker`** (the default; no dependencies): the share of the
  request's words found in the text, after lowercasing, dropping stop words
  and light stemming ("returns" matches "return").
- **`EmbeddingRanker(embed)`**: cosine similarity of embeddings, from any
  function that embeds a list of texts (sync or async). Text embeddings are
  cached, so a profile isn't re-embedded every turn. "How do I reach them?"
  finds `preferred_contact: email`.

`Memory(ranker=...)` picks one. Without one, an engine that embeds supplies
its own (`Engine.ranker()`: Graphiti's embedder, Mem0's embedding model);
else the keyword ranker. Each ranker has a `threshold`: below it, an item
doesn't count as related (`MemoryPolicy.min_recall_score` overrides it for
recall).
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional, Protocol, Sequence, Union, runtime_checkable

Embedder = Callable[[list[str]], Union[Sequence[Sequence[float]], Awaitable[Sequence[Sequence[float]]]]]

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be been by can could did do does for from had has have how i if in is it its me my "
    "of on or our should so that the their them then there these they this to us was we were what when where "
    "which who why will with would you your".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ers", "ies", "ed", "es", "er", "ly", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            stem = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            if suffix in ("ing", "ed", "er", "ers") and len(stem) > 2 and stem[-1] == stem[-2] \
                    and stem[-1] not in "lsz":
                stem = stem[:-1]  # shipped -> ship, not shipp
            return _drop_e(stem)
    return _drop_e(word)


def _drop_e(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("e") else word  # write, writing -> writ


def terms(text: str) -> set[str]:
    """The stemmed content words of `text`."""
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOP}


@runtime_checkable
class Ranker(Protocol):
    threshold: float
    """Scores below this mean "not related"."""

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """A relevance score in [0, 1] for each of `texts` against `query`."""
        ...


class KeywordRanker:
    """Shared words, normalized: the fraction of the query's terms in the text."""

    threshold = 0.01  # any shared term counts

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        wanted = terms(query)
        if not wanted:
            return [0.0 for _ in texts]
        return [len(wanted & terms(t)) / len(wanted) for t in texts]


class EmbeddingRanker:
    """Cosine similarity of embeddings from `embed(texts) -> vectors`."""

    def __init__(self, embed: Embedder, *, threshold: float = 0.3, cache_size: int = 4096) -> None:
        self.embed = embed
        self.threshold = threshold
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size

    async def _vectors(self, texts: Sequence[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            out = self.embed(missing)
            vectors = await out if inspect.isawaitable(out) else out
            for text, vector in zip(missing, vectors):
                self._cache[text] = [float(x) for x in vector]
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        for t in texts:
            self._cache.move_to_end(t)  # recently used: evicted last
        return [self._cache[t] for t in texts]

    async def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        q, *rest = await self._vectors([query, *texts])
        return [max(0.0, _cosine(q, v)) for v in rest]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def sync_embedder(embed_one: Callable[[str], Sequence[float]]) -> Embedder:
    """An `Embedder` from a function that embeds one text at a time,
    run off the event loop (Mem0's embedding models are like this)."""

    async def embed(texts: list[str]) -> list[Sequence[float]]:
        return list(await asyncio.gather(*(asyncio.to_thread(embed_one, t) for t in texts)))

    return embed


async def rank(ranker: Ranker, query: Optional[str], items: Sequence[Any], text: Callable[[Any], str],
               *, threshold: Optional[float] = None) -> tuple[list[Any], list[Any]]:
    """`(related, rest)`: the items scoring at least the threshold, best
    first, and the others in their original order. No query: nothing is
    related and the order stays."""
    if not query or not items:
        return [], list(items)
    scores = await ranker.score(query, [text(i) for i in items])
    bar = ranker.threshold if threshold is None else threshold
    related = [i for _, n, i in sorted(((s, n, i) for n, (s, i) in enumerate(zip(scores, items)) if s >= bar),
                                       key=lambda t: (-t[0], t[1]))]
    chosen = {id(i) for i in related}
    return related, [i for i in items if id(i) not in chosen]


__all__ = ["Embedder", "EmbeddingRanker", "KeywordRanker", "Ranker", "rank", "sync_embedder", "terms"]
