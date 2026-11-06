"""An organisation's documents as passages (ADR 0006, #142).

Text an organisation wrote down (policies, runbooks, a wiki) goes into memory as *passages*:
chunks of a source document, scoped `org:<name>`, searched like message windows
(`said.py`) and never asserted as facts. A passage is a `PassageAdded` event, so it lives in
the Log; the projection keeps only ids and a term index, and the text is read from the Log
when a search returns it. A changed source replaces its passages (`PassageRetracted` for the
old ones, which stay in the Log); `forget_source` retracts them all.

Passage text is data. It is shown labelled with its source, and never as an instruction.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, Sequence

from .events import PassageAdded
from .ranking import EmbeddingRanker, KeywordRanker, terms

if TYPE_CHECKING:
    from .memory import Memory

CHUNK_CHARS = 1200
"""About how long a passage is. Paragraphs are packed up to this; a longer one is cut at a sentence or space."""
OVERLAP_CHARS = 150
"""How much of the end of a cut paragraph starts the next piece, so a sentence at the cut is in both."""
CANDIDATES = 300
SCAN_PASSAGES = 400
MAX_QUERY_TERMS = 40


@dataclass(frozen=True)
class Passage:
    id: str
    scope: str
    source: str
    location: str
    ts: float
    text: str = ""
    score: float = 0.0

    def line(self) -> str:
        where = f"{self.source}, {self.location}" if self.location else self.source
        return f"[{where}]\n{self.text}"


def chunk(text: str, *, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """`text` as pieces of about `size` characters: whole paragraphs where they fit."""
    out: list[str] = []
    cur = ""
    for para in (p.strip() for p in text.replace("\r\n", "\n").split("\n\n")):
        if not para:
            continue
        if len(para) > size:
            if cur:
                out.append(cur)
                cur = ""
            start = 0
            while start < len(para):
                end = min(len(para), start + size)
                if end < len(para):
                    cut = max(para.rfind(". ", start, end), para.rfind(" ", start + size // 2, end))
                    end = cut + 1 if cut > start else end
                out.append(para[start:end].strip())
                if end >= len(para):
                    break
                start = max(end - overlap, start + 1)
            continue
        if cur and len(cur) + 2 + len(para) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def plan(source: str, text: str, live: Sequence, *, scope: str, ts: float) -> list:
    """The events that make `source` hold `text`, given its `live` passage rows: none when the chunks
    are the same, retractions for chunks that went and additions for chunks that are new."""
    from .events import PassageRetracted

    pieces = chunk(text)
    have = [r["content_hash"] for r in live]
    want = [digest(p) for p in pieces]
    if have == want:
        return []
    wanted = set(want)
    events: list = [PassageRetracted(passage_id=r["id"], reason="source changed")
                    for r in live if r["content_hash"] not in wanted]
    present = {r["content_hash"] for r in live if r["content_hash"] in wanted}
    for n, (piece, h) in enumerate(zip(pieces, want), 1):
        if h in present:
            continue
        events.append(PassageAdded(scope=scope, source=source, location=f"part {n}", content_hash=h,
                                   text=piece, ts=ts))
        present.add(h)
    return events


def add(memory: "Memory", source: str, text: str, *, scope: str, ts: Optional[float] = None) -> tuple[int, int]:
    """Record `text` as the current content of `source`: `(added, retracted)`. The same chunks again add
    nothing; a source that changed has its old passages retracted and the new ones added, in one write."""
    now = ts if ts is not None else time.time()
    events = plan(source, text, memory.knowledge.passage_sources(scope, source), scope=scope, ts=now)
    memory._append_many(events)
    added = sum(isinstance(e, PassageAdded) for e in events)
    return added, len(events) - added


def forget(memory: "Memory", source: str, *, scope: str, reason: str = "forgotten") -> int:
    """Retract every live passage of `source`. The events stay in the Log; returns how many."""
    from .events import PassageRetracted

    live = memory.knowledge.passage_sources(scope, source)
    memory._append_many([PassageRetracted(passage_id=r["id"], reason=reason) for r in live])
    return len(live)


def _read(memory: "Memory", rows: Sequence) -> list[Passage]:
    events = {e.id: e for e in memory.store.get_events([r["id"] for r in rows])}
    return [Passage(id=r["id"], scope=r["scope"], source=r["source"], location=r["location"], ts=r["ts"],
                    text=events[r["id"]].text) for r in rows if r["id"] in events]


async def search(memory: "Memory", query: Optional[str], *, scopes: Sequence[str], limit: int,
                 threshold: Optional[float] = None) -> tuple[list[Passage], int]:
    """`(the best `limit` passages, how many more matched)`, best first. Nothing without a query,
    a scope or a positive limit."""
    if not query or not scopes or limit <= 0:
        return [], 0
    k = memory.knowledge
    bar = memory.ranker.threshold if threshold is None else threshold
    wanted = sorted(terms(query))[:MAX_QUERY_TERMS]
    if isinstance(memory.ranker, KeywordRanker):
        rows = k.passage_candidates(scopes, wanted, limit=CANDIDATES)
    elif isinstance(memory.ranker, EmbeddingRanker):
        rows = list({r["id"]: r for r in [*k.passage_candidates(scopes, wanted, limit=CANDIDATES),
                                           *k.recent_passages(scopes, limit=SCAN_PASSAGES)]}.values())
    else:
        rows = k.recent_passages(scopes, limit=SCAN_PASSAGES)
    found = _read(memory, rows)
    if not found:
        return [], 0
    scores = await memory.ranker.score(query, [p.text for p in found])
    ranked = sorted(((s, n, p) for n, (s, p) in enumerate(zip(scores, found)) if s >= bar),
                    key=lambda t: (-t[0], -t[2].ts, t[1]))
    return [replace(p, score=round(s, 3)) for s, _, p in ranked[:limit]], max(0, len(ranked) - limit)


__all__ = ["Passage", "add", "chunk", "forget", "search"]
