"""What was said before: windows of a user's past messages (#62, #65).

Recaps, facts and lessons are what memory made of earlier sessions. On five
LoCoMo conversations (`docs/live-runs.md`), pasting the few excerpts of the
raw transcript that share the most words with a question was as good as
extracted facts, and much better than the recaps alone; each found questions
the other missed. So memory can also search the messages themselves:

    mem = Memory(policy=MemoryPolicy(messages=4, recall_messages=6))  # the defaults are 8 and 8

- `messages`: windows pushed with the turn (the request is the query);
- `recall_messages`: windows `memory_recall` returns.

Both default to 8 (#70): the benchmark says they are worth having, and a
search no longer slows with history. This puts a user's own past words into
the prompt, which recaps and facts do not: `MemoryPolicy(messages=0,
recall_messages=0)` (`--messages 0 --recall-messages 0`) turns it off.

`MemoryPolicy.message_dates` says when a window was said: `stamp` gives it the
weekday, date and time, `resolve` also what each relative date means (`dates.py`,
#71). The ranker scores the messages as said; the dates are for the reader.

**Windows.** The messages of one session, `WINDOW` consecutive ones, each
line clipped to `LINE_CHARS`. They are ranked against the query by the
memory's ranker (shared words by default, meaning with an embedder) and the
best come back with their session and date. A window carries the ids of its
events, so traces, `why` and the inspector show where it came from.

**Whose.** Only the user's own messages, from their other sessions: never
the current one (already in the conversation), never another user's, and a
session without a user has none (anonymous sessions share nothing).
`MemoryPolicy.recaps_across_agents=False` limits it to this agent's sessions.
Text the user asked to forget is redacted, and `forget_user` removes the
messages.

**The index (#65).** Windows are a projection of the Log (`Knowledge`): as
each message is logged it joins its session's open window, and the window's
stemmed terms go into an inverted index. A window stores the ids of its
events, not their text, so the text stays in the Log where `prune`,
`forget_user` and redaction reach it, and `prune` drops the windows and
terms of what it removed. A search asks the index for the windows that share
the most terms with the query (`CANDIDATES` of them), reads only those from
the Log and ranks them, so its cost follows the candidates, not the history.
With the keyword ranker the index returns the same windows a scan would. An
embedding ranker scores meaning, not shared words, so it scores those
windows and the user's `VECTOR_SCAN` newest ones by cosine over stored vectors
(#77): a window is embedded once, from its redacted text, in the first search that
needs it (at most `EMBED_PER_SEARCH` a search), and every later search, in any
process, embeds only the request and a fixed probe text. If the embedder gives
another vector for the probe than it did, the stored vectors are from another
model and are dropped; one made before a value was forgotten is made again;
`forget_user` and `prune` delete them. Plain Python, numpy when installed.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

from .dates import annotate, stamp
from .ranking import EmbeddingRanker, KeywordRanker, cosine, terms

if TYPE_CHECKING:
    from .memory import Memory

WINDOW = 6
"""Consecutive messages of a session per window."""

LINE_CHARS = 400
"""Each message is cut to this in a window."""

CANDIDATES = 300
"""Windows the index hands to the ranker."""

SCAN_WINDOWS = 400
"""Newest windows ranked when the ranker neither works on shared words nor is an `EmbeddingRanker`."""

VECTOR_SCAN = 2000
"""Newest windows an embedding ranker scores, besides the ones the term index finds (#77)."""

EMBED_PER_SEARCH = 500
"""Windows embedded for the first time by one search, newest first; the rest wait for later searches."""

PROBE = "The quick brown fox jumps over the lazy dog."
"""Embedded with every request: if the embedder gives another vector than it did when the stored
vectors were made, it is another model, and they are dropped."""

MAX_QUERY_TERMS = 40


@dataclass(frozen=True)
class Said:
    """One window of a past session."""

    session_id: str
    ts: float
    """When its first message was recorded."""
    agent: Optional[str]
    lines: tuple[str, ...]
    event_ids: tuple[str, ...]
    score: float = 0.0
    when: str = ""
    """When its first message was recorded, with weekday and time, if the policy asks for dates (#71)."""
    window: str = ""
    """The id of the window in the index (`session#number`)."""
    said: str = ""
    """The messages as said, without stamps or annotations: what the ranker scores, because the dates
    are for the reader (on LoCoMo, ranking on the annotated text lost evidence the plain text found, #71)."""

    @property
    def id(self) -> str:
        return self.event_ids[0]

    def text(self) -> str:
        return "\n".join(self.lines)

    def line(self) -> str:
        """The window as one block of text, saying when and in which session."""
        when = self.when or time.strftime("%Y-%m-%d", time.localtime(self.ts))
        return f"[session {self.session_id}, {when}]\n" + self.text()


def _clip(text: str, n: int = LINE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def window_line(role: str, content: str, when: Optional[str] = None) -> str:
    """One message as a window shows it; `when` is a stamp for a message recorded at another time than the window's first."""
    return f"{role} ({when}): {_clip(content)}" if when else f"{role}: {_clip(content)}"


def _windows(memory: "Memory", rows: list, *, user_id: str, dates: str = "off") -> list[Said]:
    """The windows `rows` name, read from the Log: redacted, oldest first. A window whose
    messages `prune` removed has none left and is skipped. With `dates` of `stamp` or `resolve`
    the window says when it was said, and with `resolve` what its relative dates mean (#71)."""
    ids = [i for r in rows for i in json.loads(r["event_ids"])]
    events = {e.id: e for e in memory.store.get_events(ids)}
    forgotten = memory._forgotten_values(user_id)
    pattern = re.compile("|".join(re.escape(v) for v in forgotten), re.IGNORECASE) if forgotten else None
    out = []
    for r in rows:
        messages = [events[i] for i in json.loads(r["event_ids"]) if i in events]
        if not messages:
            continue
        first = stamp(messages[0].ts)
        lines = []
        bare = []
        for m in messages:
            bare.append(window_line(m.role, m.content))  # type: ignore[attr-defined]
            when = stamp(m.ts)
            line = window_line(m.role, m.content, when if dates != "off" and when != first else None)  # type: ignore[attr-defined]
            lines.append(annotate(line, m.ts) if dates == "resolve" else line)
        if pattern:
            lines = [pattern.sub("[forgotten]", line) for line in lines]
            bare = [pattern.sub("[forgotten]", line) for line in bare]
        out.append(Said(r["session_id"], r["first_ts"], r["agent"], tuple(lines), tuple(m.id for m in messages),
                        when=first if dates != "off" else "", said="\n".join(bare), window=r["id"]))
    return sorted(out, key=lambda w: w.ts)


def _redaction(forgotten: list[str]) -> str:
    """Names the set of forgotten values a vector's text was redacted with; another set, another vector."""
    return hashlib.sha1("\n".join(sorted(forgotten)).encode()).hexdigest()[:16] if forgotten else ""


def _cosines(q: list[float], vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    try:
        import numpy as np
    except ImportError:
        return [max(0.0, cosine(q, v)) for v in vectors]
    m = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1) * float(np.linalg.norm(q))
    return np.maximum(0.0, (m @ np.asarray(q, dtype=np.float32)) / np.where(norms == 0, 1.0, norms)).tolist()


async def _by_meaning(memory: "Memory", query: str, rows: list, *, user_id: str, limit: int, bar: float,
                      dates: str) -> tuple[list[Said], int]:
    """The windows of `rows` closest in meaning to `query`, from vectors stored with the windows (#77):
    a window is embedded once, from its redacted text, and every later search (in any process)
    embeds only the request. The same as ranking the rows with the ranker, which the tests check."""
    ranker, k = memory.ranker, memory.knowledge
    assert isinstance(ranker, EmbeddingRanker)
    if not rows:
        return [], 0  # nothing to score: no embedding call for the request either
    q, probe = await ranker.vectors([query, PROBE])
    stored = k.vector_probe()
    if stored is None or len(stored) != len(probe) or cosine(stored, probe) < 0.999:
        k.reset_vectors(probe)  # another embedder: the stored vectors mean nothing now
    forgotten = memory._forgotten_values(user_id)
    redaction = _redaction(forgotten)
    have = k.window_vectors([r["wid"] for r in rows], redaction)
    todo = sorted((r for r in rows if r["wid"] not in have), key=lambda r: -r["first_ts"])[:EMBED_PER_SEARCH]
    if todo:
        wid_of = {r["id"]: r["wid"] for r in todo}
        windows = _windows(memory, todo, user_id=user_id)  # redacted, read from the Log
        vectors = await ranker.vectors([w.said for w in windows])
        fresh = [(wid_of[w.window], redaction, v) for w, v in zip(windows, vectors)]
        k.store_window_vectors(fresh)
        have.update({wid: list(v) for wid, _, v in fresh})
    scored = [r for r in rows if r["wid"] in have]
    scores = _cosines(q, [have[r["wid"]] for r in scored])
    ranked = sorted(((s, r) for s, r in zip(scores, scored) if s >= bar), key=lambda t: (-t[0], -t[1]["first_ts"], t[1]["id"]))
    shown = _windows(memory, [r for _, r in ranked[:limit]], user_id=user_id, dates=dates)
    score_of = {r["id"]: s for s, r in ranked[:limit]}
    out = [replace(w, score=round(score_of[w.window], 3)) for w in shown]
    return sorted(out, key=lambda w: (-w.score, -w.ts)), max(0, len(ranked) - limit)


async def search(memory: "Memory", query: Optional[str], *, user_id: Optional[str], agent: Optional[str],
                 exclude_session: Optional[str], limit: int, across_agents: bool = True,
                 threshold: Optional[float] = None, dates: str = "off") -> tuple[list[Said], int]:
    """`(the best `limit` windows, how many more matched)`, best first. Nothing
    without a user, a query, or a positive limit."""
    if not query or user_id is None or limit <= 0:
        return [], 0
    only_agent = None if across_agents else agent
    ids = memory.linked_users(user_id)  # one person under several ids (link_users)
    bar = memory.ranker.threshold if threshold is None else threshold
    if isinstance(memory.ranker, KeywordRanker):
        wanted = sorted(terms(query))[:MAX_QUERY_TERMS]
        rows = memory.knowledge.window_candidates(ids, wanted, exclude_session=exclude_session,
                                                  agent=only_agent, limit=CANDIDATES)
    elif isinstance(memory.ranker, EmbeddingRanker):
        # meaning, not shared words: the windows sharing the most terms (old ones included) and the newest
        wanted = sorted(terms(query))[:MAX_QUERY_TERMS]
        rows = list({r["id"]: r for r in [
            *memory.knowledge.window_candidates(ids, wanted, exclude_session=exclude_session, agent=only_agent,
                                                limit=CANDIDATES),
            *memory.knowledge.recent_windows(ids, exclude_session=exclude_session, agent=only_agent,
                                             limit=VECTOR_SCAN)]}.values())
        return await _by_meaning(memory, query, rows, user_id=user_id, limit=limit, bar=bar, dates=dates)
    else:
        rows = memory.knowledge.recent_windows(ids, exclude_session=exclude_session, agent=only_agent,
                                               limit=SCAN_WINDOWS)
    found = _windows(memory, rows, user_id=user_id, dates=dates)
    if not found:
        return [], 0
    scores = await memory.ranker.score(query, [w.said or w.text() for w in found])
    ranked = sorted(((s, n, w) for n, (s, w) in enumerate(zip(scores, found)) if s >= bar),
                    key=lambda t: (-t[0], -t[2].ts, t[1]))
    return [replace(w, score=round(s, 3)) for s, _, w in ranked[:limit]], max(0, len(ranked) - limit)


__all__ = ["CANDIDATES", "LINE_CHARS", "SCAN_WINDOWS", "Said", "WINDOW", "search", "window_line"]
