"""What was said before: windows of a user's past messages (#62, #65).

Recaps, facts and lessons are what memory made of earlier sessions. On five
LoCoMo conversations (`docs/live-runs.md`), pasting the few excerpts of the
raw transcript that share the most words with a question was as good as
extracted facts, and much better than the recaps alone; each found questions
the other missed. So memory can also search the messages themselves:

    mem = Memory(policy=MemoryPolicy(messages=4, recall_messages=6))

- `messages`: windows pushed with the turn (the request is the query);
- `recall_messages`: windows `memory_recall` returns.

Both default to 0: this puts a user's past words into the prompt, which
recaps and facts do not. The benchmark decides whether that should change.

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
embedding ranker scores meaning, not shared words, so it ranks the user's
`SCAN_WINDOWS` newest windows instead (a vector index is the next step).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

from .ranking import KeywordRanker, terms

if TYPE_CHECKING:
    from .memory import Memory

WINDOW = 6
"""Consecutive messages of a session per window."""

LINE_CHARS = 400
"""Each message is cut to this in a window."""

CANDIDATES = 300
"""Windows the index hands to the ranker."""

SCAN_WINDOWS = 400
"""Newest windows ranked when the ranker doesn't work on shared words."""

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

    @property
    def id(self) -> str:
        return self.event_ids[0]

    def text(self) -> str:
        return "\n".join(self.lines)

    def line(self) -> str:
        """The window as one block of text, saying when and in which session."""
        when = time.strftime("%Y-%m-%d", time.localtime(self.ts))
        return f"[session {self.session_id}, {when}]\n" + self.text()


def _clip(text: str, n: int = LINE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def window_line(role: str, content: str) -> str:
    """One message as a window shows it (and as its terms are indexed)."""
    return f"{role}: {_clip(content)}"


def _windows(memory: "Memory", rows: list, *, user_id: str) -> list[Said]:
    """The windows `rows` name, read from the Log: redacted, oldest first. A window whose
    messages `prune` removed has none left and is skipped."""
    ids = [i for r in rows for i in json.loads(r["event_ids"])]
    events = {e.id: e for e in memory.store.get_events(ids)}
    forgotten = memory._forgotten_values(user_id)
    pattern = re.compile("|".join(re.escape(v) for v in forgotten), re.IGNORECASE) if forgotten else None
    out = []
    for r in rows:
        messages = [events[i] for i in json.loads(r["event_ids"]) if i in events]
        if not messages:
            continue
        lines = [window_line(m.role, m.content) for m in messages]  # type: ignore[attr-defined]
        if pattern:
            lines = [pattern.sub("[forgotten]", line) for line in lines]
        out.append(Said(r["session_id"], r["first_ts"], r["agent"], tuple(lines), tuple(m.id for m in messages)))
    return sorted(out, key=lambda w: w.ts)


async def search(memory: "Memory", query: Optional[str], *, user_id: Optional[str], agent: Optional[str],
                 exclude_session: Optional[str], limit: int, across_agents: bool = True,
                 threshold: Optional[float] = None) -> tuple[list[Said], int]:
    """`(the best `limit` windows, how many more matched)`, best first. Nothing
    without a user, a query, or a positive limit."""
    if not query or user_id is None or limit <= 0:
        return [], 0
    only_agent = None if across_agents else agent
    if isinstance(memory.ranker, KeywordRanker):
        wanted = sorted(terms(query))[:MAX_QUERY_TERMS]
        rows = memory.knowledge.window_candidates(user_id, wanted, exclude_session=exclude_session,
                                                  agent=only_agent, limit=CANDIDATES)
    else:
        rows = memory.knowledge.recent_windows(user_id, exclude_session=exclude_session, agent=only_agent,
                                               limit=SCAN_WINDOWS)
    found = _windows(memory, rows, user_id=user_id)
    if not found:
        return [], 0
    scores = await memory.ranker.score(query, [w.text() for w in found])
    bar = memory.ranker.threshold if threshold is None else threshold
    ranked = sorted(((s, n, w) for n, (s, w) in enumerate(zip(scores, found)) if s >= bar),
                    key=lambda t: (-t[0], -t[2].ts, t[1]))
    return [replace(w, score=round(s, 3)) for s, _, w in ranked[:limit]], max(0, len(ranked) - limit)


__all__ = ["CANDIDATES", "LINE_CHARS", "SCAN_WINDOWS", "Said", "WINDOW", "search", "window_line"]
