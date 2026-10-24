"""What was said before: windows of a user's past messages (#62).

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

**Cost.** The latest `MAX_SCAN` messages are windowed and ranked on each
call: fine for thousands, not for a long history; an index is the next step.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

from .events import Message

if TYPE_CHECKING:
    from .memory import Memory

WINDOW = 6
"""Consecutive messages of a session per window."""

LINE_CHARS = 400
"""Each message is cut to this in a window."""

MAX_SCAN = 4000
"""The most recent messages considered per call."""


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


def windows(memory: "Memory", *, user_id: str, agent: Optional[str], exclude_session: Optional[str],
            across_agents: bool) -> list[Said]:
    """The user's past messages in windows, oldest first."""
    events = memory.store.events(user_id=user_id, kinds=[Message.kind], limit=MAX_SCAN)
    forgotten = memory._forgotten_values(user_id)
    pattern = re.compile("|".join(re.escape(v) for v in forgotten), re.IGNORECASE) if forgotten else None
    by_session: dict[str, list[Message]] = {}
    for e in events:
        if e.session_id is None or e.session_id == exclude_session:
            continue
        if not across_agents and agent is not None and e.agent != agent:
            continue
        by_session.setdefault(e.session_id, []).append(e)  # type: ignore[arg-type]
    out = []
    for sid, messages in by_session.items():
        for i in range(0, len(messages), WINDOW):
            chunk = messages[i: i + WINDOW]
            lines = [f"{m.role}: {_clip(m.content)}" for m in chunk]
            if pattern:
                lines = [pattern.sub("[forgotten]", line) for line in lines]
            out.append(Said(sid, chunk[0].ts, chunk[0].agent, tuple(lines), tuple(m.id for m in chunk)))
    return sorted(out, key=lambda w: w.ts)


async def search(memory: "Memory", query: Optional[str], *, user_id: Optional[str], agent: Optional[str],
                 exclude_session: Optional[str], limit: int, across_agents: bool = True,
                 threshold: Optional[float] = None) -> tuple[list[Said], int]:
    """`(the best `limit` windows, how many more matched)`, best first. Nothing
    without a user, a query, or a positive limit."""
    if not query or user_id is None or limit <= 0:
        return [], 0
    found = windows(memory, user_id=user_id, agent=agent, exclude_session=exclude_session,
                    across_agents=across_agents)
    if not found:
        return [], 0
    scores = await memory.ranker.score(query, [w.text() for w in found])
    bar = memory.ranker.threshold if threshold is None else threshold
    ranked = sorted(((s, n, w) for n, (s, w) in enumerate(zip(scores, found)) if s >= bar),
                    key=lambda t: (-t[0], -t[2].ts, t[1]))
    return [replace(w, score=round(s, 3)) for s, _, w in ranked[:limit]], max(0, len(ranked) - limit)


__all__ = ["LINE_CHARS", "MAX_SCAN", "Said", "WINDOW", "search", "windows"]
