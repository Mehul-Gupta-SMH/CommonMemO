"""`MemoryContext` -- what all four tiers have to say before a turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .engines.base import Episode, Fact
from .events import MemoryEvent, Message, ToolInvoked, UserFact


def _line(event: MemoryEvent) -> str:
    if isinstance(event, Message):
        return f"{event.role}: {event.content}"
    if isinstance(event, ToolInvoked):
        outcome = f"error: {event.error}" if event.error else (event.result or "")
        return f"tool {event.name} -> {outcome}"
    if isinstance(event, UserFact):
        return f"learned {event.key} = {event.value}"
    return f"{event.kind}: {event.payload}"


@dataclass
class MemoryContext:
    """Session state, recent events, relevant episodes and user facts.

    Use the fields directly, or `render()` for a block of text ready to
    prepend to an agent's instructions.
    """

    state: dict[str, Any] = field(default_factory=dict)
    recent: list[MemoryEvent] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.state or self.recent or self.episodes or self.facts)

    def render(self, *, max_chars: int = 4000) -> str:
        """Markdown sections, most durable first (user facts, then past
        episodes, then this session). Truncated to `max_chars`; empty
        string when there is nothing to say."""
        sections = []
        if self.facts:
            sections.append(
                "## What you know about this user\n"
                + "\n".join(f"- {f.key}: {f.value}" for f in self.facts)
            )
        if self.episodes:
            sections.append(
                "## Relevant past sessions\n" + "\n".join(f"- {e.summary}" for e in self.episodes)
            )
        if self.state:
            sections.append(
                "## Session state\n" + "\n".join(f"- {k}: {v}" for k, v in self.state.items())
            )
        if self.recent:
            sections.append(
                "## Earlier in this session\n" + "\n".join(f"- {_line(e)}" for e in self.recent)
            )
        text = "\n\n".join(sections)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
