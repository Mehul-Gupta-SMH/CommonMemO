"""Google ADK: `BaseMemoryService` over CommonMemo.

    from commonmemo import Memory
    from commonmemo.adapters.adk import CommonMemoMemoryService

    runner = Runner(agent=agent, app_name="shop", session_service=sessions,
                    memory_service=CommonMemoMemoryService(Memory("sqlite:///mem.db")))

How ADK's calls land in the Log:

- `add_session_to_memory` / `add_events_to_memory`: each event's text parts become
  `Message`s (`user` for the user, `assistant` for any agent), under the ADK session
  resumed by its id and the ADK app as the agent. Ids come from the ADK event id, so adding a
  session again, or a turn that was already added, records nothing new. Events without text
  (function calls, state changes) are not stored: tool memory comes from `@mem.track` or
  the hooks.
- `add_memory`: each `MemoryEntry` becomes a `UserFact` for the user (key: the entry's id, or
  a hash of its text, so the same entry is the same fact).
- `search_memory`: what `memory_recall` finds for the query -- facts about the user, world
  facts, feedback, lessons, decisions, recaps of earlier sessions, and windows of what the
  user said -- as `MemoryEntry`s, best first, with the kind in `custom_metadata["kind"]`.

What `BaseMemoryService` cannot express: it has no delete or correction (use `Memory.forget_user`,
`revert`), no approval step (an agent's world-fact proposals wait for review as usual), and a
search is scoped to `(app_name, user_id)` only, so the recaps it returns are the user's across
agents (`MemoryPolicy.recaps_across_agents`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

try:
    from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
    from google.adk.memory.memory_entry import MemoryEntry
    from google.genai import types
except ImportError as exc:  # pragma: no cover -- exercised only without google-adk
    raise ImportError('commonmemo: the ADK adapter needs google-adk -- pip install "commonmemo[adk]"') from exc

from ..events import Message, UserFact
from ..agent_tools import recall_entries
from ._core import reader, stable_id, writer

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

    from ..memory import Memory


def _text(event: "Event") -> str:
    parts = getattr(getattr(event, "content", None), "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))


class CommonMemoMemoryService(BaseMemoryService):
    """An ADK memory service backed by a `Memory` (see the module docstring)."""

    def __init__(self, memory: "Memory", *, max_results: int = 10) -> None:
        self.memory = memory
        self.max_results = max_results

    def _record(self, app_name: str, user_id: str, session_id: Optional[str], events: Sequence["Event"]) -> None:
        messages = []
        for event in events:
            text = _text(event)
            if not text:
                continue
            ts = getattr(event, "timestamp", None)
            fields: dict[str, Any] = {} if ts is None else {"ts": float(ts)}
            messages.append(Message(
                id=stable_id("adk", app_name, user_id, session_id or "", getattr(event, "id", "") or text),
                role="user" if event.author == "user" else "assistant", content=text, **fields))
        if messages:
            writer(self.memory, session_id, user_id, app_name).record_many(messages)

    async def add_session_to_memory(self, session: "Session") -> None:
        self._record(session.app_name, session.user_id, session.id, session.events)

    async def add_events_to_memory(self, *, app_name: str, user_id: str, events: Sequence["Event"],
                                   session_id: Optional[str] = None,
                                   custom_metadata: Optional[Mapping[str, object]] = None) -> None:
        self._record(app_name, user_id, session_id, events)

    async def add_memory(self, *, app_name: str, user_id: str, memories: Sequence[MemoryEntry],
                         custom_metadata: Optional[Mapping[str, object]] = None) -> None:
        facts = []
        for entry in memories:
            text = "\n".join(p.text for p in (entry.content.parts or []) if p.text)
            if text:
                key = entry.id or stable_id("adk-memory", text)
                facts.append(UserFact(id=stable_id("adk-memory", app_name, user_id, key, text), key=key, value=text))
        if facts:  # a fact reaches the user's profile when its session closes
            written = writer(self.memory, f"adapter-memory:{app_name}:{user_id}", user_id, app_name)
            written.record_many(facts)
            await written.aclose()

    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        lines, _, _ = recall_entries(reader(self.memory, user_id, app_name), query)
        return SearchMemoryResponse(memories=[
            MemoryEntry(content=types.Content(role="user", parts=[types.Part(text=text)]),
                        custom_metadata={"kind": label}, author="commonmemo")
            for label, text in lines[: self.max_results]
        ])
