"""OpenAI Agents SDK: `Session` over CommonMemo.

    from agents import Agent, Runner
    from commonmemo import Memory
    from commonmemo.adapters.openai_agents import CommonMemoSession

    session = CommonMemoSession(Memory("sqlite:///mem.db"), "ada-chat", user_id="ada")
    await Runner.run(agent, "My daughter is allergic to peanuts.", session=session)

How the SDK's calls land in the Log:

- `add_items`: every item is kept verbatim as `ExternalEvent(source="openai_agents.item")`, so
  `get_items` returns exactly what the SDK stored (function calls, outputs, reasoning and all).
  A user or assistant item with text also becomes a `Message`, so what was said is searchable
  (message windows, recaps) like anything else the user said.
- `get_items(limit)`: the live items in order; `limit` keeps the latest N.
- `pop_item` / `clear_session`: the Log is append-only, so they record a compensating
  `openai_agents.pop` / `openai_agents.clear` event and the item stops being returned. Nothing is
  deleted.

What `Session` cannot express: removing an item does not remove the `Message` made from it,
so popped or cleared text stays searchable as something the user said. To remove it, use
`Memory.forget_user` (or redact it) -- the Log keeps what happened, by design. The SDK's
`session_settings` (default item limit) is kept as the attribute the SDK reads; the adapter does
not use it.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional

try:
    from agents.items import TResponseInputItem
except ImportError as exc:  # pragma: no cover -- exercised only without openai-agents
    raise ImportError(
        'commonmemo: the OpenAI Agents adapter needs openai-agents -- pip install "commonmemo[openai-agents]"'
    ) from exc

from ..events import ExternalEvent, Message
from ._core import writer

if TYPE_CHECKING:
    from ..memory import Memory

ITEM, POP, CLEAR = "openai_agents.item", "openai_agents.pop", "openai_agents.clear"


def _said(item: Any) -> Optional[tuple[str, str]]:
    """(role, text) for a user or assistant item with text; None for everything else."""
    if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
        return None
    content = item.get("content")
    if isinstance(content, list):
        content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return (item["role"], content) if isinstance(content, str) and content.strip() else None


class CommonMemoSession:
    """An OpenAI Agents `Session` backed by a `Memory` (see the module docstring)."""

    session_settings = None

    def __init__(self, memory: "Memory", session_id: str, *, user_id: Optional[str] = None,
                 agent: Optional[str] = None) -> None:
        self.memory = memory
        self.session_id = session_id
        self.user_id = user_id
        self.agent = agent
        self._session = writer(memory, session_id, user_id, agent)

    def _live(self) -> list[dict[str, Any]]:
        live: list[dict[str, Any]] = []
        for event in self._session.events(kinds=[ExternalEvent.kind]):
            if event.source == ITEM:
                live.append(event.data["item"])
            elif event.source == POP and live:
                live.pop()
            elif event.source == CLEAR:
                live.clear()
        return live

    async def get_items(self, limit: Optional[int] = None) -> list[TResponseInputItem]:
        live = self._live()
        if limit is not None:
            live = live[-limit:] if limit > 0 else []
        return copy.deepcopy(live)  # type: ignore[return-value]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        events: list[Any] = []
        for item in items:
            event = ExternalEvent(source=ITEM, data={"item": copy.deepcopy(item)})
            events.append(event)
            said = _said(item)
            if said is not None:
                events.append(Message(id=f"{event.id}_said", role=said[0], content=said[1]))
        if events:
            self._session.record_many(events)

    async def pop_item(self) -> Optional[TResponseInputItem]:
        live = self._live()
        if not live:
            return None
        self._session.record(ExternalEvent(source=POP))
        return copy.deepcopy(live[-1])  # type: ignore[return-value]

    async def clear_session(self) -> None:
        self._session.record(ExternalEvent(source=CLEAR))
