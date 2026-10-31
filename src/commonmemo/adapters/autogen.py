"""AutoGen: `autogen_core.memory.Memory` over CommonMemo.

    from commonmemo import Memory
    from commonmemo.adapters.autogen import CommonMemoMemory

    agent = AssistantAgent("helper", model_client=client,
                           memory=[CommonMemoMemory(Memory("sqlite:///mem.db"), user_id="ana")])

How AutoGen's calls land in the Log:

- `add(content)`: a `UserFact` for the user (text as is, a dict as JSON; `metadata["key"]` names the fact,
  otherwise the key is a hash of the text, so adding the same thing twice is one fact). Binary and image
  content is refused.
- `query(text)`: what `memory_recall` finds for the text -- facts about the user, world facts, feedback,
  lessons, decisions, recaps of earlier sessions, and windows of what the user said -- as text
  `MemoryContent`s, best first, the kind in `metadata["kind"]`.
- `update_context(model_context)`: AutoGen's own hook for putting memory in the prompt. It queries with the
  last user message and appends one system message listing what was found, as `ListMemory` does. Pass
  `inject=False` to keep the adapter to store-and-search only.
- `clear()`: forgets the configured user (`Memory.forget_user`); it refuses without a `user_id`.

What `Memory` cannot express: AutoGen never reports the conversation to a memory, so turns are not recorded
by this adapter (use `Memory.session` and the hooks for that); it has no per-entry delete or correction (use
`Memory.forget_user`, `revert`); and a query carries no user, so one `CommonMemoMemory` serves one user.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

try:
    from autogen_core import CancellationToken
    from autogen_core.memory import Memory as AutoGenMemory
    from autogen_core.memory import MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
    from autogen_core.model_context import ChatCompletionContext
    from autogen_core.models import SystemMessage, UserMessage
except ImportError as exc:  # pragma: no cover -- exercised only without autogen
    raise ImportError('commonmemo: the AutoGen adapter needs autogen-agentchat -- pip install "commonmemo[autogen]"') from exc

from ..agent_tools import recall_entries
from ..events import UserFact
from ._core import reader, stable_id, writer

if TYPE_CHECKING:
    from ..memory import Memory


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True)
    raise TypeError(f"CommonMemoMemory stores text and JSON content, not {type(content).__name__}")


class CommonMemoMemory(AutoGenMemory):
    """An AutoGen memory backed by a `Memory`, for one user (see the module docstring)."""

    def __init__(self, memory: "Memory", *, user_id: Optional[str] = None, agent: Optional[str] = None,
                 max_results: int = 5, inject: bool = True, name: str = "commonmemo") -> None:
        self.memory = memory
        self.user_id = user_id
        self.agent = agent
        self.max_results = max_results
        self.inject = inject
        self.name = name

    async def add(self, content: MemoryContent, cancellation_token: Optional[CancellationToken] = None) -> None:
        text = _text(content.content)
        key = (content.metadata or {}).get("key") or stable_id("autogen-memory", text)
        fact = UserFact(id=stable_id("autogen-memory", self.agent or "", self.user_id or "", str(key), text),
                        key=str(key), value=text)
        written = writer(self.memory, f"adapter-memory:{self.agent or ''}:{self.user_id or ''}", self.user_id, self.agent)
        written.record_many([fact])
        await written.aclose()  # a fact reaches the user's profile when its session closes

    async def query(self, query: "str | MemoryContent", cancellation_token: Optional[CancellationToken] = None,
                    **kwargs: Any) -> MemoryQueryResult:
        text = query if isinstance(query, str) else _text(query.content)
        lines, _, _ = recall_entries(reader(self.memory, self.user_id, self.agent), text)
        return MemoryQueryResult(results=[
            MemoryContent(content=line, mime_type=MemoryMimeType.TEXT, metadata={"kind": label})
            for label, line in lines[: self.max_results]
        ])

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        if not self.inject:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))
        messages = await model_context.get_messages()
        last = next((m.content for m in reversed(messages) if isinstance(m, UserMessage) and isinstance(m.content, str)), None)
        found = await self.query(last) if last else MemoryQueryResult(results=[])
        if found.results:
            listing = "\n".join(f"{i}. [{r.metadata['kind']}] {r.content}" for i, r in enumerate(found.results, 1))
            await model_context.add_message(SystemMessage(content=f"\nRelevant memory content:\n{listing}"))
        return UpdateContextResult(memories=found)

    async def clear(self) -> None:
        if self.user_id is None:
            raise ValueError("CommonMemoMemory.clear forgets one user; construct it with user_id=")
        await self.memory.forget_user(self.user_id)

    async def close(self) -> None:
        return None
