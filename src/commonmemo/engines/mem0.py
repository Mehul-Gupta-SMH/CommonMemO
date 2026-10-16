"""`Mem0Engine` -- user memory (and memory-backed retrieval) on Mem0.

Mem0 (mem0ai) extracts short natural-language memories ("Prefers email
follow-ups") from conversation with an LLM, deduplicates and updates them
against what it already holds, and retrieves them by vector search. That is
the user tier's job, done well:

| CommonMemo            | Mem0                                                 |
|-----------------------|------------------------------------------------------|
| one closed session    | one `add(messages, user_id=..., metadata={commonmemo_session_id})` |
| a user                | Mem0's `user_id`                                     |
| extracted user facts  | `get_all(filters={"user_id"})`                       |
| episodic retrieval    | `search` -> memories -> the sessions they came from  |
| re-ingest a session   | delete the memories tagged with that session, then add again |
| `forget_user`         | `delete_all(user_id=...)`                            |

What stays local, exactly as in the Graphiti engine: the session digest,
explicit `UserFact`s and provenance, in a `LocalEngine` index next to the
store. Mem0 is reached through a small `Mem0Backend` interface so the
engine's logic is testable without an LLM or a vector store.

Two Mem0 behaviours worth knowing:

- No `run_id` is passed on `add`. Mem0 deduplicates against memories in the
  same scope, and scoping each session separately would make it re-learn
  "prefers email" once per session instead of updating one memory.
- Mem0 *updates* memories in place when new information refines them, so a
  memory's `commonmemo_session_id` is the session that first created it.
  Provenance for a refined memory therefore points at its origin session.

Setup::

    from commonmemo import Memory
    from commonmemo.engines.mem0 import Mem0Engine

    mem = Memory("sqlite:///mem.db", engine=Mem0Engine(config={...}))   # Mem0's own config dict
    mem = Memory("sqlite:///mem.db", engine="mem0")                       # Mem0 defaults (OpenAI)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol, Sequence

from ..events import BOOKKEEPING, MemoryEvent, Message, ToolInvoked, UserFact, describe
from ..stores.base import SessionRecord, Store
from ._ranking import rank_by_hits
from .base import Engine, Episode, Fact
from .local import LocalEngine

SESSION_KEY = "commonmemo_session_id"
_INSTALL_HINT = 'commonmemo: the Mem0 engine needs mem0ai -- pip install "commonmemo[mem0]"'


def mem0_user_id(user_id: str) -> str:
    """Mem0 rejects ids with internal whitespace; such an id is sanitized
    and hash-suffixed so two different ids never share a Mem0 user."""
    if not re.search(r"\s", user_id):
        return user_id
    digest = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    sanitized = re.sub(r"\s+", "_", user_id.strip())
    return f"{sanitized}_{digest}"


def to_messages(session: SessionRecord, events: Sequence[MemoryEvent]) -> list[dict[str, str]]:
    """The session as Mem0 chat messages."""
    out = []
    for e in events:
        if isinstance(e, Message):
            out.append({"role": e.role if e.role in ("user", "assistant", "system") else "user",
                        "content": e.content})
        elif isinstance(e, UserFact):
            out.append({"role": "user", "content": f"My {e.key} is {e.value}."})
        elif isinstance(e, ToolInvoked):
            outcome = f"failed with {e.error}" if e.error else f"returned {e.result}"
            out.append({"role": "assistant", "content": f"(tool {e.name}({e.arguments}) {outcome})"})
        elif isinstance(e, BOOKKEEPING):
            continue  # bookkeeping: noise for extraction, kept in the event tier
        elif describe(e):
            out.append({"role": "user", "content": describe(e)})
        else:
            out.append({"role": "user", "content": f"(event {e.kind}: {e.payload})"})
    return out


@dataclass(frozen=True)
class MemoryHit:
    """One Mem0 memory, as the engine uses it."""

    id: str
    text: str
    session_id: Optional[str]
    created_at: Optional[str] = None
    score: Optional[float] = None


class Mem0Backend(Protocol):
    async def add(self, messages: list[dict[str, str]], *, user_id: str, session_id: str) -> None: ...

    async def search(self, query: str, *, user_id: str, limit: int) -> list[MemoryHit]: ...

    async def list(self, *, user_id: str, limit: int, session_id: Optional[str] = None) -> list[MemoryHit]: ...

    async def delete(self, memory_id: str) -> None: ...

    async def delete_user(self, user_id: str) -> None: ...

    async def close(self) -> None: ...


def _hit(item: dict[str, Any]) -> MemoryHit:
    meta = item.get("metadata") or {}
    return MemoryHit(
        id=item["id"],
        text=item.get("memory", ""),
        session_id=meta.get(SESSION_KEY),
        created_at=item.get("created_at"),
        score=item.get("score"),
    )


class AsyncMemoryBackend:
    """`Mem0Backend` over a live `mem0.AsyncMemory`."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory

    async def add(self, messages: list[dict[str, str]], *, user_id: str, session_id: str) -> None:
        await self.memory.add(messages, user_id=user_id, metadata={SESSION_KEY: session_id})

    async def search(self, query: str, *, user_id: str, limit: int) -> list[MemoryHit]:
        found = await self.memory.search(query, filters={"user_id": user_id}, top_k=limit)
        return [_hit(i) for i in found.get("results", [])]

    async def list(self, *, user_id: str, limit: int, session_id: Optional[str] = None) -> list[MemoryHit]:
        filters: dict[str, Any] = {"user_id": user_id}
        if session_id is not None:
            filters[SESSION_KEY] = session_id
        found = await self.memory.get_all(filters=filters, top_k=limit)
        return [_hit(i) for i in found.get("results", [])]

    async def delete(self, memory_id: str) -> None:
        await self.memory.delete(memory_id)

    async def delete_user(self, user_id: str) -> None:
        await self.memory.delete_all(user_id=user_id)

    async def close(self) -> None:
        close = getattr(self.memory, "close", None)
        if close is not None:
            close()


def _async_memory(config: Optional[dict[str, Any]]) -> Any:
    try:
        from mem0 import AsyncMemory
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return AsyncMemory.from_config(config) if config is not None else AsyncMemory()


class Mem0Engine(Engine):
    """User memory on Mem0, with a local index for the parts that must be
    exact (see the module docstring).

    Pass `memory=` (a `mem0.AsyncMemory`), `config=` (Mem0's config dict) or
    `backend=` (any `Mem0Backend`); with none, Mem0's defaults are used.
    `path` is the local index's SQLite file -- leave it unset inside
    `Memory(...)` and it shares the store's file.
    """

    name = "mem0"

    def __init__(
        self,
        memory: Any = None,
        *,
        config: Optional[dict[str, Any]] = None,
        backend: Optional[Mem0Backend] = None,
        path: Optional[str] = None,
        search_memories: int = 20,
        max_memories: int = 100,
    ) -> None:
        if backend is None:
            backend = AsyncMemoryBackend(memory if memory is not None else _async_memory(config))
        self.backend = backend
        self.search_memories = search_memories
        self.max_memories = max_memories
        self._index: Optional[LocalEngine] = LocalEngine(path) if path is not None else None

    def bind(self, store: Store) -> None:
        if self._index is None:
            self._index = LocalEngine(getattr(store, "path", ":memory:"))

    @property
    def index(self) -> LocalEngine:
        if self._index is None:
            self._index = LocalEngine(":memory:")
        return self._index

    async def ingest(self, session: SessionRecord, events: Sequence[MemoryEvent]) -> list[Episode]:
        episodes = await self.index.ingest(session, events)
        if not episodes or session.user_id is None:
            # Mem0 memory is per user; an anonymous session stays local only.
            return episodes
        uid = mem0_user_id(session.user_id)
        # Idempotent re-ingest: drop what this session produced last time.
        for old in await self.backend.list(user_id=uid, limit=self.max_memories, session_id=session.session_id):
            await self.backend.delete(old.id)
        messages = to_messages(session, events)
        if messages:
            await self.backend.add(messages, user_id=uid, session_id=session.session_id)
        return episodes

    async def _local_episodes(self, user_id: Optional[str], agent: Optional[str]) -> dict[str, Episode]:
        everything = await self.index.search_episodes("", user_id=user_id, agent=agent, limit=1_000_000)
        return {e.session_id: e for e in everything}

    async def search_episodes(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 5,
    ) -> list[Episode]:
        if not query.strip() or user_id is None:
            # Mem0 search must be scoped to a user; unscoped stays local.
            return await self.index.search_episodes(query, user_id=user_id, agent=agent, limit=limit)
        hits = await self.backend.search(query, user_id=mem0_user_id(user_id), limit=self.search_memories)
        local = await self._local_episodes(user_id, agent)
        ranked = rank_by_hits(
            ((h.text, [h.session_id] if h.session_id else []) for h in hits),
            local,
            limit=limit,
            label="Related memories",
        )
        return ranked or await self.index.search_episodes(query, user_id=user_id, agent=agent, limit=limit)

    async def user_facts(self, user_id: str) -> list[Fact]:
        explicit = await self.index.user_facts(user_id)
        local = await self._local_episodes(user_id, None)
        extracted = []
        for hit in await self.backend.list(user_id=mem0_user_id(user_id), limit=self.max_memories):
            origin = local.get(hit.session_id) if hit.session_id else None
            extracted.append(
                Fact(
                    user_id=user_id,
                    key="memory",
                    value=hit.text,
                    valid_from=_timestamp(hit.created_at),
                    source_event_ids=origin.source_event_ids if origin else (),
                )
            )
        # Explicit facts first: they are direct statements and always win.
        return explicit + extracted

    async def forget_user(self, user_id: str) -> int:
        n = await self.index.forget_user(user_id)
        await self.backend.delete_user(mem0_user_id(user_id))
        return n

    async def aclose(self) -> None:
        await self.index.aclose()
        await self.backend.close()


def _timestamp(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


__all__ = ["Mem0Engine", "AsyncMemoryBackend", "Mem0Backend", "MemoryHit", "mem0_user_id", "to_messages"]
