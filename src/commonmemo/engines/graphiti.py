"""`GraphitiEngine` -- episodic and user memory on a Graphiti knowledge graph.

Graphiti (graphiti-core) turns text "episodes" into a temporal knowledge
graph: entities, and facts between them that carry when they became true and
when a later episode invalidated them. That maps directly onto CommonMemo's
derived tiers:

| CommonMemo            | Graphiti                                        |
|-----------------------|-------------------------------------------------|
| one closed session    | one episode (`add_episode`, uuid derived from the session id) |
| a user                | one `group_id` (a graph partition)              |
| episodic retrieval    | hybrid `search` -> facts -> the episodes they came from |
| extracted user facts  | the group's currently valid entity edges        |
| `forget_user`         | delete the group                                |

What stays local (a `LocalEngine` index in the same SQLite file as the store):
the per-session digest, explicit `UserFact`s and provenance. That keeps the
engine rules (docs/design.md, "Rules every engine keeps") exact rather than
best-effort: explicit facts are stored verbatim, every episode and fact keeps
its `source_event_ids`, and re-ingesting a session replaces its graph episode
(`remove_episode`, then `add_episode` with the same uuid) instead of
duplicating it.

Graphiti itself is reached through a small `GraphBackend` interface, so the
engine's logic is testable without a graph database or an LLM, and
`graphiti_core` is imported only when the real backend is built.

Setup -- build the Graphiti client yourself (driver, LLM, embedder are
Graphiti's own configuration), then hand it over::

    from graphiti_core import Graphiti
    from commonmemo import Memory
    from commonmemo.engines.graphiti import GraphitiEngine

    mem = Memory("sqlite:///mem.db", engine=GraphitiEngine(graphiti=Graphiti(uri, user, password)))

or `Memory(..., engine="graphiti")`, which builds the client from
`NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid as uuidlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence

from ..events import ExternalEvent, MemoryEvent, Message, ToolInvoked, UserFact
from ..stores.base import SessionRecord, Store
from .base import Engine, Episode, Fact
from .local import LocalEngine

_NAMESPACE = uuidlib.UUID("5f0c2a52-8f43-4d0e-9a0d-6c1e0b7d4a11")
_GROUP_OK = re.compile(r"^[A-Za-z0-9_-]+$")
_INSTALL_HINT = 'commonmemo: the Graphiti engine needs graphiti-core -- pip install "commonmemo[graphiti]"'


def episode_uuid(session_id: str) -> str:
    """The graph episode uuid for a session -- deterministic, so a re-ingest
    targets the same episode and an edge's episode list maps back to it."""
    return str(uuidlib.uuid5(_NAMESPACE, session_id))


def group_id_for(user_id: Optional[str]) -> str:
    """The Graphiti partition for a user. Graphiti accepts only
    `[A-Za-z0-9_-]`; an id with anything else is sanitized and suffixed with
    a hash so two different ids can never share a partition."""
    if user_id is None:
        return "commonmemo_anonymous"
    if _GROUP_OK.match(user_id):
        return f"u_{user_id}"
    digest = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    return f"u_{re.sub(r'[^A-Za-z0-9_-]', '_', user_id)}_{digest}"


def transcript(session: SessionRecord, events: Sequence[MemoryEvent]) -> str:
    """The session as Graphiti episode text: one `speaker: content` line per
    event (Graphiti's `message` episode format)."""
    lines = []
    for e in events:
        if isinstance(e, Message):
            speaker = (e.agent or session.agent or "assistant") if e.role == "assistant" else e.role
            lines.append(f"{speaker}: {e.content}")
        elif isinstance(e, ToolInvoked):
            outcome = f"failed with {e.error}" if e.error else f"returned {e.result}"
            lines.append(f"tool: {e.name}({e.arguments}) {outcome}")
        elif isinstance(e, UserFact):
            lines.append(f"user: my {e.key} is {e.value}")
        elif isinstance(e, ExternalEvent):
            # Framework bookkeeping (token counts, run boundaries) is noise
            # for fact extraction; it stays in the event tier.
            continue
        else:
            lines.append(f"event: {e.kind} {e.payload}")
    return "\n".join(lines)


@dataclass(frozen=True)
class GraphFact:
    """A fact (entity edge) as the graph reports it."""

    name: str
    fact: str
    episode_ids: tuple[str, ...]
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None


class GraphBackend(Protocol):
    """The slice of Graphiti the engine uses."""

    async def add_episode(
        self, *, uuid: str, group_id: str, name: str, body: str, source_description: str,
        reference_time: datetime,
    ) -> None: ...

    async def remove_episode(self, uuid: str) -> bool: ...

    async def search(self, query: str, *, group_ids: Optional[list[str]], limit: int) -> list[GraphFact]: ...

    async def current_facts(self, group_id: str, *, limit: int) -> list[GraphFact]: ...

    async def delete_group(self, group_id: str) -> None: ...

    async def close(self) -> None: ...


def _graph_fact(edge: Any) -> GraphFact:
    return GraphFact(
        name=edge.name,
        fact=edge.fact,
        episode_ids=tuple(edge.episodes or ()),
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at or edge.expired_at,
    )


class GraphitiBackend:
    """`GraphBackend` over a live `graphiti_core.Graphiti` client."""

    def __init__(self, graphiti: Any) -> None:
        self.graphiti = graphiti

    async def add_episode(
        self, *, uuid: str, group_id: str, name: str, body: str, source_description: str,
        reference_time: datetime,
    ) -> None:
        from graphiti_core.nodes import EpisodeType

        await self.graphiti.add_episode(
            name=name,
            episode_body=body,
            source_description=source_description,
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=group_id,
            uuid=uuid,
        )

    async def remove_episode(self, uuid: str) -> bool:
        from graphiti_core.errors import NodeNotFoundError

        try:
            await self.graphiti.remove_episode(uuid)
        except NodeNotFoundError:
            return False
        return True

    async def search(self, query: str, *, group_ids: Optional[list[str]], limit: int) -> list[GraphFact]:
        edges = await self.graphiti.search(query, group_ids=group_ids, num_results=limit)
        return [_graph_fact(e) for e in edges]

    async def current_facts(self, group_id: str, *, limit: int) -> list[GraphFact]:
        from graphiti_core.edges import EntityEdge

        edges = await EntityEdge.get_by_group_ids(self.graphiti.driver, [group_id], limit=limit)
        return [f for f in map(_graph_fact, edges) if f.invalid_at is None]

    async def delete_group(self, group_id: str) -> None:
        from graphiti_core.nodes import Node

        await Node.delete_by_group_id(self.graphiti.driver, group_id)

    async def close(self) -> None:
        await self.graphiti.close()


def _graphiti_from_env() -> Any:
    try:
        from graphiti_core import Graphiti
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    missing = [v for v in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if not os.environ.get(v)]
    if missing:
        raise ValueError(
            f"commonmemo: engine 'graphiti' by name reads its connection from the environment; "
            f"missing: {', '.join(missing)}. Or pass GraphitiEngine(graphiti=Graphiti(...)) yourself."
        )
    return Graphiti(os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])


class GraphitiEngine(Engine):
    """Episodic and user memory on Graphiti, with a local index for the
    parts that must be exact (see the module docstring).

    Pass `graphiti=` (a `graphiti_core.Graphiti`) or `backend=` (any
    `GraphBackend`); with neither, the client is built from the environment.
    `path` is the local index's SQLite file -- leave it unset inside
    `Memory(...)` and it shares the store's file.
    """

    name = "graphiti"

    def __init__(
        self,
        graphiti: Any = None,
        *,
        backend: Optional[GraphBackend] = None,
        path: Optional[str] = None,
        search_facts: int = 20,
        max_extracted_facts: int = 50,
    ) -> None:
        if backend is None:
            backend = GraphitiBackend(graphiti if graphiti is not None else _graphiti_from_env())
        self.backend = backend
        self.search_facts = search_facts
        self.max_extracted_facts = max_extracted_facts
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
        if not episodes:
            return []
        body = transcript(session, events)
        uid = episode_uuid(session.session_id)
        await self.backend.remove_episode(uid)  # idempotent re-ingest
        if body:
            await self.backend.add_episode(
                uuid=uid,
                group_id=group_id_for(session.user_id),
                name=f"commonmemo session {session.session_id}",
                body=body,
                source_description=f"commonmemo session {session.session_id}"
                + (f" with agent {session.agent}" if session.agent else ""),
                reference_time=datetime.fromtimestamp(max(e.ts for e in events), tz=timezone.utc),
            )
        return episodes

    async def _local_episodes(self, user_id: Optional[str], agent: Optional[str]) -> dict[str, Episode]:
        everything = await self.index.search_episodes("", user_id=user_id, agent=agent, limit=1_000_000)
        return {episode_uuid(e.session_id): e for e in everything}

    async def search_episodes(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 5,
    ) -> list[Episode]:
        if not query.strip():
            return await self.index.search_episodes(query, user_id=user_id, agent=agent, limit=limit)
        group_ids = [group_id_for(user_id)] if user_id is not None else None
        facts = await self.backend.search(query, group_ids=group_ids, limit=self.search_facts)
        local = await self._local_episodes(user_id, agent)

        order: list[str] = []
        related: dict[str, list[str]] = {}
        for fact in facts:  # best fact first -> its episodes rank first
            for uid in fact.episode_ids:
                if uid not in local:
                    continue
                if uid not in related:
                    order.append(uid)
                    related[uid] = []
                if fact.fact not in related[uid]:
                    related[uid].append(fact.fact)
        if not order:
            return await self.index.search_episodes(query, user_id=user_id, agent=agent, limit=limit)
        out = []
        for rank, uid in enumerate(order[:limit]):
            ep = local[uid]
            out.append(
                replace(
                    ep,
                    summary=f"{ep.summary} Related facts: {'; '.join(related[uid])}.",
                    score=1.0 / (rank + 1),
                )
            )
        return out

    async def user_facts(self, user_id: str) -> list[Fact]:
        explicit = await self.index.user_facts(user_id)
        local = await self._local_episodes(user_id, None)
        seen = {(f.key, f.value) for f in explicit}
        extracted = []
        for gf in await self.backend.current_facts(group_id_for(user_id), limit=self.max_extracted_facts):
            if (gf.name, gf.fact) in seen:
                continue
            seen.add((gf.name, gf.fact))
            # Provenance through the episodes the fact came from.
            sources = tuple(
                eid for uid in gf.episode_ids if uid in local for eid in local[uid].source_event_ids
            )
            valid_from = gf.valid_at.timestamp() if gf.valid_at else 0.0
            extracted.append(
                Fact(user_id=user_id, key=gf.name, value=gf.fact, valid_from=valid_from, source_event_ids=sources)
            )
        # Explicit facts first: they are direct statements and always win.
        return explicit + sorted(extracted, key=lambda f: (f.key, f.value))

    async def forget_user(self, user_id: str) -> int:
        n = await self.index.forget_user(user_id)
        await self.backend.delete_group(group_id_for(user_id))
        return n

    async def aclose(self) -> None:
        await self.index.aclose()
        await self.backend.close()


__all__ = [
    "GraphitiEngine",
    "GraphitiBackend",
    "GraphBackend",
    "GraphFact",
    "episode_uuid",
    "group_id_for",
    "transcript",
]
