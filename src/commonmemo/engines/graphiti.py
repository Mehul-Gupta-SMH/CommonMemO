"""`GraphitiEngine` -- Recaps and the UserProfile on a Graphiti knowledge graph.

Graphiti (graphiti-core) turns text "episodes" into a temporal knowledge
graph: entities, and facts between them that carry when they became true and
when a later episode invalidated them. That maps directly onto CommonMemo's
derived tiers:

| CommonMemo            | Graphiti                                        |
|-----------------------|-------------------------------------------------|
| one closed session    | one episode (`add_episode`; its uuid is recorded in the local index) |
| a user                | one `group_id` (a graph partition)              |
| recap retrieval       | hybrid `search` -> facts -> the recaps they came from |
| extracted profile facts | the group's currently valid entity edges      |
| `forget_user`         | delete the group                                |

What stays local (a `LocalEngine` index in the same SQLite file as the store):
the per-session digest, explicit `UserFact`s and provenance. That keeps the
engine rules (docs/design.md, "Rules every engine keeps") exact rather than
best-effort: explicit facts are stored verbatim, every recap and fact keeps
its `source_event_ids`, and re-ingesting a session replaces its graph episode
(`remove_episode` on the recorded uuid, then a fresh `add_episode`) instead
of duplicating it.

Graphiti assigns episode uuids itself: `add_episode(uuid=...)` does not
create an episode with that uuid, it *reprocesses an existing* one and
raises `NodeNotFoundError` when there is none. So the engine never passes a
uuid; it records the one Graphiti returns (`LocalEngine.set_external_id`)
and maps graph facts back to sessions through it.

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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence

from ..events import BOOKKEEPING, MemoryEvent, Message, ToolInvoked, UserFact, describe
from ..ranking import EmbeddingRanker
from ..stores.base import SessionRecord, Store
from ._ranking import rank_by_hits
from .base import Engine, Recap, Fact
from .local import LocalEngine

SYSTEM = "graphiti"
"""The local index's name for Graphiti episode ids."""
_GROUP_OK = re.compile(r"^[A-Za-z0-9_-]+$")
_INSTALL_HINT = 'commonmemo: the Graphiti engine needs graphiti-core -- pip install "commonmemo[graphiti]"'


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
        elif isinstance(e, BOOKKEEPING):
            # Framework and step bookkeeping (token counts, run boundaries,
            # tombstones) is noise for extraction; it stays in the Log.
            continue
        elif describe(e):
            lines.append(f"user: {describe(e)}")
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
        self, *, group_id: str, name: str, body: str, source_description: str,
        reference_time: datetime,
    ) -> str:
        """Create an episode; return the uuid the graph assigned it."""

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
        self, *, group_id: str, name: str, body: str, source_description: str,
        reference_time: datetime,
    ) -> str:
        from graphiti_core.nodes import EpisodeType

        result = await self.graphiti.add_episode(
            name=name,
            episode_body=body,
            source_description=source_description,
            reference_time=reference_time,
            source=EpisodeType.message,
            group_id=group_id,
        )
        return result.episode.uuid

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
    """Recaps and the UserProfile on Graphiti, with a local index for the
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

    def ranker(self) -> Optional[EmbeddingRanker]:
        """Rank facts with the embedder Graphiti already uses."""
        embedder = getattr(getattr(self.backend, "graphiti", None), "embedder", None)
        return EmbeddingRanker(embedder.create_batch) if embedder is not None else None

    @property
    def index(self) -> LocalEngine:
        if self._index is None:
            self._index = LocalEngine(":memory:")
        return self._index

    async def ingest(self, session: SessionRecord, events: Sequence[MemoryEvent]) -> list[Recap]:
        recaps = await self.index.ingest(session, events)
        if not recaps:
            return []
        body = transcript(session, events)
        previous = self.index.external_id(session.session_id, SYSTEM)
        if previous is not None:
            await self.backend.remove_episode(previous)  # idempotent re-ingest
            self.index.delete_external_id(session.session_id, SYSTEM)
        if body:
            uid = await self.backend.add_episode(
                group_id=group_id_for(session.user_id),
                name=f"commonmemo session {session.session_id}",
                body=body,
                source_description=f"commonmemo session {session.session_id}"
                + (f" with agent {session.agent}" if session.agent else ""),
                reference_time=datetime.fromtimestamp(max(e.ts for e in events), tz=timezone.utc),
            )
            self.index.set_external_id(session.session_id, SYSTEM, uid)
        return recaps

    async def _local_recaps(self, user_id: Optional[str], agent: Optional[str]) -> dict[str, Recap]:
        """Local recaps keyed by their Graphiti episode uuid."""
        everything = await self.index.search_recaps("", user_id=user_id, agent=agent, limit=1_000_000)
        by_session = {e.session_id: e for e in everything}
        graph_ids = self.index.external_ids(SYSTEM, list(by_session))
        return {uid: by_session[sid] for uid, sid in graph_ids.items()}

    async def search_recaps(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        limit: int = 5,
    ) -> list[Recap]:
        if not query.strip():
            return await self.index.search_recaps(query, user_id=user_id, agent=agent, limit=limit)
        group_ids = [group_id_for(user_id)] if user_id is not None else None
        facts = await self.backend.search(query, group_ids=group_ids, limit=self.search_facts)
        local = await self._local_recaps(user_id, agent)
        ranked = rank_by_hits(
            ((f.fact, f.episode_ids) for f in facts), local, limit=limit, label="Related facts"
        )
        return ranked or await self.index.search_recaps(query, user_id=user_id, agent=agent, limit=limit)

    async def user_profile(self, user_id: str) -> list[Fact]:
        explicit = await self.index.user_profile(user_id)
        local = await self._local_recaps(user_id, None)
        seen = {(f.key, f.value) for f in explicit}
        extracted = []
        for gf in await self.backend.current_facts(group_id_for(user_id), limit=self.max_extracted_facts):
            if (gf.name, gf.fact) in seen:
                continue
            seen.add((gf.name, gf.fact))
            # Provenance through the recaps the fact came from.
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
    "SYSTEM",
    "group_id_for",
    "transcript",
]
