"""LangGraph: `BaseStore` over CommonMemo.

    from commonmemo import Memory
    from commonmemo.adapters.langgraph import CommonMemoStore

    store = CommonMemoStore(Memory("sqlite:///mem.db"))
    graph = builder.compile(store=store)

How the store's calls land in the Log:

- `put`: `ExternalEvent(source="langgraph.put")` with the namespace, key and value; `put(..., None)` and
  `delete` are `langgraph.delete` events. Nothing is overwritten: the Log keeps every version, and
  `Memory.history` / `revert`'s tools can see them. The store's current contents are the Log replayed in
  order, so two processes on one Log see each other's writes (each call first reads what was appended
  since the last).
- `get`, `search` and `list_namespaces` read that state with LangGraph's own semantics: namespace prefix,
  the `$eq` / `$ne` / `$gt` / `$gte` / `$lt` / `$lte` filter operators, wildcard and `max_depth` listing. A namespace whose last item was deleted stops being listed (as in
  LangGraph's Postgres store; `InMemoryStore` keeps listing it).
  The tests run the same calls on LangGraph's `InMemoryStore` and require the same answers.
- `search(query=...)` ranks the items by the memory's ranker (keywords, or meaning when the memory has an
  embedder) over the value's text. Without a query, items come in the order they were first stored.

One `session_id` is one store: stores with different ids on the same Log are separate, and a store
reads only what its own id wrote.

What `BaseStore` cannot express here: `ttl` and `index` are accepted and ignored (items do not expire, and
every value's text is searchable); values are not turned into CommonMemo facts, so `memory_recall` does not
find them -- the store is LangGraph's own key-value memory, kept durably in the Log. `Item.created_at` is
when the key was first put; `updated_at` is the last put.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Optional

try:
    from langgraph.store.base import (
        BaseStore, GetOp, Item, ListNamespacesOp, MatchCondition, Op, PutOp, Result, SearchItem, SearchOp,
    )
except ImportError as exc:  # pragma: no cover -- exercised only without langgraph
    raise ImportError(
        'commonmemo: the LangGraph adapter needs langgraph -- pip install "commonmemo[langgraph]"'
    ) from exc

from ..agent_tools import run_sync
from ..events import ExternalEvent
from ._core import writer

if TYPE_CHECKING:
    from ..memory import Memory

PUT, DELETE = "langgraph.put", "langgraph.delete"
Namespace = tuple[str, ...]


def _time(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _matches(value: Any, wanted: Any) -> bool:
    """LangGraph's filter semantics: a dict of operators, a nested dict, a list, or equality."""
    if isinstance(wanted, dict):
        if any(k.startswith("$") for k in wanted):
            return all(_operator(value, op, arg) for op, arg in wanted.items())
        return isinstance(value, dict) and all(_matches(value.get(k), v) for k, v in wanted.items())
    if isinstance(wanted, (list, tuple)):
        return (isinstance(value, (list, tuple)) and len(value) == len(wanted)
                and all(_matches(a, b) for a, b in zip(value, wanted)))
    return value == wanted


def _operator(value: Any, op: str, arg: Any) -> bool:
    if op == "$eq":
        return value == arg
    if op == "$ne":
        return value != arg
    if op in ("$gt", "$gte", "$lt", "$lte"):
        a, b = float(value), float(arg)
        return {"$gt": a > b, "$gte": a >= b, "$lt": a < b, "$lte": a <= b}[op]
    raise ValueError(f"Unsupported operator: {op}")


def _namespace_matches(condition: MatchCondition, namespace: Namespace) -> bool:
    path = condition.path
    if len(namespace) < len(path):
        return False
    if condition.match_type == "prefix":
        pairs = zip(namespace, path)
    elif condition.match_type == "suffix":
        pairs = zip(reversed(namespace), reversed(path))
    else:
        raise ValueError(f"Unsupported match type: {condition.match_type}")
    return all(p == "*" or n == p for n, p in pairs)


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return "" if value is None else str(value)


class CommonMemoStore(BaseStore):
    """A LangGraph `BaseStore` backed by a `Memory` (see the module docstring)."""

    supports_ttl = False

    def __init__(self, memory: "Memory", *, session_id: str = "langgraph-store",
                 user_id: Optional[str] = None, agent: Optional[str] = None) -> None:
        self.memory = memory
        self.session_id = session_id
        self._session = writer(memory, session_id, user_id, agent)
        self._data: dict[Namespace, dict[str, Item]] = {}
        self._seen = 0  # the log position this view has read up to

    # -- the state: the Log, replayed ---------------------------------------

    def _sync(self) -> None:
        for seq, event in self.memory.store.events_after(self._seen):
            self._seen = seq
            if not isinstance(event, ExternalEvent) or event.session_id != self.session_id:
                continue
            namespace, key = tuple(event.data["namespace"]), event.data["key"]
            if event.source == PUT:
                items = self._data.setdefault(namespace, {})
                created = items[key].created_at if key in items else _time(event.ts)
                items[key] = Item(value=event.data["value"], key=key, namespace=namespace,
                                  created_at=created, updated_at=_time(event.ts))
            elif event.source == DELETE and namespace in self._data:
                self._data[namespace].pop(key, None)
                if not self._data[namespace]:
                    del self._data[namespace]

    def _flush(self, pending: list[ExternalEvent]) -> None:
        if pending:
            self._session.record_many(pending)
            pending.clear()
        self._sync()

    # -- BaseStore ----------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return run_sync(self.abatch(ops))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        pending: list[ExternalEvent] = []
        self._sync()
        for op in ops:
            if isinstance(op, PutOp):
                namespace = list(op.namespace)
                if op.value is None:
                    pending.append(ExternalEvent(source=DELETE, data={"namespace": namespace, "key": op.key}))
                else:
                    pending.append(ExternalEvent(source=PUT, data={
                        "namespace": namespace, "key": op.key, "value": dict(op.value)}))
                results.append(None)
                continue
            self._flush(pending)  # a read sees every put before it in the batch
            if isinstance(op, GetOp):
                results.append(self._data.get(tuple(op.namespace), {}).get(op.key))
            elif isinstance(op, SearchOp):
                results.append(await self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._namespaces(op))
            else:  # pragma: no cover -- a future op type
                raise ValueError(f"Unknown operation type: {type(op)}")
        self._flush(pending)
        return results

    async def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        found = [item for namespace, items in self._data.items() if namespace[: len(prefix)] == prefix
                 for item in items.values()
                 if not op.filter or all(_matches(item.value.get(k), v) for k, v in op.filter.items())]
        scores: list[Optional[float]] = [None] * len(found)
        if op.query and found:
            scores = list(await self.memory.ranker.score(op.query, [_text(i.value) for i in found]))
            order = sorted(range(len(found)), key=lambda n: (-scores[n], n))  # type: ignore[operator]
            found, scores = [found[n] for n in order], [scores[n] for n in order]
        window = slice(op.offset, op.offset + op.limit)
        return [SearchItem(namespace=i.namespace, key=i.key, value=i.value, created_at=i.created_at,
                           updated_at=i.updated_at, score=s) for i, s in zip(found[window], scores[window])]

    def _namespaces(self, op: ListNamespacesOp) -> list[Namespace]:
        found = list(self._data)
        if op.match_conditions:
            found = [n for n in found if all(_namespace_matches(c, n) for c in op.match_conditions)]
        if op.max_depth is not None:
            found = list({n[: op.max_depth] for n in found})
        return sorted(found)[op.offset: op.offset + op.limit]
