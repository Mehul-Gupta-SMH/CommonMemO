"""Claude Agent SDK: `SessionStore` over CommonMemo.

    from claude_agent_sdk import ClaudeAgentOptions, query
    from commonmemo import Memory
    from commonmemo.adapters.claude_agent_sdk import CommonMemoSessionStore

    options = ClaudeAgentOptions(session_store=CommonMemoSessionStore(Memory("sqlite:///mem.db")))

How the store's calls land in the Log:

- `append`: one `ExternalEvent(source="claude_agent_sdk.append")` with the key and the batch of
  transcript entries. An entry whose `uuid` is already stored is skipped, as the protocol requires
  (the SDK retries and replays batches); entries without a `uuid` are always appended. `delete` is a
  `claude_agent_sdk.delete` event; deleting a session also drops its subagent transcripts. Nothing is
  overwritten: the Log keeps every batch. The store's current contents are the Log replayed in order,
  so two processes on one Log see each other's writes.
- `load`, `list_sessions`, `list_session_summaries` and `list_subkeys` read that state with the SDK's
  own semantics. The tests run the same calls on the SDK's `InMemorySessionStore` and require the same
  answers.
- The text of `user` and `assistant` entries on a main transcript is also recorded as `Message`s under
  the SDK session's id, so `memory_recall` finds what was said in earlier Claude Agent SDK runs. Those
  messages stay in the Log when a session is deleted from the store; use `Memory.forget_user` to erase.

One `session_id` is one store: stores with different ids on the same Log are separate, and a store reads
only what its own id wrote.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

try:
    from claude_agent_sdk import SessionStore, fold_session_summary
except ImportError as exc:  # pragma: no cover -- exercised only without claude-agent-sdk
    raise ImportError(
        'commonmemo: the Claude Agent SDK adapter needs claude-agent-sdk -- pip install "commonmemo[claude-agent-sdk]"'
    ) from exc

from ..events import ExternalEvent, Message
from ._core import stable_id, writer

if TYPE_CHECKING:
    from ..memory import Memory

APPEND, DELETE = "claude_agent_sdk.append", "claude_agent_sdk.delete"
Key = tuple[str, str, Optional[str]]


def _key(key: Any) -> Key:
    return key["project_key"], key["session_id"], key.get("subpath")


def _wire(key: Key) -> dict[str, Any]:
    project, session, subpath = key
    return {"project_key": project, "session_id": session, "subpath": subpath}


def _text(entry: Any) -> str:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
                         and isinstance(b.get("text"), str))
    return ""


class CommonMemoSessionStore(SessionStore):
    """A Claude Agent SDK `SessionStore` backed by a `Memory` (see the module docstring)."""

    def __init__(self, memory: "Memory", *, session_id: str = "claude-agent-sdk-store",
                 user_id: Optional[str] = None, agent: Optional[str] = None) -> None:
        self.memory = memory
        self.session_id = session_id
        self.user_id = user_id
        self.agent = agent
        self._session = writer(memory, session_id, user_id, agent)
        self._entries: dict[Key, list[Any]] = {}
        self._uuids: dict[Key, set[str]] = {}
        self._mtimes: dict[Key, int] = {}
        self._summaries: dict[tuple[str, str], Any] = {}
        self._seen = 0  # the log position this view has read up to

    # -- the state: the Log, replayed ---------------------------------------

    def _apply_append(self, key: Key, entries: list[Any], mtime: int) -> list[Any]:
        seen = self._uuids.setdefault(key, set())
        fresh = []
        for entry in entries:
            uuid = entry.get("uuid")
            if uuid is not None:
                if uuid in seen:
                    continue
                seen.add(uuid)
            fresh.append(entry)
        self._entries.setdefault(key, []).extend(fresh)
        self._mtimes[key] = max(mtime, self._mtimes.get(key, 0) + 1)
        if key[2] is None:
            self._summaries[key[:2]] = {**fold_session_summary(self._summaries.get(key[:2]), _wire(key), fresh),
                                        "mtime": self._mtimes[key]}
        return fresh

    def _apply_delete(self, key: Key) -> None:
        doomed = [k for k in self._entries if k == key or (key[2] is None and k[:2] == key[:2])]
        for k in doomed:
            for table in (self._entries, self._uuids, self._mtimes):
                table.pop(k, None)
        if key[2] is None:
            self._summaries.pop(key[:2], None)

    def _sync(self) -> None:
        for seq, event in self.memory.store.events_after(self._seen):
            self._seen = seq
            if not isinstance(event, ExternalEvent) or event.session_id != self.session_id:
                continue
            key = _key(event.data["key"])
            if event.source == APPEND:
                self._apply_append(key, event.data["entries"], int(event.ts * 1000))
            elif event.source == DELETE:
                self._apply_delete(key)

    # -- SessionStore -------------------------------------------------------

    async def append(self, key: Any, entries: list[Any]) -> None:
        self._sync()
        k = _key(key)
        seen = self._uuids.get(k, set())
        batch: list[Any] = []
        for entry in entries:  # skip what is stored, and repeats inside the batch
            uuid = entry.get("uuid")
            if uuid is None or (uuid not in seen and uuid not in {e.get("uuid") for e in batch}):
                batch.append(entry)
        if not batch:
            return
        self._session.record(ExternalEvent(source=APPEND, data={"key": _wire(k), "entries": batch}))
        if k[2] is None:
            self._record_messages(k, batch)
        self._sync()

    def _record_messages(self, key: Key, entries: list[Any]) -> None:
        messages = []
        for entry in entries:
            role, text = entry.get("type"), _text(entry)
            if role in ("user", "assistant") and text and entry.get("uuid"):
                messages.append(Message(id=stable_id("claude-agent-sdk", key[0], key[1], entry["uuid"]),
                                        role=role, content=text))
        if messages:
            writer(self.memory, key[1], self.user_id, self.agent).record_many(messages)

    async def load(self, key: Any) -> Optional[list[Any]]:
        self._sync()
        entries = self._entries.get(_key(key))
        return None if entries is None else list(entries)

    async def list_sessions(self, project_key: str) -> list[Any]:
        self._sync()
        return [{"session_id": s, "mtime": self._mtimes[(p, s, None)]}
                for (p, s, sub) in self._entries if p == project_key and sub is None]

    async def list_session_summaries(self, project_key: str) -> list[Any]:
        self._sync()
        return [s for (p, _), s in self._summaries.items() if p == project_key]

    async def delete(self, key: Any) -> None:
        self._sync()
        k = _key(key)
        if any(x == k or (k[2] is None and x[:2] == k[:2]) for x in self._entries):
            self._session.record(ExternalEvent(source=DELETE, data={"key": _wire(k)}))
            self._sync()

    async def list_subkeys(self, key: Any) -> list[str]:
        self._sync()
        return [sub for (p, s, sub) in self._entries
                if (p, s) == (key["project_key"], key["session_id"]) and sub is not None]
