# Framework adapters

An adapter puts CommonMemo behind an agent SDK's own memory or session interface (issue [#6](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/6)), so an
application using that SDK directly, without CommonADK, gets the same memory. Every adapter shares one `Memory`, so a session started under one SDK can be continued
under another. Adapters store and search; they do not push context into a prompt (the hooks and the MCP server do that).

Each is imported by name and needs its SDK (one extra each), and each is tested against that SDK's own installed types and base class.

| SDK | Adapter | Extra | Status |
|---|---|---|---|
| Google ADK | `commonmemo.adapters.adk.CommonMemoMemoryService` | `commonmemo[adk]` | shipped ([#109](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/109)) |
| OpenAI Agents | `commonmemo.adapters.openai_agents.CommonMemoSession` | `commonmemo[openai-agents]` | shipped ([#110](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/110)) |
| LangGraph | `commonmemo.adapters.langgraph.CommonMemoStore` | `commonmemo[langgraph]` | shipped ([#111](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/111)) |
| AutoGen | `Memory` | `commonmemo[autogen]` | planned ([#112](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/112)) |
| Claude Agent SDK | `SessionStore` | `commonmemo[claude-agent-sdk]` | planned ([#113](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/113)) |
| CrewAI | `StorageBackend` | | deferred: `search` takes an embedding vector, which assumes a vector database |

## Google ADK

Needs Python 3.11 or later (google-adk 2 does).

```python
from commonmemo import Memory
from commonmemo.adapters.adk import CommonMemoMemoryService

runner = Runner(agent=agent, app_name="shop", session_service=sessions,
                memory_service=CommonMemoMemoryService(Memory("sqlite:///mem.db")))
```

| ADK call | In CommonMemo |
|---|---|
| `add_session_to_memory(session)` | each event's text becomes a `Message` (`user` or `assistant`) in the ADK session, resumed by its id, with the ADK app as the agent |
| `add_events_to_memory(...)` | the same, for a delta |
| `add_memory(...)` | each `MemoryEntry` becomes a `UserFact` (key: the entry's id, or a hash of its text) |
| `search_memory(app_name, user_id, query)` | what `memory_recall` finds, as `MemoryEntry`s with the kind in `custom_metadata["kind"]` |

Ids come from the ADK event id, so adding a session again records nothing new.

What `BaseMemoryService` cannot express:
- It has no delete or correction. Use `Memory.forget_user` and `revert`.
- It has no approval step. An agent's world-fact proposals still wait for review.
- A search is scoped to `(app_name, user_id)`, so the recaps it returns are the user's across agents.
- Events without text (function calls, state changes) are not stored. Tool memory comes from `@mem.track` or the hooks.

## OpenAI Agents

```python
from commonmemo import Memory
from commonmemo.adapters.openai_agents import CommonMemoSession

session = CommonMemoSession(Memory("sqlite:///mem.db"), "ada-chat", user_id="ada")
await Runner.run(agent, "My daughter is allergic to peanuts.", session=session)
```

| `Session` call | In CommonMemo |
|---|---|
| `add_items(items)` | each item kept verbatim as an `ExternalEvent`; a user or assistant item with text also becomes a `Message`, so it is searchable |
| `get_items(limit)` | the live items in order, exactly as added; `limit` keeps the latest N |
| `pop_item()` | a compensating `openai_agents.pop` event; the item stops being returned |
| `clear_session()` | a compensating `openai_agents.clear` event |

The test suite runs the same calls on the SDK's own `SQLiteSession` and requires the same answers.

What `Session` cannot express:
- The Log is append-only, so `pop_item` and `clear_session` hide items rather than delete them. The `Message` made from a popped or cleared item stays searchable as something the user said; `Memory.forget_user` removes it.
- Items are the SDK's own dicts and are returned untouched; CommonMemo does not interpret function calls or reasoning items. Tool memory comes from `@mem.track` or the hooks.
- A session id belongs to one user: resuming it as another raises `PermissionError`.

## LangGraph

```python
from commonmemo import Memory
from commonmemo.adapters.langgraph import CommonMemoStore

graph = builder.compile(store=CommonMemoStore(Memory("sqlite:///mem.db")))
```

| `BaseStore` call | In CommonMemo |
|---|---|
| `put(namespace, key, value)` | an `ExternalEvent(source="langgraph.put")`; every version stays in the Log |
| `delete` / `put(..., None)` | an `ExternalEvent(source="langgraph.delete")` |
| `get`, `search`, `list_namespaces` | the Log replayed in order, with LangGraph's own semantics for namespace prefixes, filter operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`), wildcards and `max_depth` |
| `search(query=...)` | ranked by the memory's ranker over the value's text (keywords, or meaning when the memory has an embedder) |
| `batch` / `abatch` | puts in a batch are written in one transaction; a read after a put in the same batch sees it |

The test suite runs the same calls on LangGraph's `InMemoryStore` and requires the same answers, and runs a compiled graph over two separate memories on one Log. Two processes on one Log see each other's writes on the next call.

What `BaseStore` cannot express here:
- `ttl` and `index` are accepted and ignored: items do not expire, and every value's text is searchable.
- The values are LangGraph's own key-value memory kept in the Log; they are not turned into CommonMemo facts, so `memory_recall` does not find them.
- One `session_id` is one store; stores with different ids on one Log are separate.
- A namespace whose last item was deleted stops being listed (as in LangGraph's Postgres store; `InMemoryStore` keeps listing it).

## Attaching everything at once: `commonmemo.instrument` (#116)

```python
import commonmemo
commonmemo.instrument(memory)                       # every installed SDK
commonmemo.instrument(memory, ["adk"], user_id=lambda: current_user())
commonmemo.uninstrument()
```

It patches each SDK's *defaults* so the adapter above is used where the
application did not pick a memory; anything passed explicitly wins. Nothing
happens on import, an SDK that is not installed is skipped, and a hook that
has moved in a newer SDK warns and is skipped rather than raising.

| SDK | Default patched |
|---|---|
| Google ADK | `Runner(memory_service=None)` |
| OpenAI Agents | `Runner.run / run_sync / run_streamed` without `session=` (string/list input only) |
| LangGraph | `StateGraph.compile(store=None)` |

`user_id` and `session_id` are a string or a callable read at each use. Without
`user_id` everything belongs to one anonymous user, which is right for a
single-user agent and wrong for a multi-user one: pass a resolver. The OpenAI
Agents SDK supplies no session id, so runs share `session_id` (default
`"auto"`) and therefore one conversation. ADK takes its user and session ids
from the framework. Like the adapters, it stores and searches; it does not
push context into a prompt. AutoGen and Claude Agent SDK hooks arrive with
their adapters.
