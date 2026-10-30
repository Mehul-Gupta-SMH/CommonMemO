# Framework adapters

An adapter puts CommonMemo behind an agent SDK's own memory or session interface (issue [#6](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/6)), so an
application using that SDK directly, without CommonADK, gets the same memory. Every adapter shares one `Memory`, so a session started under one SDK can be continued
under another. Adapters store and search; they do not push context into a prompt (the hooks and the MCP server do that).

Each is imported by name and needs its SDK (one extra each), and each is tested against that SDK's own installed types and base class.

| SDK | Adapter | Extra | Status |
|---|---|---|---|
| Google ADK | `commonmemo.adapters.adk.CommonMemoMemoryService` | `commonmemo[adk]` | shipped ([#109](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/109)) |
| OpenAI Agents | `Session` over the Log | `commonmemo[openai-agents]` | planned ([#110](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/110)) |
| LangGraph | `BaseStore` | `commonmemo[langgraph]` | planned ([#111](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/111)) |
| AutoGen | `Memory` | `commonmemo[autogen]` | planned ([#112](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/112)) |
| Claude Agent SDK | `SessionStore` | `commonmemo[claude-agent-sdk]` | planned ([#113](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/113)) |
| CrewAI | `StorageBackend` | | deferred: `search` takes an embedding vector, which assumes a vector database |

## Google ADK

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
