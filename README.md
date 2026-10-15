# CommonMemo

Framework-neutral memory for agents and workflows. Four tiers behind one
`Memory` handle, one install, and a decorator or a subclass to get events in
-- everything else (session state, consolidation into episodes, user facts,
retrieval, context for the next turn) is built in.

| Tier | Holds | Owned by |
|---|---|---|
| **Event** | append-only log of what happened: messages, tool calls, your domain events | CommonMemo (SQLite) |
| **Session** | live working state of one conversation or workflow run, resumable by id | CommonMemo (SQLite) |
| **Episodic** | consolidated summaries of past sessions, retrieved by relevance | a wrapped engine |
| **User** | durable facts about a user; a newer value supersedes, history is kept | a wrapped engine |

CommonMemo **wraps** memory engines rather than building one: extraction,
retrieval and their optimizations belong to engines that specialize in them
(Graphiti, Mem0 -- planned). The built-in `local` engine is offline and
deterministic, so everything works out of the box and in tests.

**Status:** v0.0.1 -- the design and the skeleton. See
[`docs/design.md`](docs/design.md) for the design and the roadmap.

## Install

```bash
pip install commonmemo                 # core: stdlib only, local engine
pip install "commonmemo[commonadk]"    # + CommonADK integration
```

## Use

```python
from commonmemo import Memory, MemoryEvent

mem = Memory("sqlite:///mem.db")       # one path is the whole setup

# 1. Decorator: every call becomes an event (arguments, result/error, duration)
@mem.track
def lookup_order(order_id: str) -> str: ...

# 2. Subclass: your own domain events -- no @dataclass needed
class TicketOpened(MemoryEvent):
    ticket: str

# 3. Session: everything else is automatic
async with mem.session(user_id="u1", agent="support") as s:
    ctx = await s.context("where is my refund?")   # user facts + past episodes + session
    s.message("user", "where is my refund?")
    lookup_order("42")                              # recorded into this session
    s.record(TicketOpened(ticket="T-9"))
    s.state["ticket"] = "T-9"                       # working memory, persisted
    s.remember("preferred_contact", "email")        # explicit user fact
# on exit: the session's events are consolidated into an episode and user facts
```

`ctx.render()` gives a block of text ready to put in front of an agent's
instructions or prompt.

## With CommonADK

CommonADK normalizes all six of its targets (Google ADK, OpenAI Agents,
Claude Agent SDK, CrewAI, AutoGen, LangGraph) into one event stream, so one
hook gives every target the same memory:

```python
from commonmemo.integrations.commonadk import run_with_memory

async with mem.session(user_id="u1", agent="coordinator") as s:
    trace = await run_with_memory(s, runner, project, "coordinator", prompt)
```

## Development

```bash
pip install -e ".[dev,commonadk]"
pytest
```

## License

Apache-2.0
