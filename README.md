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

Beside the tiers, CommonMemo keeps **structured kinds** of memory itself:
**tool memory** (which tools fail, how, and what fixed it — never the
users' data), **world facts** (with expiry), **feedback** (ratings and
corrections), **negatives** ("never do X", and tombstones for "forget
this"), and **resume points** (checkpoints and replayable steps, so a
stopped process continues where it was).

Agents use memory in **hybrid** mode: a bounded, stable part is pushed into
their instructions, a per-turn part with the request, tool notes into the
tools' own descriptions — and agents pull the rest, and write, through
memory tools (`memory_recall`, `memory_remember`, ...).

CommonMemo **wraps** memory engines rather than building one: extraction,
retrieval and their optimizations belong to engines that specialize in them
(Graphiti and Mem0). The built-in `local` engine is offline and
deterministic, so everything works out of the box and in tests.

**Status:** v0.0.1 -- the design and the skeleton. See
[`docs/design.md`](docs/design.md) for the design and the roadmap.

## Install

```bash
pip install commonmemo                 # core: stdlib only, local engine
pip install "commonmemo[commonadk]"    # + CommonADK integration
pip install "commonmemo[graphiti]"     # + Graphiti engine (temporal knowledge graph)
pip install "commonmemo[mem0]"         # + Mem0 engine (LLM-extracted user memories)
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

`ctx.stable_text()` belongs in an agent's instructions and `ctx.turn_text()`
with the turn's request (`ctx.render()` gives both). `mem.agent_tools(s)`
returns the memory tools for an agent.

## Resume a stopped process

```python
@mem.step                                  # finished calls replay on a rerun
def fetch(order_id: str) -> dict: ...

@mem.step(side_effects=True)               # never silently run twice
def refund(order_id: str) -> str: ...

with mem.session("job-42") as s:           # same id in a new process = resume
    order = fetch("A1")
    s.checkpoint("fetched", {"order": "A1"})
    refund("A1")
```

Rerunning the same code after a crash skips what finished and continues at
the first step that did not; a side-effecting step interrupted mid-way
raises `StepInDoubt` instead of repeating itself.

## With Graphiti

```python
from graphiti_core import Graphiti
from commonmemo.engines.graphiti import GraphitiEngine

mem = Memory("sqlite:///mem.db", engine=GraphitiEngine(graphiti=Graphiti(uri, user, password)))
# or Memory("sqlite:///mem.db", engine="graphiti") with NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD set
```

Each closed session becomes one Graphiti episode in the user's graph
partition; retrieval ranks past sessions by the graph facts that match, and
extracted facts join the user's explicit ones.

## With Mem0

```python
from commonmemo.engines.mem0 import Mem0Engine

mem = Memory("sqlite:///mem.db", engine=Mem0Engine(config={...}))  # Mem0's own config dict
# or Memory("sqlite:///mem.db", engine="mem0") for Mem0's defaults (OpenAI)
```

Each closed session is handed to Mem0, which extracts and updates the
user's memories; they join the user's explicit facts, and retrieval ranks
past sessions by the memories that match.

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
pip install -e ".[dev,commonadk,graphiti,mem0]"
pytest
```

## License

Apache-2.0
