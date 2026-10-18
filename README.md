# CommonMemo

Framework-neutral memory for agents and workflows. Four tiers behind one
`Memory` handle, one install, and a decorator or a subclass to get events in
-- everything else (working state, recaps of past sessions, the user profile,
retrieval, context for the next turn) is built in.

| Tier | Holds | Owned by |
|---|---|---|
| **Log** | everything that happened, in order, never edited: messages, tool calls, your domain events | CommonMemo (SQLite) |
| **Working** | what the current conversation or workflow run has in hand: live state and resume points, resumable by id | CommonMemo (SQLite) |
| **Recaps** | short summaries of finished sessions, retrieved by relevance | a wrapped engine |
| **UserProfile** | durable facts about one user; a newer value supersedes, history is kept | a wrapped engine |

These names are new (before: event, session, episodic, user). In code, a
recap is a `Recap` (`ctx.recaps`, `engine.search_recaps`) and the profile is
`engine.user_profile(user_id)`, a list of `Fact`s ([#19](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/19)).

Beside the tiers, CommonMemo keeps **structured kinds** of memory itself:
**tool memory** (which tools fail, how, and what fixed it — never the
users' data), **world facts** (with expiry; an agent's are proposals until
an operator approves them), **feedback** (ratings and
corrections), **negatives** ("never do X", and tombstones for "forget
this"), and **resume points** (checkpoints and replayable steps, so a
stopped process continues where it was).

Agents use memory in **hybrid** mode: a bounded part is *pushed* each turn —
a stable part (meant for the agent's instructions), a per-turn part with the
request, and tool notes in the tools' own descriptions — and agents *pull*
the rest, and write, through memory tools (`memory_recall`,
`memory_remember`, ...). All pushed memory is framed as reference data, not
instructions.

CommonMemo **wraps** memory engines rather than building one: extraction,
retrieval and their optimizations belong to engines that specialize in them
(Graphiti and Mem0). The built-in `local` engine is offline and
deterministic, so everything works out of the box and in tests.

**Status:** v0.0.1, not yet on PyPI ([#12](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/12)). Four tiers, three engines
(local, Graphiti, Mem0), the structured kinds, hybrid agent memory and the
CommonADK integration are in. The Graphiti and Mem0 engines are tested
against their real libraries' types but have not had a live run against a
real backend and LLM yet ([#3](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/3)). See [`docs/design.md`](docs/design.md) for
the design and the [roadmap issue (#14)](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/14) for what's next.

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

mem = Memory("sqlite:///mem.db", project="shop")   # one path is the whole setup

# 1. Decorator: every call becomes an event (arguments, result/error, duration)
@mem.track
def lookup_order(order_id: str) -> str: ...

# 2. Subclass: your own domain events -- no @dataclass needed
class TicketOpened(MemoryEvent):
    ticket: str

# 3. Session: everything else is automatic
async with mem.session(user_id="u1", agent="support") as s:
    ctx = await s.context("where is my refund?")   # what memory pushes this turn
    s.message("user", "where is my refund?")
    lookup_order("42")                              # recorded into this session; feeds tool memory
    s.record(TicketOpened(ticket="T-9"))
    s.state["ticket"] = "T-9"                       # working memory, persisted
    s.remember("preferred_contact", "email")        # explicit user fact
    s.learn("refund_window", "30 days")             # world fact (ttl= for ones that go stale)
    s.feedback(-1, "too long")                      # rating / comment
    s.correct("Pune", "Bangalore")                  # correction
    s.never("suggest phone calls")                  # standing constraint
    s.forget("old_address")                         # tombstone: stays forgotten
    tools = mem.agent_tools(s)                      # memory_recall, memory_remember, ... for an agent
# on exit: the session's events are consolidated into a recap and UserProfile facts
```

`ctx.stable_text()` is meant for an agent's instructions and
`ctx.turn_text()` for the turn's request (`ctx.render()` gives both).
`mem.tool_notes(agent)` gives tool memory's advice per tool, and
`mem.tool_profile(tool)` the numbers behind it.

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
raises `StepInDoubt` instead of repeating itself. Side-effecting steps are
claimed before they run, so even several workers resuming one session at
once run each of them at most once; the others get `StepInProgress` while
the winner's lease is live.

## See how memory changes

```python
with mem.run("deploy-42"):                  # tags everything written inside
    mem.learn("refund_window", "30 days")

mem.snapshot("before v3")
h = mem.history()
h.velocity(bucket=3600)                     # writes per hour, by kind/source/scope
h.churn(); h.review_flow(); h.by_run("deploy-42")
mem.as_of("before v3").world_facts()        # memory as it was then
mem.diff("before v3")                       # what changed since
```

And the reins, for when memory goes wrong:

```python
mem.revert(to="before v3")                 # world facts and negatives back to a snapshot
mem.revert(run_id="commonadk:r-8812")      # undo one run; later edits by others survive
mem.freeze(source="agent")                 # agent world-fact writes become proposals
mem.alerts()                               # velocity spikes, churn, review backlog
```

Reverts append compensating events and never delete, so they can be undone
too. Alerts are on by default and logged to `commonmemo.alerts` ([#27](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/27)).
A mockup of the inspector these feed ([#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25)), with sample data, is at
<https://mehul-gupta-smh.github.io/CommonMemO/> (source: `pages/index.html`).

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

Each turn, on a copy of the project (CommonADK itself is never changed):

- **capture:** every CommonADK event is recorded, and tool calls feed tool memory;
- **pull:** the memory tools are added to the agent, and tool memory's notes are
  written into the agent's own tools' descriptions;
- **push:** memory goes in front of the prompt, the only place released
  CommonADK can take it. Putting stable memory in the agent's instructions
  instead is requested in [CommonADK #30](https://github.com/Mehul-Gupta-SMH/CommonADK/issues/30);
  `run_with_memory` uses it automatically if a runner supports it ([#7](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/7)).

## Live runs

`scripts/live_engines.py` runs a short scenario on the Graphiti and Mem0
engines against real backends and an LLM. It is not part of the test
suite because it spends model tokens:

```bash
pip install -e ".[graphiti,mem0]" kuzu
OPENAI_API_KEY=... python scripts/live_engines.py      # Neo4j if NEO4J_* is set, else embedded Kuzu
```

## Does it help? The evaluation suite

`evals/` runs scenarios with memory and without it, against a scripted agent
that decides only from the text it is shown: the prompt and the tools'
descriptions. Among the scenarios:
- a tool failure with a known fix is not repeated;
- a user's "never" is respected;
- the latest preference wins;
- a crashed pipeline resumes without charging twice;
- a forgotten fact never returns;
- stored text never becomes an instruction;
- one user can't change what another is told;
- a burst of bad writes is detected, frozen and reverted.

```
python -m evals                   # prints the report; see docs/evals.md for the baseline
```

It runs in CI on every PR ([#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)). A live mode against a real model comes
after the engines' live run (#3).

## Development

```bash
pip install -e ".[dev,commonadk,graphiti,mem0]"
pytest
```

Planned work is tracked in issues. The [roadmap (#14)](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/14)
lists them; nothing gets coded without an issue.

## License

Apache-2.0
