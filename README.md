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
(local, Graphiti, Mem0), the structured kinds, hybrid agent memory, the
CommonADK integration, an MCP server and hooks for Claude Code, Codex and
OpenCode are in. The Graphiti and Mem0 engines are tested
against their real libraries' types but have not had a live run against a
real backend and LLM yet ([#3](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/3)). See [`docs/design.md`](docs/design.md) for
the design and the [roadmap issue (#14)](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/14) for what's next.

## Install

```bash
pip install commonmemo                 # core: stdlib only, local engine
pip install "commonmemo[commonadk]"    # + CommonADK integration
pip install "commonmemo[graphiti]"     # + Graphiti engine (temporal knowledge graph)
pip install "commonmemo[mem0]"         # + Mem0 engine (LLM-extracted user memories)
pip install "commonmemo[otel]"         # + turn traces as OpenTelemetry spans
```

The core also gives you the `commonmemo` command (`commonmemo mcp`, `commonmemo hook`, below).

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

## Ranking by meaning

Which of a user's 50 facts go in the push, which world facts a request is
about, what `memory_recall` returns: one ranker decides ([#21](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/21)). By default it
matches words (stemmed, without stop words). Give it an embedder and it
matches meaning, so "How do I reach them?" finds `preferred_contact: email`:

```python
from commonmemo import EmbeddingRanker

mem = Memory(ranker=EmbeddingRanker(my_embed))   # any texts -> vectors function, sync or async
```

The Graphiti and Mem0 engines bring their own: their embedder ranks facts
too, with no setup. Embeddings are cached, and a profile that fits the push
is never ranked.

## World facts as configuration

```python
from commonmemo import FactKey

mem = Memory(project="shop", fact_keys=[
    FactKey("refund_window", "How long customers have to return an order", lifetime="90d", pinned=True),
    FactKey("plan_tier", "The customer's plan", allowed=("free", "pro")),
])
mem.load_facts("facts.yaml")     # keys and values, versioned with the app; loading again writes nothing new
```

The registry is optional ([#24](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/24)). With one, agents are shown the keys to reuse,
and their writes to an unregistered or near-duplicate key (`refund_period`)
stay proposals for review. Registered lifetimes make facts go stale by
default. Pinned facts are always pushed; the rest only when the request is
about them, so hundreds of facts don't crowd the prompt.

## How a task is done: procedural memory

Templates, rules, step-by-step procedures and accepted examples, per task
type ([#26](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/26), [ADR 0003](docs/adr/0003-procedural-memory.md)). Write them as files next to your code:

```markdown
<!-- procedures/weekly_report/default.md -->
---
kind: template
use_when: Writing the weekly status report for the team
---
## Highlights
## Risks
## Next week
```

```python
mem.load_procedures("procedures/")        # current at once; a changed file is a new version
with mem.session(user_id="ada", task="weekly_report") as s:   # or leave task out: matched by use_when
    ctx = await s.push_turn(prompt)       # rules, templates and step names, in the turn half
mem.procedure_outcomes("weekly_report")   # runs, feedback and skipped steps, per version
```

Without a named task, the request is matched against the `use_when` lines
with the same ranker as facts. Agents get `memory_get_procedure` (full steps
and examples) and `memory_save_procedure`: an agent's entry is its user's own
way at once, or, for everyone, a proposal an operator approves. Accepted
examples (`s.save_exemplar(...)`) are the user's data: no one else sees one
until an operator shares it with `confirm_procedure(id, scrubbed=True)`, and
`forget_user` removes them.

### Learning from runs

Off by default. Give memory any model, as a function from a prompt to text,
and each finished session is distilled into decision records, lessons
(merged when they repeat, pushed to the same user) and proposed
procedures, rules and exemplar candidates that wait for review:

```python
from commonmemo import ModelLearner

mem = Memory(learner=ModelLearner(lambda prompt: my_model(prompt)))   # sync or async
await mem.distill(session_id)        # or run it later, in a batch
mem.revert(run_id=f"learn:{session_id}")   # undo what one session taught
```

The learner sees messages, tool calls, feedback and step results, with
forgotten text redacted; never raw reasoning.

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
A mockup of where the inspector is heading ([#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25)), with sample data, is at
<https://mehul-gupta-smh.github.io/CommonMemO/> (source: `pages/index.html`).

## See what memory did

Every push into an agent's context is recorded as a **turn trace**: what was
pushed, by reference, where each item came from, and what was left out
(pushed text only with `MemoryPolicy(trace_text=True)`; `forget_user`
removes a user's traces).

```bash
commonmemo trace s2 --why refund_window   # why the agent got it: the events it came from
commonmemo show user ada                  # profile, constraints, what was forgotten, past sessions
commonmemo inspect --open                 # the same, in a local read-only web page
```

```python
mem.traces(session_id="s2"); mem.why("refund_window", session_id="s2")
from commonmemo.otel import instrument; instrument(mem)   # pushes as OpenTelemetry spans
```

More in [docs/seeing-what-memory-did.md](docs/seeing-what-memory-did.md) ([#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25)).

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

## As an MCP server (Claude Code, Codex, any MCP client)

`commonmemo mcp` serves the memory tools to MCP clients over stdio, bound to
one session. It needs nothing beyond the core. For Claude Code, add it to
the project's `.mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "commonmemo",
      "args": ["mcp", "--user", "ada", "--agent", "claude-code"],
      "env": { "COMMONMEMO_STORE": "sqlite:///.commonmemo/memory.db" }
    }
  }
}
```

- **Tools:** `memory_recall`, `memory_remember`, `memory_note_fact`,
  `memory_avoid`, `memory_save_progress`; `--read-only` offers only
  `memory_recall`. Writes are recorded with `source="agent"`, and an agent's
  world facts are proposals until approved, as in Python.
- **Push:** the server's instructions carry the stable memory (negatives,
  profile, world facts, lessons); the resource `memory://context` has all of it.
- **Config:** `--store`, `--engine`, `--project`, `--user`, `--agent`,
  `--session`, `--read-only`, or the same as `COMMONMEMO_*` environment
  variables. The same `--session` from another harness continues the
  session; another user's session is refused. Closing the client closes and
  consolidates the session.

The model decides when to call a tool, so MCP alone doesn't capture every
event; the hooks below do. Tested over real stdio, and against the official
MCP client (`pip install "commonmemo[mcp]"`) ([#4](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/4)).

## Hooks: automatic capture and push in coding harnesses

With hooks, the harness itself hands memory every prompt, tool call and
reply, and puts memory in front of the session and of each prompt -- nothing
depends on the model calling a tool. `commonmemo setup <harness>` prints
the config; every event runs `commonmemo hook`, which reads the event's JSON
on stdin.

**Claude Code:** merge the output into `.claude/settings.json`:

```bash
commonmemo setup claude-code --user ada
```

```jsonc
{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "commonmemo hook --harness claude-code --user ada" }] }],
    // the same entry for UserPromptSubmit, PostToolUse, PostToolUseFailure, Stop, PreCompact and SessionEnd
  }
}
```

**Codex:** merge `commonmemo setup codex --user ada` into `.codex/hooks.json`
(or `~/.codex/hooks.json`). Codex asks you to review new hooks at startup;
trust them, or they won't run.

**OpenCode:** save the plugin and give it the config through the
environment:

```bash
mkdir -p .opencode/plugin && commonmemo setup opencode > .opencode/plugin/commonmemo.js
export COMMONMEMO_USER=ada
```

| Event | Recorded | Pushed |
|---|---|---|
| session start | opens or resumes the session | stable memory, and where a resumed session stands |
| prompt | the prompt | memory for this turn (relevant recaps, state) |
| tool call | the call, its result or error (feeds tool memory) | recaps the result brings up that the turn missed |
| reply | the reply (Claude Code, Codex) | |
| compaction | a checkpoint, so work survives it | |
| session end | closes and consolidates the session | |

Memory is also **refreshed mid-run**: when a tool result brings up an
earlier session that the prompt's push missed, its recap is pushed before
the next model call (at most 3 times per turn; `MemoryPolicy(refresh=False)`
turns it off; [#22](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/22)). A hook never blocks the harness: on any error it writes to
stderr and exits 0.
Each harness session is its own memory session by default. To continue
**one session across harnesses**, give them the same `--session` (or
`COMMONMEMO_SESSION`): start in Claude Code, resume in Codex, and Codex is
told where the work stood; each event records which harness wrote it.
Recaps of a user's past sessions are shared across harnesses either way, so
Claude Code also sees what the user did in Codex yesterday, labelled with
where it happened (`MemoryPolicy(recaps_across_agents=False)` keeps them per
agent; [#34](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/34)). The
hook payloads are tested against recorded payloads from Claude Code
2.1.302, and against the schemas built into Codex 0.163.0 ([#5](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/5)).
Use hooks and the MCP server together: hooks capture and push, the tools let
the agent recall and save on purpose. Add `--recall-tool` to the hook command
(or set `COMMONMEMO_RECALL_TOOL=1`) when the MCP server is configured too:
the push then says when it left things out ("More in memory than shown here:
5 facts about the user. Call memory_recall with what you need."), so the
model knows to look ([#23](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/23)).

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
- a burst of bad writes is detected, frozen and reverted;
- a tool result that brings up an older session gets its recap mid-run;
- when the push leaves things out, the agent is told and recalls them;
- a fact past its lifetime isn't quoted, and an agent's write doesn't override an operator's;
- a fact asked about in other words is recalled (by meaning, with an embedder).

```
python -m evals                   # prints the report; see docs/evals.md for the baseline
```

It runs in CI on every PR ([#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)). A live mode against a real model comes
after the engines' live run (#3).

## Development

```bash
pip install -e ".[dev,commonadk,graphiti,mem0,mcp,otel]"
pytest
```

Planned work is tracked in issues. The [roadmap (#14)](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/14)
lists them; nothing gets coded without an issue.

## License

Apache-2.0
