# Seeing what memory did

Three questions come up the moment an agent with memory says something
surprising:

- *Why did the agent say the refund window is 90 days?*
- *What do we know about this user, and where did it come from?*
- *What did the agent get this turn, and what was left out?*

CommonMemo answers them from **turn traces** ([#25](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/25)): every time memory is
pushed into an agent's context, the Log records what went in.

## Turn traces

Each push appends a `ContextPushed` event:

| Field | What it holds |
|---|---|
| `reason` | `start` (a session began), `turn` (a request) or `refresh` (mid-run, after a tool result) |
| `items` | one reference per pushed item: its kind (`fact`, `world`, `negative`, `lesson`, `recap`), key or id, source, score, and the events it came from |
| `more` | what the push left out (facts past the limit, near-miss recaps, entries cut for length) |
| `chars` | how much was pushed (~4 characters per token) |
| `text` | the pushed text itself, **only** with `MemoryPolicy(trace_text=True)` |

Traces hold references, not text, by default: the text repeats user data,
and references are enough to answer "why". Traces carry the user's id, so
`forget_user` removes them, and `prune` ages them out with the rest of the
Log.

Pushes are recorded by `Session.push` / `push_turn` / `refresh`. The harness
hooks, the CommonADK integration and the evaluation suite use them; your own
code can too:

```python
ctx = await session.push_turn(request)        # what context() returns, plus a trace
mem.traces(session_id="s2")                   # every push in the session, oldest first
mem.why("refund_window", session_id="s2")     # the latest push of it, and the events it came from
```

`why` answers with the trace item and the source events: for a profile fact,
the `UserFact` (which session, which source: the app, the agent, an
extraction); for a world fact, the write and who made it.

## From a terminal

```bash
commonmemo show user ada               # profile with dates, constraints, what was forgotten, past sessions, feedback
commonmemo show session s2             # timeline, where a resume would start, turn traces
commonmemo trace s2                    # just the turn traces
commonmemo trace s2 --why refund_window
```

```
fact preferred_contact: pushed in a turn push to session s2
  from user_fact 93040ff3… in session s1, source explicit
```

All take `--store` (or `COMMONMEMO_STORE`), like the other commands.

## The inspector

```bash
commonmemo inspect --store sqlite:///.commonmemo/memory.db --open
```

A local web page at `http://127.0.0.1:8765/`, from the standard library: no
server to deploy, no account. It shows the overview (recent sessions, users,
world facts and proposals waiting for review), each user's page, and each
session's timeline with its turn traces; every pushed item has a **why** link.
The **Review** page lists what waits for review: proposed and disputed
world facts, and procedure entries (agent proposals, learned ones,
exemplar candidates), each with what it would replace. The same views are
JSON under `/api/` (`/api/overview`, `/api/user/<id>`, `/api/session/<id>`,
`/api/why?session=&key=`, `/api/review`).

It is **read-only by default**: it answers GET only. It binds to 127.0.0.1
only. Stored memory is untrusted text, so every value is escaped and the
pages run no scripts (a strict Content-Security-Policy tells the browser so).

### Reviewing

```bash
commonmemo inspect --review --reviewer ada
```

adds Approve (a world fact optionally corrected), Reject (with a reason)
and, for an exemplar candidate, Share with everyone, which needs "personal
data checked" and redacts the user's known profile values
([#45](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/45)). The actions are plain form posts. Each must carry a token the
server made at start, from a page with the inspector's own Host and Origin,
so another site open in the browser can't post an approval (CSRF) or read
pages through a rebound DNS name. Every action runs under
`run_id="review:<reviewer>"`: it is in the Log, and
`mem.revert(run_id="review:ada")` undoes a review session.

## Next to your LLM calls: OpenTelemetry

```bash
pip install "commonmemo[otel]"
```

```python
from commonmemo.otel import instrument

instrument(mem)        # every push becomes a `commonmemo.push` span from now on
```

The span is a child of whatever span is current, typically the LLM call, so
tracing tools (Jaeger, Honeycomb, Grafana, and Langfuse or LangSmith through
their OTLP endpoints) show memory next to the call it fed. Attributes include
the reason, characters pushed, item counts per kind, the fact and world-fact
keys, recap ids, and what was left out; each item is also a span event. The
user's id is added only with `instrument(mem, include_user=True)`, and pushed
text only with `include_text=True` on traces that kept it.
`commonmemo.otel.export(mem, session_id=...)` sends stored traces after the
fact. An exporter that fails is logged and never breaks a write.

## Engine UIs

For graph-level detail, use the engines' own tools: Neo4j Browser for
Graphiti, Mem0's dashboard. The inspector shows what CommonMemo holds and
pushed; it doesn't rebuild those.
