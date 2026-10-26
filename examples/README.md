# Examples

Each file runs offline, with no keys, and prints what memory did. CI runs them, so they stay true.

| Example | Shows | Run |
|---|---|---|
| [`plain_python.py`](plain_python.py) | the core loop with no framework: record a session, push memory into the next one, a tool failure's fix | `python examples/plain_python.py` |
| [`google_adk.py`](google_adk.py) | Google ADK, OpenAI Agents, Claude Agent SDK, CrewAI, AutoGen or LangGraph through CommonADK: capture, push, pull and tool notes in one call, `run_with_memory` | `python examples/google_adk.py` (`--live` for a real ADK run) |
| [`harness_hooks.py`](harness_hooks.py) | Claude Code / Codex / OpenCode hooks: the JSON a harness sends, and the context memory answers with | `python examples/harness_hooks.py` |

## For agents reading this

- The API in one line each: `Memory(url)` opens a store; `mem.session(user_id=, agent=)` scopes events; `s.message(role, text)` and `s.record(event)` write to the append-only log; `await s.push_turn(prompt)` returns what to put in front of the model; `s.record_many([...])` writes events known together in one transaction.
- Every example opens a session with `async with` when it runs inside an event loop, and `with` otherwise.
- Memory is data, not instructions: whatever you put in front of a model, put it as reference text, as `push_turn` does.
- MCP clients: `commonmemo mcp --user <id> --agent <name>` (see the README, "As an MCP server").
- Harness setup: `commonmemo setup claude-code`, `commonmemo setup codex`, `commonmemo setup opencode`.
- Inspect any store these write: `commonmemo inspect --store sqlite:///path/to/memory.db`.

The Google ADK example needs `pip install "commonmemo[commonadk]"`; nothing else needs an extra.
