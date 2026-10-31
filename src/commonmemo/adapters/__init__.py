"""Adapters: CommonMemo behind each agent SDK's own memory or session interface (#6).

Each module implements one SDK's interface over a `Memory`, so an application
using that SDK directly -- without CommonADK -- gets the same memory, and
a session started under one SDK can be continued under another. They are
imported by name and need their SDK installed (one extra each):

- `commonmemo.adapters.adk`: Google ADK `BaseMemoryService` (`commonmemo[adk]`)
- `commonmemo.adapters.openai_agents`: OpenAI Agents `Session` (`commonmemo[openai-agents]`)
- `commonmemo.adapters.langgraph`: LangGraph `BaseStore` (`commonmemo[langgraph]`)
- `commonmemo.adapters.autogen`: AutoGen `Memory` (`commonmemo[autogen]`)
- `commonmemo.adapters.claude_agent_sdk`: Claude Agent SDK `SessionStore` (`commonmemo[claude-agent-sdk]`)
- `commonmemo.adapters.auto`: `commonmemo.instrument(memory)` attaches the above to the SDKs' defaults

The adapters store and search; they do not push context into a prompt (that is
the hooks' and the MCP server's job), except AutoGen's, whose `Memory` interface is that hook. A record the SDK's interface cannot
hold is not lost: it is kept in the Log as an `ExternalEvent`, and each
adapter's docstring says what its interface cannot express.
"""
