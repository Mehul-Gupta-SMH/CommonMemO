"""Adapters: CommonMemo behind each agent SDK's own memory or session interface (#6).

Each module implements one SDK's interface over a `Memory`, so an application
using that SDK directly -- without CommonADK -- gets the same memory, and
a session started under one SDK can be continued under another. They are
imported by name and need their SDK installed (one extra each):

- `commonmemo.adapters.adk`: Google ADK `BaseMemoryService` (`commonmemo[adk]`)

The adapters store and search; they do not push context into a prompt (that is
the hooks' and the MCP server's job). A record the SDK's interface cannot
hold is not lost: it is kept in the Log as an `ExternalEvent`, and each
adapter's docstring says what its interface cannot express.
"""
