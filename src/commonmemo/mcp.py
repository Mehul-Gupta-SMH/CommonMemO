"""An MCP server for CommonMemo memory, over stdio, using only the standard library.

Coding harnesses (Claude Code, Codex, OpenCode) and other MCP clients can't
import Python tools, but they all speak MCP. This server gives them the
memory tools, bound to one session:

    commonmemo mcp --user ada --agent claude-code      # or: python -m commonmemo.mcp

- **Tools** (the pull half): `memory_recall`, and unless `--read-only`,
  `memory_remember`, `memory_note_fact`, `memory_avoid`,
  `memory_save_progress`. They are `agent_tools.py`'s functions, so agent
  writes are recorded with `source="agent"` and agent-written world facts
  wait for approval, exactly as in Python.
- **Instructions** (the push half, where the harness supports it): the
  `initialize` result's `instructions` carry the stable memory (negatives,
  profile facts, world facts, lessons) and how to use the tools.
- **Resource** `memory://context`: the full rendered context for the session.

Configuration comes from flags or the environment (flag wins):

| Flag | Environment | Default |
|---|---|---|
| `--store` | `COMMONMEMO_STORE` | `sqlite:///.commonmemo/memory.db` (per working directory) |
| `--engine` | `COMMONMEMO_ENGINE` | `local` |
| `--project` | `COMMONMEMO_PROJECT` | none |
| `--user` | `COMMONMEMO_USER` | none |
| `--agent` | `COMMONMEMO_AGENT` | `mcp` |
| `--session` | `COMMONMEMO_SESSION` | a new session per server start |
| `--read-only` | `COMMONMEMO_READ_ONLY=1` | off |

Passing the same `--session` from another harness continues that session.
Opening a session that belongs to another user is refused, and the server
exits with an error. When the client closes stdin, the session is closed
and consolidated.

The protocol is newline-delimited JSON-RPC 2.0 over stdin/stdout, using
the `initialize` handshake (protocol versions 2024-11-05 to 2025-11-25).
`server/discover` (the stateless 2026 revision) answers "method not found",
which tells a modern client to fall back to the handshake. Nothing but
protocol messages is ever written to stdout; logs go to stderr.

Capturing events automatically is out of scope: MCP depends on the model
choosing to call a tool. That is the harness hook adapters' job (#5).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import sys
from typing import IO, Any, Callable, Optional

from . import __version__
from .agent_tools import build_agent_tools, run_sync
from .memory import Memory
from .session import Session

log = logging.getLogger("commonmemo.mcp")

PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
"""Handshake protocol revisions this server speaks, oldest to newest."""

CONTEXT_URI = "memory://context"

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602


# -- tool schemas from the Python functions ------------------------------------


def _doc_parts(fn: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    """`(description, {param: description})` from a Google-style docstring."""
    doc = inspect.cleandoc(fn.__doc__ or "")
    head, _, args = doc.partition("\nArgs:\n")
    params: dict[str, str] = {}
    current = None
    for line in args.splitlines():
        m = re.match(r"\s{2,}(\w+):\s*(.*)", line)
        if m:
            current = m.group(1)
            params[current] = m.group(2).strip()
        elif current and line.strip():
            params[current] += " " + line.strip()
    return " ".join(head.split()), params


def tool_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """The MCP tool definition for one of the memory tools."""
    description, param_docs = _doc_parts(fn)
    properties: dict[str, Any] = {}
    required = []
    for name, p in inspect.signature(fn).parameters.items():
        annotation = p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", "str")
        prop: dict[str, Any] = {"type": _JSON_TYPES.get(annotation, "string")}
        if name in param_docs:
            prop["description"] = param_docs[name]
        if p.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = p.default
        properties[name] = prop
    return {
        "name": fn.__name__,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required,
                        "additionalProperties": False},
    }


# -- the server ----------------------------------------------------------------


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class MemoryMCPServer:
    """Answers MCP requests for one memory session. Transport-agnostic:
    `handle(message)` takes one decoded JSON-RPC message and returns the
    response to send, or None for a notification."""

    def __init__(self, memory: Memory, session: Session, *, read_only: bool = False) -> None:
        self.memory = memory
        self.session = session
        self.read_only = read_only
        self.tools = {fn.__name__: fn for fn in build_agent_tools(session, writes=not read_only)}
        self.initialized = False

    # -- protocol --------------------------------------------------------------

    def handle(self, message: Any) -> Optional[dict[str, Any]]:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(message.get("id") if isinstance(message, dict) else None,
                          INVALID_REQUEST, "expected a JSON-RPC 2.0 object")
        method, msg_id = message.get("method"), message.get("id")
        if method is None:
            return None  # a response to something we never send; ignore
        is_notification = "id" not in message
        handler = getattr(self, "_m_" + method.replace("/", "_"), None)
        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
            return None
        if handler is None:
            return _error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}")
        try:
            return {"jsonrpc": "2.0", "id": msg_id, "result": handler(message.get("params") or {})}
        except RpcError as e:
            return _error(msg_id, e.code, str(e))
        except Exception as e:  # noqa: BLE001 -- a bug must not take the server down
            log.exception("commonmemo mcp: %s failed", method)
            return _error(msg_id, -32603, f"internal error: {type(e).__name__}: {e}")

    def _m_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
            "serverInfo": {"name": "commonmemo", "version": __version__},
            "instructions": self.instructions(),
        }

    def _m_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _m_logging_setLevel(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _m_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": [tool_schema(fn) for fn in self.tools.values()]}

    def _m_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name, arguments = params.get("name"), params.get("arguments") or {}
        fn = self.tools.get(name)
        if fn is None:
            raise RpcError(INVALID_PARAMS, f"unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise RpcError(INVALID_PARAMS, "arguments must be an object")
        try:
            inspect.signature(fn).bind(**arguments)
        except TypeError as e:
            return _tool_result(f"Invalid arguments for {name}: {e}", error=True)
        try:
            return _tool_result(str(fn(**arguments)))
        except Exception as e:  # noqa: BLE001 -- tool errors are results the model can read
            return _tool_result(f"{name} failed: {type(e).__name__}: {e}", error=True)

    def _m_resources_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"resources": [{
            "uri": CONTEXT_URI,
            "name": "memory-context",
            "title": "Memory context",
            "description": "Everything memory holds for this session, rendered as reference data.",
            "mimeType": "text/markdown",
        }]}

    def _m_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("uri") != CONTEXT_URI:
            raise RpcError(INVALID_PARAMS, f"unknown resource: {params.get('uri')}")
        text = run_sync(self.session.context("")).render() or "(memory is empty)"
        return {"contents": [{"uri": CONTEXT_URI, "mimeType": "text/markdown", "text": text}]}

    # -- content ---------------------------------------------------------------

    def instructions(self) -> str:
        who = f" for user {self.session.user_id}" if self.session.user_id else ""
        lines = [
            f"CommonMemo memory{who} (session {self.session.session_id}).",
            "Call memory_recall when the request may depend on an earlier conversation that is not in front of you.",
        ]
        if not self.read_only:
            lines.append("Save durable facts the user states with memory_remember, things to stop doing with "
                         "memory_avoid, and progress on long tasks with memory_save_progress.")
        stable = run_sync(self.session.context("")).stable_text()
        return "\n".join(lines) + (f"\n\n{stable}" if stable else "")

    # -- lifecycle -------------------------------------------------------------

    def serve(self, stdin: IO[str], stdout: IO[str]) -> None:
        """Read JSON-RPC messages, one per line, until stdin closes."""
        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                response: Optional[dict[str, Any]] = _error(None, PARSE_ERROR, f"parse error: {e}")
            else:
                response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                stdout.flush()

    def close(self) -> None:
        run_sync(self.session.aclose())
        run_sync(self.memory.aclose())


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_result(text: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


# -- command line --------------------------------------------------------------


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def build_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(prog="commonmemo mcp", description="CommonMemo memory as an MCP server (stdio).")
    env = os.environ.get
    p.add_argument("--store", default=env("COMMONMEMO_STORE", "sqlite:///.commonmemo/memory.db"))
    p.add_argument("--engine", default=env("COMMONMEMO_ENGINE", "local"))
    p.add_argument("--project", default=env("COMMONMEMO_PROJECT"))
    p.add_argument("--user", default=env("COMMONMEMO_USER"))
    p.add_argument("--agent", default=env("COMMONMEMO_AGENT", "mcp"))
    p.add_argument("--session", default=env("COMMONMEMO_SESSION"))
    p.add_argument("--read-only", action="store_true", default=_env_flag("COMMONMEMO_READ_ONLY"))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(name)s: %(message)s")
    memory = Memory(args.store, args.engine, project=args.project)
    try:
        session = memory.session(args.session, user_id=args.user, agent=args.agent)
    except PermissionError as e:
        print(str(e), file=sys.stderr)
        run_sync(memory.aclose())
        return 2
    server = MemoryMCPServer(memory, session, read_only=args.read_only)
    try:
        server.serve(sys.stdin, sys.stdout)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
