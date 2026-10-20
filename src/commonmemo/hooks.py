"""Hook adapters for coding harnesses: automatic capture and push.

The MCP server (`mcp.py`) gives a harness the memory tools, but whether
anything is recorded then depends on the model calling one. Hooks don't:
the harness runs `commonmemo hook` on its own events, with the event's JSON
on stdin, and memory sees every prompt, tool call and reply.

    commonmemo setup claude-code --user ada     # prints the hook config to add
    commonmemo hook --harness claude-code        # what that config runs

Claude Code and Codex send the same payload shape (checked against Claude
Code 2.1.302 and the schemas built into Codex 0.163.0); OpenCode's plugin
(`opencode_plugin.js`, printed by `commonmemo setup opencode`) translates its
events into it. So one handler serves all three:

| `hook_event_name` | Recorded | Pushed back (`additionalContext`) |
|---|---|---|
| `SessionStart` | opens or resumes the session | everything memory holds for the session: stable memory, and where a resumed session stands (its checkpoint and latest events) |
| `UserPromptSubmit` | the prompt, as a user `Message` | memory for this turn: relevant recaps, state |
| `PostToolUse` / `PostToolUseFailure` | a `ToolInvoked` (feeds tool memory) | recaps of earlier sessions the result brings up, if the turn's push missed them (`Session.refresh`, at most `MemoryPolicy.max_refreshes` per turn) |
| `Stop` | the reply, as an assistant `Message` | |
| `PreCompact` | a checkpoint, so the work survives compaction | |
| `SessionEnd` | closes the session and consolidates it | |

Other events are ignored. A hook never blocks or breaks the harness: any
error is written to stderr and the hook exits 0 with no output.

**Pointing at more.** With `--recall-tool` (the harness also runs `commonmemo
mcp`), the push says when it left things out, so the model knows to call
`memory_recall`.

**Which session.** By default each harness session is its own memory
session, `<harness>-<harness session id>`. Set `--session` /
`COMMONMEMO_SESSION` to the same id in two harnesses (and the MCP server) to
continue one session across them; events then carry the agent that recorded
them. Resuming a session that belongs to another user is refused.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from importlib import resources
from typing import IO, Any, Optional

from .agent_tools import run_sync
from .cli import DEFAULT_STORE, env_flag, memory_options
from .events import ToolInvoked
from .memory import Memory

HARNESSES = ("claude-code", "codex", "opencode")

MAX_TEXT = 4000
"""Longest string kept from a tool's input or output; the rest is cut."""

RESUME_EVENTS = 10
"""How many of a session's latest events SessionStart pushes."""

_MEMORY_TOOLS = ("memory_recall", "memory_get_procedure", "memory_remember", "memory_note_fact", "memory_avoid",
                 "memory_save_progress", "memory_save_procedure")

# The events each harness is configured for in `setup`; Codex has no
# PostToolUseFailure (a failed call arrives as a PostToolUse).
_EVENTS = {
    "claude-code": ("SessionStart", "UserPromptSubmit", "PostToolUse", "PostToolUseFailure", "Stop",
                    "PreCompact", "SessionEnd"),
    "codex": ("SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "PreCompact", "SessionEnd"),
}


def _clip(value: Any) -> Any:
    """`value` with long strings cut, so a file's contents don't fill the Log."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + f"... [{len(value) - MAX_TEXT} more chars]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _clip(value if isinstance(value, str) else json.dumps(value, default=str))


def _tool_error(response: Any) -> Optional[str]:
    """The error in a tool response that reports one, else None."""
    if isinstance(response, dict) and (response.get("is_error") or response.get("isError")):
        return _text(response.get("error") or response.get("content") or response)
    return None


def session_id_for(harness: str, payload: dict[str, Any], configured: Optional[str]) -> str:
    """The memory session for a hook payload: the configured one, else one per harness session."""
    return configured or f"{harness}-{payload['session_id']}"


def handle(
    payload: dict[str, Any], memory: Memory, *, harness: str, user: Optional[str] = None,
    agent: Optional[str] = None, session: Optional[str] = None, read_only: bool = False,
    recall_tool: bool = False,
) -> Optional[dict[str, Any]]:
    """Record one hook event; returns the JSON to print, or None.
    `read_only` pushes memory but records nothing."""
    event = payload.get("hook_event_name")
    if event not in _EVENTS["claude-code"]:
        return None  # before opening anything: an unrelated event never reopens a closed session
    s = memory.session(session_id_for(harness, payload, session), user_id=user, agent=agent or harness)
    s.recall_tool = recall_tool  # the harness also runs `commonmemo mcp`, so the push can point at it
    if read_only and event not in ("SessionStart", "UserPromptSubmit"):
        return None

    if event == "SessionStart":
        # A session resumed here may have been started in another harness,
        # whose conversation this one never saw: include its latest events.
        if read_only:
            return _push(event, run_sync(s.context("", recent=RESUME_EVENTS)).render())
        return _push(event, run_sync(s.push("", reason="start", part="all", recent=RESUME_EVENTS)).render())
    if event == "UserPromptSubmit":
        prompt = payload.get("prompt") or ""
        if read_only:
            return _push(event, run_sync(s.context(prompt, recent=0)).turn_text())
        if prompt:
            s.message("user", prompt)
        return _push(event, run_sync(s.push_turn(prompt, recent=0)).turn_text())
    if event in ("PostToolUse", "PostToolUseFailure"):
        name = payload.get("tool_name") or "unknown"
        if not name.endswith(_MEMORY_TOOLS):  # the memory tools' own calls are already in memory
            failed = event == "PostToolUseFailure"
            response = payload.get("tool_response")
            arguments = payload.get("tool_input")
            s.record(ToolInvoked(
                name=name,
                arguments=_clip(arguments) if isinstance(arguments, dict) else {"input": _clip(arguments)},
                result=None if failed else _text(response),
                error=_text(payload.get("error")) if failed else _tool_error(response),
                duration_ms=payload.get("duration_ms"),
            ))
            # The result may bring up an earlier session the turn's push missed.
            seen = " ".join(filter(None, (_text(arguments), _text(response), _text(payload.get("error")))))
            refreshed = run_sync(s.refresh(seen))
            return _push(event, refreshed.turn_text()) if refreshed else None
    elif event == "Stop":
        reply = payload.get("last_assistant_message")
        if reply:
            s.message("assistant", _clip(reply))
    elif event == "PreCompact":
        s.checkpoint("context compacted", {"trigger": payload.get("trigger"), "harness": harness})
    elif event == "SessionEnd":
        run_sync(s.aclose())
    return None


def _push(event: str, text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


# -- command line --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="commonmemo hook", description="Handle one harness hook event (JSON on stdin).")
    p.add_argument("--harness", choices=HARNESSES, required=True)
    _recall_option(p)
    return memory_options(p, agent=None)


def _recall_option(p: argparse.ArgumentParser) -> None:
    p.add_argument("--recall-tool", action="store_true", default=env_flag("COMMONMEMO_RECALL_TOOL"),
                   help="the harness also has the memory MCP server, so the push may say "
                        "\"call memory_recall\" when it left things out")


def run(argv: Optional[list[str]], stdin: IO[str], stdout: IO[str], stderr: IO[str]) -> int:
    args = build_parser().parse_args(argv)
    memory: Optional[Memory] = None
    try:
        payload = json.load(stdin)
        if not isinstance(payload, dict) or "session_id" not in payload:
            raise ValueError("expected a hook payload with a session_id")
        memory = Memory(args.store, args.engine, project=args.project)
        out = handle(payload, memory, harness=args.harness, user=args.user, agent=args.agent,
                     session=args.session, read_only=args.read_only, recall_tool=args.recall_tool)
        if out is not None:
            stdout.write(json.dumps(out) + "\n")
    except Exception as e:  # noqa: BLE001 -- a memory failure must never block the harness
        print(f"commonmemo hook: {type(e).__name__}: {e}", file=stderr)
    finally:
        if memory is not None:
            run_sync(memory.aclose())
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(argv, sys.stdin, sys.stdout, sys.stderr)


# -- setup ---------------------------------------------------------------------


def _command(harness: str, args: argparse.Namespace) -> str:
    parts = ["commonmemo", "hook", "--harness", harness]
    for flag in ("store", "engine", "project", "user", "agent", "session"):
        value = getattr(args, flag)
        if value is not None and value != {"store": DEFAULT_STORE, "engine": "local"}.get(flag):
            parts += [f"--{flag}", value]
    if args.read_only:
        parts.append("--read-only")
    if args.recall_tool:
        parts.append("--recall-tool")
    return " ".join(shlex.quote(p) for p in parts)


def hook_config(harness: str, args: argparse.Namespace) -> dict[str, Any]:
    """The `hooks` settings block for Claude Code (`.claude/settings.json`)
    or Codex (`.codex/hooks.json`): every event runs the same command."""
    command = _command(harness, args)
    return {"hooks": {
        event: [{"hooks": [{"type": "command", "command": command}]}] for event in _EVENTS[harness]
    }}


def opencode_plugin() -> str:
    """The OpenCode plugin (`.opencode/plugin/commonmemo.js`)."""
    return resources.files("commonmemo").joinpath("opencode_plugin.js").read_text(encoding="utf-8")


def setup_main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="commonmemo setup", description="Print the hook config for a harness.")
    p.add_argument("harness", choices=HARNESSES)
    _recall_option(p)
    args = memory_options(p, agent=None).parse_args(argv)
    if args.harness == "opencode":
        sys.stdout.write(opencode_plugin())
        print("// Save as .opencode/plugin/commonmemo.js. It runs `commonmemo hook --harness opencode`;\n"
              "// set COMMONMEMO_USER and the other COMMONMEMO_* variables in OpenCode's environment.",
              file=sys.stderr)
    else:
        where = ".claude/settings.json" if args.harness == "claude-code" else ".codex/hooks.json"
        print(json.dumps(hook_config(args.harness, args), indent=2))
        print(f"# Merge into {where}." + (" Codex asks you to review new hooks at startup: trust them,"
                                          " or they won't run." if args.harness == "codex" else ""),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
