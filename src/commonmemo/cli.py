"""The `commonmemo` command. Subcommands:

    commonmemo mcp [options]                  # memory as an MCP server over stdio (mcp.py)
    commonmemo hook --harness NAME [options]  # one harness hook event, JSON on stdin (hooks.py)
    commonmemo setup NAME [options]           # print the hook config for a harness (hooks.py)
    commonmemo show user ID | session ID      # what memory holds, read-only (views.py)
    commonmemo trace SESSION [--why KEY]      # what each push gave the agent, and why (traces.py)
    commonmemo inspect [--port N]             # the same, in a local read-only web page (inspector.py)

They share the memory options below: a flag wins over its `COMMONMEMO_*`
environment variable, which wins over the default.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import argparse

    from .context import MemoryPolicy
    from .memory import Memory

DEFAULT_STORE = "sqlite:///.commonmemo/memory.db"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


WINDOWS = "8"
"""`MemoryPolicy.messages` and `recall_messages`, as the option default (a test keeps them equal)."""


def memory_options(p: argparse.ArgumentParser, *, agent: Optional[str]) -> argparse.ArgumentParser:
    """Add the options that pick the memory, user, agent and session."""
    env = os.environ.get
    p.add_argument("--store", default=env("COMMONMEMO_STORE", DEFAULT_STORE))
    p.add_argument("--engine", default=env("COMMONMEMO_ENGINE", "local"))
    p.add_argument("--project", default=env("COMMONMEMO_PROJECT"))
    p.add_argument("--user", default=env("COMMONMEMO_USER"))
    p.add_argument("--agent", default=env("COMMONMEMO_AGENT", agent))
    p.add_argument("--session", default=env("COMMONMEMO_SESSION"))
    p.add_argument("--read-only", action="store_true", default=env_flag("COMMONMEMO_READ_ONLY"))
    p.add_argument("--messages", type=int, default=int(env("COMMONMEMO_MESSAGES", WINDOWS)),
                   help="windows of the user's past messages to push with each turn (default 8; 0: none)")
    p.add_argument("--recall-messages", type=int,
                   default=int(env("COMMONMEMO_RECALL_MESSAGES", WINDOWS)),
                   help="windows of past messages memory_recall returns (default 8; 0: none)")
    p.add_argument("--message-dates", choices=("off", "stamp", "resolve"), default=env("COMMONMEMO_MESSAGE_DATES", "off"),
                   help="what windows say about when: off, stamp (dated messages), resolve (also relative dates)")
    return p


def policy_from(args: argparse.Namespace) -> "MemoryPolicy":
    """The memory policy the command line asks for."""
    from .context import MemoryPolicy

    return MemoryPolicy(messages=args.messages, recall_messages=args.recall_messages, message_dates=args.message_dates)


def main(argv: Optional[list[str]] = None) -> int:
    args_ = sys.argv[1:] if argv is None else argv
    if args_[:1] == ["hook"] and ("--resident" in args_ or env_flag("COMMONMEMO_RESIDENT")):
        from .hookclient import main as client_main  # imports almost nothing: that is the point (#92)

        return client_main(args_[1:])
    import argparse

    parser = argparse.ArgumentParser(prog="commonmemo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mcp", help="serve memory to MCP clients over stdio", add_help=False)
    sub.add_parser("hook", help="handle one harness hook event (JSON on stdin)", add_help=False)
    sub.add_parser("setup", help="print the hook config for claude-code, codex or opencode", add_help=False)
    sub.add_parser("show", help="show a user or a session, read-only", add_help=False)
    sub.add_parser("trace", help="show a session's turn traces", add_help=False)
    sub.add_parser("inspect", help="open a local read-only web inspector", add_help=False)
    args, rest = parser.parse_known_args(argv)
    if args.command == "mcp":
        from .mcp import main as mcp_main

        return mcp_main(rest)
    if args.command in ("hook", "setup"):
        from . import hooks

        return hooks.main(rest) if args.command == "hook" else hooks.setup_main(rest)
    if args.command == "inspect":
        from .inspector import main as inspect_main

        return inspect_main(rest)
    return show_main(rest) if args.command == "show" else trace_main(rest)


def _open(args: argparse.Namespace) -> "Memory":
    from .memory import Memory

    return Memory(args.store, args.engine, project=args.project, policy=policy_from(args))


def show_main(argv: Optional[list[str]] = None) -> int:
    from .agent_tools import run_sync
    from .views import session_lines, session_view, user_lines, user_view

    import argparse

    p = argparse.ArgumentParser(prog="commonmemo show", description="Show what memory holds, read-only.")
    p.add_argument("what", choices=("user", "session"))
    p.add_argument("id")
    args = memory_options(p, agent=None).parse_args(argv)
    mem = _open(args)
    try:
        if args.what == "user":
            lines = user_lines(user_view(mem, args.id))
        else:
            view = session_view(mem, args.id)
            if view is None:
                print(f"no session {args.id!r}", file=sys.stderr)
                return 1
            lines = session_lines(view)
        print("\n".join(lines))
        return 0
    finally:
        run_sync(mem.aclose())


def trace_main(argv: Optional[list[str]] = None) -> int:
    from .agent_tools import run_sync
    from .views import session_lines, session_view

    import argparse

    p = argparse.ArgumentParser(prog="commonmemo trace", description="What each push gave the agent, and why.")
    p.add_argument("session")
    p.add_argument("--why", metavar="KEY", help="trace one pushed item (a fact key, world fact key or id) to its sources")
    args = memory_options(p, agent=None).parse_args(argv)
    mem = _open(args)
    try:
        if args.why:
            found = mem.why(args.why, session_id=args.session)
            if found is None:
                print(f"{args.why!r} was never pushed in session {args.session!r}", file=sys.stderr)
                return 1
            print("\n".join(found.lines()))
            return 0
        view = session_view(mem, args.session)
        if view is None:
            print(f"no session {args.session!r}", file=sys.stderr)
            return 1
        print("\n".join(session_lines(view, traces_only=True)))
        return 0
    finally:
        run_sync(mem.aclose())


if __name__ == "__main__":
    sys.exit(main())
