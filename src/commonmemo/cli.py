"""The `commonmemo` command. Subcommands:

    commonmemo mcp [options]                  # memory as an MCP server over stdio (mcp.py)
    commonmemo hook --harness NAME [options]  # one harness hook event, JSON on stdin (hooks.py)
    commonmemo setup NAME [options]           # print the hook config for a harness (hooks.py)

They share the memory options below: a flag wins over its `COMMONMEMO_*`
environment variable, which wins over the default.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

DEFAULT_STORE = "sqlite:///.commonmemo/memory.db"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def memory_options(p: argparse.ArgumentParser, *, agent: Optional[str]) -> argparse.ArgumentParser:
    """Add the options that pick the memory, user, agent and session."""
    env = os.environ.get
    p.add_argument("--store", default=env("COMMONMEMO_STORE", DEFAULT_STORE))
    p.add_argument("--engine", default=env("COMMONMEMO_ENGINE", "local"))
    p.add_argument("--project", default=env("COMMONMEMO_PROJECT"))
    p.add_argument("--user", default=env("COMMONMEMO_USER"))
    p.add_argument("--agent", default=env("COMMONMEMO_AGENT", agent))
    p.add_argument("--session", default=env("COMMONMEMO_SESSION"))
    p.add_argument("--read-only", action="store_true", default=_env_flag("COMMONMEMO_READ_ONLY"))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="commonmemo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mcp", help="serve memory to MCP clients over stdio", add_help=False)
    sub.add_parser("hook", help="handle one harness hook event (JSON on stdin)", add_help=False)
    sub.add_parser("setup", help="print the hook config for claude-code, codex or opencode", add_help=False)
    args, rest = parser.parse_known_args(argv)
    if args.command == "mcp":
        from .mcp import main as mcp_main

        return mcp_main(rest)
    from . import hooks

    return hooks.main(rest) if args.command == "hook" else hooks.setup_main(rest)


if __name__ == "__main__":
    sys.exit(main())
