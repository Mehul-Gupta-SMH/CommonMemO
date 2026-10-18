"""The `commonmemo` command. Subcommands:

    commonmemo mcp [options]     # memory as an MCP server over stdio (see mcp.py)
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="commonmemo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mcp", help="serve memory to MCP clients over stdio", add_help=False)
    args, rest = parser.parse_known_args(argv)
    if args.command == "mcp":
        from .mcp import main as mcp_main

        return mcp_main(rest)
    return 2  # pragma: no cover -- argparse rejects unknown commands first


if __name__ == "__main__":
    sys.exit(main())
