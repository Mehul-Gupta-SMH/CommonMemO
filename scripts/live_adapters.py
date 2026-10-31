"""Live run: the framework adapters inside a real agent loop (#122).

Not part of the test suite -- it spends model tokens through the `claude` CLI.
The adapters are tested against each SDK's own interfaces; this runs a real
agent through them. Claude Code is the model (Haiku by default), through the
Claude Agent SDK.

Scenarios (each prints its check and the result):

1. store     -- `query()` with `session_store=CommonMemoSessionStore(...)`: the
                transcript is mirrored into the Log, and the session lists.
2. resume    -- the local transcript file is deleted, a new Memory on the same
                database and a new store resume the session by id: the model
                answers from the transcript the Log kept.
3. recall    -- `memory_recall`'s search (`recall_entries`) finds what was said.
4. instrument -- `commonmemo.instrument(memory)`, then `ClaudeAgentOptions()` with
                no store: the SDK's default is ours, and a run lands in the Log.

    python scripts/live_adapters.py [--model haiku] [--out results.json] [1 2 3 4]

Exit code 0 only if every scenario that ran passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

# This script may itself run inside a Claude Code session; a child must not report into, or share
# the session id of, its parent, and the CLI's account email has no place in a memory test.
for _var in ("CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID", "CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION",
             "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_TEE_SDK_STDOUT",
             "CLAUDE_CODE_SYNC_SESSION_REFS", "CLAUDE_CODE_DIAGNOSTICS_FILE", "CLAUDE_CODE_DEBUG",
             "CLAUDE_CODE_INCLUDE_PARTIAL_MESSAGES", "CLAUDE_CODE_USER_EMAIL"):
    os.environ.pop(_var, None)

import commonmemo  # noqa: E402
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query  # noqa: E402
from commonmemo import Memory  # noqa: E402
from commonmemo.adapters.claude_agent_sdk import CommonMemoSessionStore  # noqa: E402
from commonmemo.agent_tools import recall_entries  # noqa: E402

FACT = "my favorite colour is teal"
BUDGET = 0.30


async def ask(prompt: str, options: ClaudeAgentOptions) -> tuple[str, str, float]:
    """Run one prompt; return (reply text, session id, cost in USD)."""
    reply, session, cost = "", "", 0.0
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            reply, session, cost = message.result or "", message.session_id, message.total_cost_usd or 0.0
    return reply, session, cost


def options(cwd: str, model: str, **extra: Any) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(cwd=cwd, model=model, max_turns=2, max_budget_usd=BUDGET, allowed_tools=[],
                              setting_sources=[], **extra)


def drop_local_transcripts(session_id: str) -> int:
    """Delete the CLI's own copy of a session, so a resume can only use the store."""
    gone = 0
    for root in {Path.home() / ".claude" / "projects", Path(os.environ.get("CLAUDE_CONFIG_DIR", "/nonexistent")) / "projects"}:
        for path in root.glob(f"*/{session_id}.jsonl") if root.exists() else []:
            path.unlink()
            gone += 1
    return gone


def events(memory: Memory, source: str) -> list:
    return [e for e in memory.store.events() if getattr(e, "source", None) == source]


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    work = Path(tempfile.mkdtemp(prefix="live-adapters-"))
    db = f"sqlite:///{work / 'memory.db'}"
    cwd = str(work / "project")
    os.makedirs(cwd)
    user = f"live-{uuid.uuid4().hex[:8]}"
    results: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    def record(n: int, name: str, passed: bool, detail: str, cost: float = 0.0, seconds: float = 0.0) -> None:
        results.append({"scenario": n, "name": name, "passed": passed, "detail": detail, "cost_usd": round(cost, 4),
                        "seconds": round(seconds, 1)})
        print(f"[{'PASS' if passed else 'FAIL'}] {n} {name}: {detail}", flush=True)

    wanted = set(args.scenarios or [1, 2, 3, 4])

    if wanted & {1, 2, 3}:
        # 1. the store mirrors a real run
        memory = Memory(db)
        store = CommonMemoSessionStore(memory, user_id=user)
        t = time.time()
        reply, session, cost = await ask(f"Remember this: {FACT}. Reply with just OK.",
                                         options(cwd, args.model, session_store=store))
        state.update(session=session, cost=cost)
        appended = events(memory, "claude_agent_sdk.append")
        # list by the project key the SDK used, read back from the first append event
        project = appended[0].data["key"]["project_key"] if appended else ""
        listed = await store.list_sessions(project) if project else []
        loaded = await store.load({"project_key": project, "session_id": session}) if project else None
        ok = bool(session) and bool(appended) and any(s["session_id"] == session for s in listed) and bool(loaded)
        record(1, "store", ok,
               f"session {session[:8]}, {len(appended)} append events, {len(loaded or [])} entries loaded, "
               f"reply {reply[:40]!r}", cost, time.time() - t)
        state["project"] = project
        memory.close() if hasattr(memory, "close") else None

    if 2 in wanted and state.get("session"):
        # 2. resume from the Log alone: a new Memory, a new store, no local transcript
        gone = drop_local_transcripts(state["session"])
        memory = Memory(db)
        store = CommonMemoSessionStore(memory, user_id=user)
        t = time.time()
        reply, session, cost = await ask("What is my favourite colour? Answer in one word.",
                                         options(cwd, args.model, session_store=store, resume=state["session"]))
        ok = "teal" in reply.lower()
        record(2, "resume", ok, f"{gone} local transcript(s) removed; reply {reply[:60]!r}", cost, time.time() - t)
        memory.close() if hasattr(memory, "close") else None

    if 3 in wanted and state.get("session"):
        # 3. what was said is searchable
        memory = Memory(db)
        entries, _, _ = recall_entries(memory.session("recall-check", user_id=user), "favourite colour")
        found = [text for _, text in entries if "teal" in text.lower()]
        record(3, "recall", bool(found), f"{len(entries)} entries; {len(found)} mention teal")
        memory.close() if hasattr(memory, "close") else None

    if 4 in wanted:
        # 4. instrument(): no store passed, the SDK's default is ours
        memory = Memory(f"sqlite:///{work / 'instrumented.db'}")
        report = commonmemo.instrument(memory, ["claude_agent_sdk"], user_id=user)
        try:
            default = ClaudeAgentOptions().session_store
            t = time.time()
            reply, session, cost = await ask(f"Reply with the single word PONG.", options(cwd, args.model))
            wrote = events(memory, "claude_agent_sdk.append")
            ok = isinstance(default, CommonMemoSessionStore) and bool(wrote)
            record(4, "instrument", ok, f"default store {type(default).__name__}; {len(wrote)} append events; "
                   f"reply {reply[:20]!r}; hooked {report}", cost, time.time() - t)
        finally:
            commonmemo.uninstrument()
        memory.close() if hasattr(memory, "close") else None
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenarios", nargs="*", type=int)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--out")
    args = parser.parse_args()
    results = asyncio.run(run(args))
    total = sum(r["cost_usd"] for r in results)
    print(f"cost ${total:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0 if results and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
