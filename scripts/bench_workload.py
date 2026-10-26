"""The memory workload, measured (#17): what ADR 0001's estimates become on real hardware.

A synthetic agent workload, recorded exactly as the Claude Code hooks record a turn
(`hooks.py`): the user's prompt as a `Message`, the turn's push (`Session.push_turn`, which
also searches the user's past messages and records a trace), each tool call as a `ToolInvoked`
followed by `Session.refresh`, and the reply as an assistant `Message`. Sessions close every
`--turns-per-session` turns (a recap). Tools are drawn from a Zipf-like distribution, so a few
are hot, the way one popular tool hits one counter row (`--skew 0` is uniform; a large skew sends
nearly every call to one tool).

Several worker processes write at once, each with its own `Memory` on the same store, each with
its own users (a user's turns are never concurrent, as with a real agent). Reported:

- events written per second, all workers together;
- p50 / p99 latency to append one event (the log write and the projection update it triggers);
- p50 / p99 latency to build one turn's context (`push_turn`);
- p50 / p99 to close a session;
- storage after the run, and per million turns at that rate;
- optionally (`--cold N`), N hook invocations as Claude Code makes them, a new process each;
- with `--only-tools`, tool calls back to back and nothing else: the hot-counter test (a large
  `--skew` sends every call to one tool row).

    python scripts/bench_workload.py --store sqlite --workers 1 4 --turns 500
    python scripts/bench_workload.py --store postgresql://user@host:5432/db --workers 1 4 8 --skew 0 1.2 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

WORDS = [f"w{i}" for i in range(4000)]
CUM_WEIGHTS = list(__import__("itertools").accumulate(1 / (i + 1) for i in range(len(WORDS))))


def text(rng: random.Random, chars: int) -> str:
    """About `chars` characters of Zipf-distributed words (cheap, so the benchmark measures memory, not itself)."""
    return " ".join(rng.choices(WORDS, cum_weights=CUM_WEIGHTS, k=max(1, chars // 5)))


def tool_weights(tools: int, skew: float) -> list[float]:
    return [1 / (i + 1) ** skew for i in range(tools)]


def worker(args: dict[str, Any], index: int, out: "mp.Queue[dict[str, Any]]") -> None:
    from commonmemo import AlertPolicy, Memory
    from commonmemo.events import ToolInvoked

    rng = random.Random(index)
    # Bulk synthetic writes look like a spike to the default alerts; that isn't what is measured.
    mem = Memory(args["url"], project="bench", alerts=AlertPolicy(enabled=False))
    weights = tool_weights(args["tools"], args["skew"])
    appends: list[float] = []
    contexts: list[float] = []
    closes: list[float] = []
    events = 0
    started = time.perf_counter()
    turn = 0
    session = None
    while turn < args["turns"]:
        if session is None or turn % args["turns_per_session"] == 0:
            if session is not None:
                t = time.perf_counter()
                asyncio.run(session.aclose())
                closes.append(time.perf_counter() - t)
            user = f"u{index}-{rng.randrange(args['users'])}"
            session = mem.session(f"s{index}-{turn}-{uuid.uuid4().hex[:6]}", user_id=user, agent="bench")
            session.__enter__()
        prompt = text(rng, rng.randint(60, 400))
        if args["only_tools"] and args["batch"]:  # the same calls, written `batch` at a time in one transaction
            for _ in range(0, args["tools_per_turn"], args["batch"]):
                calls = [ToolInvoked(name=f"tool{rng.choices(range(args['tools']), weights)[0]}",
                                     arguments={"input": "x"}, result="ok", duration_ms=1.0)
                         for _ in range(min(args["batch"], args["tools_per_turn"]))]
                t = time.perf_counter()
                session.record_many(calls)
                appends.extend([(time.perf_counter() - t) / len(calls)] * len(calls))
                events += len(calls)
            turn += 1
            continue
        if args["only_tools"]:  # the hot-counter test: tool calls back to back, nothing else
            for _ in range(args["tools_per_turn"]):
                name = f"tool{rng.choices(range(args['tools']), weights)[0]}"
                t = time.perf_counter()
                session.record(ToolInvoked(name=name, arguments={"input": "x"}, result="ok", duration_ms=1.0))
                appends.append(time.perf_counter() - t)
                events += 1
            turn += 1
            continue
        t = time.perf_counter()
        session.message("user", prompt)
        appends.append(time.perf_counter() - t)
        t = time.perf_counter()
        asyncio.run(session.push_turn(prompt, recent=0))
        contexts.append(time.perf_counter() - t)
        events += 2  # the message and the push's trace
        for _ in range(args["tools_per_turn"]):
            name = f"tool{rng.choices(range(args['tools']), weights)[0]}"
            failed = rng.random() < 0.05
            result = None if failed else text(rng, rng.randint(args["payload"] // 2, args["payload"] * 3 // 2))
            t = time.perf_counter()
            session.record(ToolInvoked(name=name, arguments={"input": text(rng, 80)}, result=result,
                                       error="boom" if failed else None, duration_ms=rng.uniform(5, 500)))
            appends.append(time.perf_counter() - t)
            asyncio.run(session.refresh(result or ""))
            events += 1
        t = time.perf_counter()
        session.message("assistant", text(rng, rng.randint(100, 800)))
        appends.append(time.perf_counter() - t)
        events += 1
        turn += 1
    if session is not None:
        t = time.perf_counter()
        asyncio.run(session.aclose())
        closes.append(time.perf_counter() - t)
    out.put({"appends": appends, "contexts": contexts, "closes": closes, "events": events, "turns": turn,
             "seconds": time.perf_counter() - started})


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def storage(url: str, path: Path | None) -> int:
    if path is not None:
        return sum(f.stat().st_size for f in path.parent.glob(path.name + "*"))
    import psycopg

    base, _, schema = url.partition("?schema=")
    with psycopg.connect(base) as c:
        row = c.execute("SELECT COALESCE(SUM(pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename))), 0) "
                        "FROM pg_tables WHERE schemaname = %s", (schema,)).fetchone()
    return int(row[0])


def cold_hooks(url: str, n: int) -> list[float]:
    """N hook invocations as Claude Code makes them: a new `commonmemo hook` process per prompt."""
    times = []
    env = dict(os.environ, COMMONMEMO_STORE=url)
    for i in range(n):
        payload = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": f"cold{i}", "cwd": ".",
                              "prompt": "where did we leave the w12 w40 deploy"})
        t = time.perf_counter()
        subprocess.run([sys.executable, "-m", "commonmemo", "hook", "--harness", "claude-code", "--user", "u0-0",
                        "--no-environment"], input=payload, text=True, capture_output=True, env=env, check=True)
        times.append(time.perf_counter() - t)
    return times


def run(store: str, workers: int, skew: float, a: argparse.Namespace) -> dict[str, Any]:
    path = None
    if store == "sqlite":
        path = Path(tempfile.mkdtemp(prefix="cm-bench-")) / "memory.db"
        url = f"sqlite:///{path}"
    else:
        url = f"{store}{'&' if '?' in store else '?'}schema=bench_{uuid.uuid4().hex[:8]}"
    args = {"url": url, "turns": a.turns, "turns_per_session": a.turns_per_session, "users": a.users,
            "tools": a.tools, "skew": skew, "tools_per_turn": a.tools_per_turn, "payload": a.payload,
            "only_tools": a.only_tools, "batch": a.batch}
    ctx = mp.get_context("spawn")
    q: "mp.Queue[dict[str, Any]]" = ctx.Queue()
    # One process creates the schema first, so the workers measure writes, not setup.
    from commonmemo import Memory

    Memory(url, project="bench")
    started = time.perf_counter()
    procs = [ctx.Process(target=worker, args=(args, i, q)) for i in range(workers)]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()
    wall = time.perf_counter() - started
    appends = [x for r in results for x in r["appends"]]
    contexts = [x for r in results for x in r["contexts"]]
    closes = [x for r in results for x in r["closes"]]
    events = sum(r["events"] for r in results)
    turns = sum(r["turns"] for r in results)
    size = storage(url, path)
    row = {"store": "SQLite" if path else "Postgres", "workers": workers, "skew": skew, "turns": turns,
           "events": events, "events_per_s": events / wall, "turns_per_s": turns / wall,
           "append_p50_ms": 1000 * pct(appends, 50), "append_p99_ms": 1000 * pct(appends, 99),
           "context_p50_ms": 1000 * pct(contexts, 50), "context_p99_ms": 1000 * pct(contexts, 99),
           "close_p50_ms": 1000 * pct(closes, 50), "close_p99_ms": 1000 * pct(closes, 99),
           "bytes": size, "bytes_per_turn": size / max(1, turns), "gb_per_million_turns": size / max(1, turns) * 1e6 / 1e9}
    if a.cold:
        cold = cold_hooks(url, a.cold)
        row.update(cold_p50_ms=1000 * statistics.median(cold), cold_p99_ms=1000 * pct(cold, 99))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default="sqlite", help="`sqlite` (a new temp file) or a postgresql:// URL (a new schema)")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    p.add_argument("--skew", type=float, nargs="+", default=[1.2], help="tool popularity: 0 uniform, larger is hotter")
    p.add_argument("--turns", type=int, default=300, help="turns per worker")
    p.add_argument("--turns-per-session", type=int, default=20)
    p.add_argument("--users", type=int, default=5, help="users per worker")
    p.add_argument("--tools", type=int, default=40)
    p.add_argument("--tools-per-turn", type=int, default=8)
    p.add_argument("--payload", type=int, default=1200, help="mean characters of a tool result")
    p.add_argument("--only-tools", action="store_true",
                   help="record tool calls only, back to back (isolates the tool-counter row under --skew)")
    p.add_argument("--batch", type=int, default=0,
                   help="with --only-tools, write this many tool calls per transaction (Session.record_many)")
    p.add_argument("--cold", type=int, default=0, help="also time this many hook processes, as Claude Code starts them")
    p.add_argument("--json", default=None, help="write the rows here too")
    a = p.parse_args()
    print(f"python {platform.python_version()}, sqlite {sqlite3.sqlite_version}, {os.cpu_count()} CPUs, {platform.platform()}",
          file=sys.stderr)
    rows = []
    print("| store | workers | tool skew | turns | events/s | turns/s | append p50 / p99 | context p50 / p99 "
          "| close p50 / p99 | bytes/turn | GB per 1M turns |" + (" hook process p50 / p99 |" if a.cold else ""))
    print("|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if a.cold else ""))
    for workers in a.workers:
        for skew in a.skew:
            r = run(a.store, workers, skew, a)
            rows.append(r)
            line = (f"| {r['store']} | {workers} | {skew:g} | {r['turns']:,} | {r['events_per_s']:,.0f} | {r['turns_per_s']:,.1f} "
                    f"| {r['append_p50_ms']:.1f} / {r['append_p99_ms']:.1f} ms | {r['context_p50_ms']:.1f} / {r['context_p99_ms']:.1f} ms "
                    f"| {r['close_p50_ms']:.0f} / {r['close_p99_ms']:.0f} ms | {r['bytes_per_turn']:,.0f} | {r['gb_per_million_turns']:.1f} |")
            if a.cold:
                line += f" {r['cold_p50_ms']:.0f} / {r['cold_p99_ms']:.0f} ms |"
            print(line, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
