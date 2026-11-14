"""Two processes write to one CommonMemo SQLite file at once, under each journal mode (#177).

    python scripts/bench_wal.py [--writers 2] [--sessions 30] [--messages 10] [--reps 3]

Each writer opens its own `Memory` on the same file and records `sessions` closed sessions of
`messages` messages (one writer per harness, as when two agents share a Log). Reports wall time,
messages per second, the longest single record call, and errors (for example "database is locked").
The journal mode is set on the file before the writers start; commonmemo's own default is not changed."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def writer(db: str, name: str, sessions: int, messages: int, q: "mp.Queue[dict]") -> None:
    from commonmemo import AlertPolicy, Memory

    errors, worst, n = [], 0.0, 0
    mem = Memory(f"sqlite:///{db}", project="wal", alerts=AlertPolicy(enabled=False))
    for i in range(sessions):
        try:
            with mem.session(user_id=name) as s:
                for j in range(messages):
                    t = time.perf_counter()
                    s.message("user" if j % 2 == 0 else "assistant", f"{name} session {i} message {j} " + "word " * 40)
                    worst = max(worst, time.perf_counter() - t)
                    n += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}"[:80])
    q.put({"writer": name, "recorded": n, "worst_s": worst, "errors": errors})


def trial(mode: str, writers: int, sessions: int, messages: int) -> dict:
    d = Path(tempfile.mkdtemp(prefix="walbench-"))
    db = str(d / "m.db")
    if mode != "default":
        con = sqlite3.connect(db)
        con.execute(f"PRAGMA journal_mode={mode.split('+')[0]}")
        con.commit()
        con.close()
    q: "mp.Queue[dict]" = mp.Queue()
    ps = [mp.Process(target=writer, args=(db, f"w{k}", sessions, messages, q)) for k in range(writers)]
    t0 = time.perf_counter()
    for p in ps:
        p.start()
    out = [q.get() for _ in ps]
    for p in ps:
        p.join()
    wall = time.perf_counter() - t0
    con = sqlite3.connect(db)
    final = con.execute("PRAGMA journal_mode").fetchone()[0]
    rows = con.execute("SELECT COUNT(*) FROM events").fetchone()[0] if con.execute(
        "SELECT name FROM sqlite_master WHERE name='events'").fetchone() else -1
    con.close()
    return {"mode": mode, "final_mode": final, "wall_s": wall, "recorded": sum(o["recorded"] for o in out),
            "worst_call_s": max(o["worst_s"] for o in out), "errors": sum((o["errors"] for o in out), []), "log_rows": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--writers", type=int, default=2)
    p.add_argument("--sessions", type=int, default=30)
    p.add_argument("--messages", type=int, default=10)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    results = []
    for mode in ("default", "wal"):
        for _ in range(args.reps):
            r = trial(mode, args.writers, args.sessions, args.messages)
            results.append(r)
            print(f"{mode:8} final={r['final_mode']:7} wall {r['wall_s']:.2f}s recorded {r['recorded']} "
                  f"({r['recorded'] / r['wall_s']:.0f}/s) worst call {r['worst_call_s']:.2f}s errors {len(r['errors'])}"
                  + (f" e.g. {r['errors'][0]}" if r["errors"] else ""))
    for mode in ("default", "wal"):
        mine = [r for r in results if r["mode"] == mode]
        print(f"{mode}: median {statistics.median(r['recorded'] / r['wall_s'] for r in mine):.0f} msg/s, "
              f"worst call {max(r['worst_call_s'] for r in mine):.2f}s, errors {sum(len(r['errors']) for r in mine)}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
