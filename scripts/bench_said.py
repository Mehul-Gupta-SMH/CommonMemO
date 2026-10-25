"""How long does a search of a user's past messages take as the history grows? (#65)

One user, N messages in sessions of 20, random words from a Zipf-like
vocabulary, with a few planted needles ("the zebra enclosure code is 4471").
Reports the time to build the projections (the index, where there is one),
the median and slowest of 30 `said.search` calls, and whether the needles are
found. Uses only the public API, so it runs against any version of the library
(`PYTHONPATH=/path/to/other/src python scripts/bench_said.py`).

    python scripts/bench_said.py 10000 100000 500000
"""

from __future__ import annotations

import asyncio
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from commonmemo import Memory, MemoryPolicy
from commonmemo.events import Message
from commonmemo.said import search

USER, AGENT = "bench", "chat"
NEEDLES = [("zebra enclosure code", "the zebra enclosure code is 4471"),
           ("sister Lena visits", "my sister Lena visits on Friday 14 November"),
           ("guest room key", "the guest room key is under the blue pot")]


def build(n: int, chunk: int = 50_000) -> tuple[Memory, float]:
    rng = random.Random(1)
    vocab = [f"w{i}" for i in range(3000)]
    weights = [1 / (i + 1) for i in range(len(vocab))]
    mem = Memory(f"sqlite:///{Path(tempfile.mkdtemp()) / 'bench.db'}", policy=MemoryPolicy(messages=8))
    needle_at = {rng.randrange(n // 10, n): text for _, text in NEEDLES}
    spent = 0.0
    for start in range(0, n, chunk):
        for i in range(start, min(n, start + chunk)):
            text = needle_at.get(i) or " ".join(rng.choices(vocab, weights, k=rng.randint(6, 18)))
            mem.store.append(Message(role="user" if i % 2 == 0 else "assistant", content=text)
                             .bind(session_id=f"s{i // 20}", user_id=USER, agent=AGENT))
        t = time.perf_counter()
        mem.knowledge.catch_up(mem.store)
        spent += time.perf_counter() - t
    return mem, spent


def main() -> None:
    print("| messages | projections built in | median search | slowest of 30 | needles found |")
    print("|---|---|---|---|---|")
    for n in (int(a) for a in sys.argv[1:] or ["10000"]):
        mem, built = build(n)
        queries = [q for q, _ in NEEDLES] * 10
        times, found = [], 0
        for q in queries:
            t = time.perf_counter()
            windows, _ = asyncio.run(search(mem, q, user_id=USER, agent=AGENT, exclude_session="now", limit=8))
            times.append(time.perf_counter() - t)
            found += any(w for w in windows if any(word in " ".join(w.lines) for word in ("4471", "Lena visits", "blue pot")))
        print(f"| {n:,} | {built:.1f} s | {1000 * statistics.median(times):.1f} ms | {1000 * max(times):.1f} ms | {found}/{len(queries)} |",
              flush=True)


if __name__ == "__main__":
    main()
