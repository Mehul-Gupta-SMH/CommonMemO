"""How fast is a search by meaning with stored window vectors? (#77)

A fake embedder (a deterministic random unit vector per text, 384 dimensions) stands in for a
model, so this measures the index and the arithmetic, not the quality of any embedding. One user,
N messages in sessions of 20 (see `bench_said.py`). Reports, for a new process on the same file
(a new ranker, an empty cache, as a hook is): the texts embedded by the first search (the windows
it scored, at most 500) and by later searches (the request and the probe), the median search with
the vectors stored, and the cost of scoring every window of the user by brute force, with numpy and
without.

    python scripts/bench_vectors.py 10000 100000
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_said as bs  # noqa: E402

import commonmemo.said as said  # noqa: E402
from commonmemo import Memory  # noqa: E402
from commonmemo.ranking import EmbeddingRanker  # noqa: E402

DIM = 384


class Fake:
    def __init__(self) -> None:
        self.texts = 0
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        out = []
        for t in texts:
            rng = random.Random(hashlib.sha1(t.encode()).digest())
            v = [rng.gauss(0, 1) for _ in range(DIM)]
            norm = sum(x * x for x in v) ** 0.5
            out.append([x / norm for x in v])
        return out


def search_once(mem: Memory, query: str) -> float:
    t = time.perf_counter()
    asyncio.run(said.search(mem, query, user_id=bs.USER, agent=bs.AGENT, exclude_session="now", limit=8, threshold=0.0))
    return time.perf_counter() - t


def main() -> None:
    print("| messages | first search embeds | later search embeds | median search (numpy) | median search (plain Python) "
          "| vector size | all windows, numpy | all windows, plain Python |")
    print("|---|---|---|---|---|---|---|---|")
    for n in (int(a) for a in sys.argv[1:] or ["10000"]):
        mem, _ = bs.build(n)
        mem.knowledge.catch_up(mem.store)
        first = Fake()
        mem.ranker = EmbeddingRanker(first)
        search_once(mem, "sister Lena visits")
        embedded_first = first.texts
        stored = mem.knowledge._conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(vec)), 0) AS b FROM window_vectors").fetchone()
        for _ in range(12):  # later searches embed the remaining windows, 500 at a time, until all are stored
            search_once(mem, "sister Lena visits")
        later = Fake()
        mem.ranker = EmbeddingRanker(later)  # "a new process": an empty cache
        times = [search_once(mem, "guest room key") for _ in range(15)]
        medians = {"numpy": statistics.median(times)}
        import builtins

        real = builtins.__import__
        builtins.__import__ = lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "numpy" else real(name, *a, **k)
        try:
            medians["plain"] = statistics.median(search_once(mem, "guest room key") for _ in range(5))
        finally:
            builtins.__import__ = real
        windows = mem.knowledge._conn.execute("SELECT COUNT(*) AS n FROM message_windows").fetchone()["n"]
        rng = random.Random(0)
        q = [rng.gauss(0, 1) for _ in range(DIM)]
        vecs = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(min(windows, 20000))]
        t = time.perf_counter()
        said._cosines(q, vecs)
        np_t = time.perf_counter() - t
        builtins.__import__ = lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "numpy" else real(name, *a, **k)
        try:
            t = time.perf_counter()
            said._cosines(q, vecs[:2000])
            py_t = (time.perf_counter() - t) * windows / 2000
        finally:
            builtins.__import__ = real
        np_all = np_t * windows / len(vecs)
        print(f"| {n:,} | {embedded_first} | {later.texts // max(1, later.calls)} per call, {later.texts} in {later.calls} "
              f"| {1000 * medians['numpy']:.0f} ms | {1000 * medians['plain']:.0f} ms | {stored['b'] / max(1, stored['n']) / 1000:.1f} KB "
              f"| {1000 * np_all:.0f} ms ({windows:,} windows) | {1000 * py_t:.0f} ms |", flush=True)


if __name__ == "__main__":
    main()
