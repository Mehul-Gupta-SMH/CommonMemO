"""HyDE for window retrieval (#138): does a hypothetical answer find more of the evidence?

For every LoCoMo question with evidence labels, Haiku writes a short hypothetical answer from the question alone
(cached in `--work/hyde.jsonl`). Windows are ranked by keyword score for the question (baseline), the hypothesis
alone, and question + hypothesis. Reports the share of questions with all the evidence among the top k windows,
by category, with a paired sign test against the baseline. Retrieval only; no answer run.

    python scripts/hyde_retrieval.py --model haiku --budget 1.5 --out hyde.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo_benchmark as lb  # noqa: E402
from locomo_enrich import sign_p, specs, top  # noqa: E402

from commonmemo.ranking import KeywordRanker  # noqa: E402

KS = (8, 16, 24)
ARMS = ("question", "hypothesis", "question+hypothesis")
PROMPT = ("Two people, {a} and {b}, talked over several months. Without having seen their conversations, write the "
          "most plausible answer to this question as one or two plain sentences, naming the kinds of things, "
          "places, people and activities such an answer would mention. Do not say you cannot know.\n\n"
          "Question: {q}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=None)
    p.add_argument("--model", default="haiku")
    p.add_argument("--budget", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--work", default=None)
    p.add_argument("--out", default="hyde.json")
    args = p.parse_args()
    data = lb.load_data(args.data)
    work = Path(args.work or "hyde-work")
    work.mkdir(parents=True, exist_ok=True)
    bare = work / "bare"
    bare.mkdir(exist_ok=True)
    cache_file = work / "hyde.jsonl"
    cache: dict[str, str] = {}
    if cache_file.exists():
        cache = {r["key"]: r["hypothesis"] for r in map(json.loads, cache_file.read_text().splitlines())}

    items: list[dict[str, Any]] = []
    for ci in range(len(data)):
        conv = lb.disguise(data[ci], ci)
        windows = specs(conv)
        owner = {d: i for i, w in enumerate(windows) for d in w["ids"]}
        a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
        for qi, q in enumerate(conv["qa"]):
            ev = [e for e in q.get("evidence", []) if e in owner]
            if q["category"] not in lb.CATEGORIES or "answer" not in q or not ev:
                continue
            items.append({"key": f"{ci}:{qi}", "ci": ci, "cat": q["category"], "q": q["question"], "a": a, "b": b,
                          "need": {owner[e] for e in ev}, "bodies": [w["body"] for w in windows]})

    def hyp(it: dict[str, Any]) -> None:
        if it["key"] in cache or lb.SPENT["usd"] >= args.budget:
            return
        r = lb.ask(bare, PROMPT.format(a=it["a"], b=it["b"], q=it["q"]), model=args.model)
        if r["error"] or not r["reply"].strip():
            return
        cache[it["key"]] = r["reply"].strip()
        with lb._lock, cache_file.open("a") as f:
            f.write(json.dumps({"key": it["key"], "hypothesis": cache[it["key"]]}) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(hyp, items))
    items = [it for it in items if it["key"] in cache]
    print(f"{len(items)} questions with a hypothesis; ${lb.SPENT['usd']:.2f} over {lb.SPENT['calls']} calls", file=sys.stderr)

    kw = KeywordRanker()
    hits: dict[str, dict[int, dict[str, list[bool]]]] = {c: {k: {m: [] for m in ARMS} for k in KS} for c in (0, *lb.CATEGORIES)}
    for it in items:
        h = cache[it["key"]]
        texts = {"question": it["q"], "hypothesis": h, "question+hypothesis": f"{it['q']} {h}"}
        for arm, text in texts.items():
            scores = asyncio.run(kw.score(text, it["bodies"]))
            for k in KS:
                ok = it["need"] <= set(top(scores, k))
                for cat in (0, it["cat"]):
                    hits[cat][k][arm].append(ok)
        # keep order aligned across arms: items iterate once, arms append in the same order

    names = {0: "all", **lb.CATEGORIES}
    out: dict[str, Any] = {"n": len(items), "cost_usd": round(lb.SPENT["usd"], 4), "results": {}}
    print("| k | category | n | " + " | ".join(ARMS) + " | q+h vs q (+/-, p) |")
    print("|---|---|---|" + "---|" * (len(ARMS) + 1))
    for k in KS:
        for cat, name in names.items():
            row = hits[cat][k]
            n = len(row["question"])
            base, comb = row["question"], row["question+hypothesis"]
            w = sum(c and not b for b, c in zip(base, comb))
            l = sum(b and not c for b, c in zip(base, comb))
            pct = {arm: round(100 * sum(row[arm]) / n, 1) for arm in ARMS}
            out["results"][f"{name}@{k}"] = {"n": n, **pct, "gain": w, "loss": l, "p": round(sign_p(w, l), 4)}
            print(f"| {k} | {name} | {n} | " + " | ".join(f"{pct[a]}%" for a in ARMS) + f" | +{w}/-{l}, p={sign_p(w, l):.3f} |")
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
