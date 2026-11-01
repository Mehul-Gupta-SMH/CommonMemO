"""Answer step of the embedder bake-off (#125, ADR 0005): do better windows give more correct answers?

`embed_bakeoff.py` picks the 8 best windows a question gets from each ranker and
writes them to `bakeoff-rankings.json`. This script puts exactly those windows in
front of a model, one condition per ranker, and has a judge grade the answers,
as `locomo_benchmark.py` does (same prompt wording, same disguise, same questions:
12 a category, seed 7). The bar (ADR 0005): at least 11 more correct answers of 450
than keyword ranking for a default; fewer makes it opt-in.

    python scripts/embed_answers.py --rankings bakeoff-rankings.json \\
        --rankers keyword,potion-base-32M+keyword,all-MiniLM-L6-v2+keyword --model haiku --out answers.json

Needs the `claude` CLI (Claude Code); every call is one headless session with no
tools. Resumable: answers are cached in `--work/answers.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
import locomo_benchmark as lb  # noqa: E402
from locomo_enrich import sign_p, specs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rankings", default="bakeoff-rankings.json")
    p.add_argument("--rankers", default="keyword")
    p.add_argument("--model", default="haiku")
    p.add_argument("--judge-model", default="haiku")
    p.add_argument("--budget", type=float, default=12.0, help="stop starting new calls past this many USD")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data", default=None)
    p.add_argument("--work", default=None, help="working dir; reuse it to resume")
    p.add_argument("--out", default="answers.json")
    args = p.parse_args()

    saved = json.loads(Path(args.rankings).read_text())
    rankings = saved["rankings"]
    rankers = [r for r in args.rankers.split(",") if r]
    for r in rankers:
        missing = [q for q, by in rankings.items() if r not in by]
        if missing:
            p.error(f"ranker {r!r} has no ranking for {len(missing)} questions in {args.rankings}")
    data = lb.load_data(args.data)
    convs: dict[int, dict[str, Any]] = {}
    wins: dict[int, list[dict[str, Any]]] = {}
    questions: list[dict[str, Any]] = []
    for ci in sorted({int(i.split(":")[0]) for i in rankings}):
        convs[ci] = lb.disguise(data[ci], ci)
        wins[ci] = specs(convs[ci])
        questions += [q for q in lb.sample(convs[ci], ci, saved["per_category"], saved["seed"]) if q["id"] in rankings]
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="commonmemo-embed-answers-"))
    work.mkdir(parents=True, exist_ok=True)
    bare = work / "bare"
    bare.mkdir(exist_ok=True)
    cache_file = work / "answers.jsonl"
    cached = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            row = json.loads(line)
            cached[f"{row['condition']}|{row['id']}"] = row
    print(f"{len(questions)} questions x {len(rankers)} rankers; work dir {work}", file=sys.stderr)

    def run(task: tuple[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
        ranker, q = task
        key = f"{ranker}|{q['id']}"
        if key in cached:
            return cached[key]
        if lb.SPENT["usd"] >= args.budget:
            return None
        ci = int(q["id"].split(":")[0])
        picked = sorted(rankings[q["id"]][ranker])
        texts = [wins[ci][i]["text"] for i in picked]
        r = lb.ask(bare, lb.question_prompt(convs[ci], q["question"], "rag", texts), model=args.model)
        ok, judge_cost = lb.judge(q["question"], q["gold"], r["reply"], args.judge_model) if not r["error"] else (None, 0.0)
        row = {"condition": ranker, **q, "windows": picked, "answer": r["reply"], "correct": ok,
               "cost": round(r["cost"], 5), "judge_cost": round(judge_cost, 5), "error": r["error"]}
        with lb._lock:
            with cache_file.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"  [{ranker} {q['id']}] {'ok' if ok else 'wrong' if ok is False else '??'}", file=sys.stderr)
        return row

    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(run, [(r, q) for q in questions for r in rankers]) if r]

    by = {(r["condition"], r["id"]): r["correct"] for r in rows}
    complete = [q["id"] for q in questions if all(by.get((r, q["id"])) is not None for r in rankers)]
    summary: dict[str, Any] = {}
    for r in rankers:
        mine = [x for x in rows if x["condition"] == r and x["id"] in complete]
        cell = {"correct": sum(bool(x["correct"]) for x in mine), "of": len(mine), "by_category": {}}
        for cat, name in lb.CATEGORIES.items():
            sub = [x for x in mine if x["category"] == cat]
            cell["by_category"][name] = {"correct": sum(bool(x["correct"]) for x in sub), "of": len(sub)}
        if r != rankers[0]:
            a = sum(1 for i in complete if by[(r, i)] and not by[(rankers[0], i)])
            b = sum(1 for i in complete if by[(rankers[0], i)] and not by[(r, i)])
            cell.update({"only": a, "baseline_only": b, "p": sign_p(a, b),
                         "over_baseline": cell["correct"] - summary[rankers[0]]["correct"]})
        summary[r] = cell
    out = {"date": time.strftime("%Y-%m-%d"), "baseline": rankers[0], "model": args.model, "judge_model": args.judge_model,
           "questions": len(complete), "dropped_incomplete": len(questions) - len(complete),
           "cost_usd": round(lb.SPENT["usd"], 4), "claude_calls": lb.SPENT["calls"], "claude_models": sorted(lb.MODELS),
           "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    cats = list(lb.CATEGORIES.values())
    print("\n| Ranker | Correct | " + " | ".join(cats) + " | vs " + rankers[0] + " (only / only, p) |")
    print("|---|---|" + "---|" * (len(cats) + 1))
    for r, c in summary.items():
        cmp = "" if r == rankers[0] else f"{c['over_baseline']:+d} ({c['only']} / {c['baseline_only']}, p = {c['p']:.4f})"
        print(f"| {r} | {c['correct']}/{c['of']} | " + " | ".join(f"{v['correct']}/{v['of']}" for v in c["by_category"].values())
              + f" | {cmp} |")
    print(f"\n${lb.SPENT['usd']:.2f} over {lb.SPENT['calls']} claude calls; {len(complete)} questions graded for every ranker; results in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
