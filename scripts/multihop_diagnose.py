"""Multi-hop diagnosis (#131): is a miss a retrieval failure or an answer failure, and does a larger k help?

Step 1 (no model calls): join the answer step's records (`embed_answers.py`,
`answers.jsonl`) to LoCoMo's evidence labels. Accuracy by category when all the
evidence was among the windows the answerer saw, when only some, and when none.
Step 2 (retrieval only, keyword ranking): share of questions with all the
evidence among the k best windows, k = 8, 16, 24, 32, with how many windows
the evidence spans and the prompt size.

    python scripts/multihop_diagnose.py --answers answers/answers.jsonl --out multihop.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo_benchmark as lb  # noqa: E402
from locomo_enrich import specs, top  # noqa: E402

from commonmemo.ranking import KeywordRanker  # noqa: E402

KS = (8, 16, 24, 32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--answers", default=None, help="answers.jsonl from embed_answers.py (step 1)")
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="multihop.json")
    args = p.parse_args()
    data = lb.load_data(args.data)
    cats = lb.CATEGORIES
    kw = KeywordRanker()
    needs: dict[str, set[int]] = {}
    sweep = {c: {k: {"n": 0, "all": 0, "chars": 0} for k in KS} for c in cats}
    span = {c: [] for c in cats}
    for ci in range(len(data)):
        conv = lb.disguise(data[ci], ci)
        windows = specs(conv)
        bodies = [w["body"] for w in windows]
        owner = {d: i for i, w in enumerate(windows) for d in w["ids"]}
        for qi, q in enumerate(conv["qa"]):
            ev = [e for e in q.get("evidence", []) if e in owner]
            if q["category"] not in cats or "answer" not in q or not ev:
                continue
            need = {owner[e] for e in ev}
            needs[f"{ci}:{qi}"] = need
            span[q["category"]].append(len(need))
            scores = asyncio.run(kw.score(q["question"], bodies))
            for k in KS:
                best = top(scores, k)
                s = sweep[q["category"]][k]
                s["n"] += 1
                s["all"] += need <= set(best)
                s["chars"] += sum(len(bodies[i]) for i in best)
    out: dict = {"sweep": {}, "span": {}}
    print("Evidence spans (windows needed) and keyword all-evidence recall at k\n")
    print("| Category | questions | windows needed (mean) | 1 window | 3+ windows | " + " | ".join(f"k={k}" for k in KS) + " | chars at k=8 / 24 |")
    print("|---|---|---|---|---|" + "---|" * (len(KS) + 1))
    for c, name in cats.items():
        sp = span[c]
        row = [f"{100 * sweep[c][k]['all'] / max(1, sweep[c][k]['n']):.0f}%" for k in KS]
        n = max(1, sweep[c][8]["n"])
        print(f"| {name} | {len(sp)} | {mean(sp):.2f} | {100 * sum(x == 1 for x in sp) / n:.0f}% | "
              f"{100 * sum(x >= 3 for x in sp) / n:.0f}% | " + " | ".join(row) +
              f" | {sweep[c][8]['chars'] // n} / {sweep[c][24]['chars'] // n} |")
        out["sweep"][name] = {str(k): sweep[c][k] for k in KS}
        out["span"][name] = {"n": len(sp), "mean": round(mean(sp), 2)}
    if args.answers:
        recs = [json.loads(line) for line in Path(args.answers).read_text().splitlines() if line.strip()]
        tab: dict = {}
        for r in recs:
            if r.get("error") or r["id"] not in needs:
                continue
            need = needs[r["id"]]
            got = need & set(r["windows"])
            kind = "all" if got == need else ("some" if got else "none")
            t = tab.setdefault(lb.CATEGORIES[r["category"]], {}).setdefault(kind, [0, 0])
            t[0] += bool(r["correct"])
            t[1] += 1
        print("\nAnswers correct by how much of the evidence the answerer saw (all rankers pooled)\n")
        print("| Category | all evidence | some | none |")
        print("|---|---|---|---|")
        for name, t in tab.items():
            print(f"| {name} | " + " | ".join(
                f"{t[k][0]}/{t[k][1]} ({100 * t[k][0] / t[k][1]:.0f}%)" if k in t else "-" for k in ("all", "some", "none")) + " |")
        out["answers"] = tab
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
