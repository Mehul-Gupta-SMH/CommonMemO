"""Bake-off: does a very small local embedder find LoCoMo evidence better than keyword ranking? (#125, ADR 0005)

No answer calls and no API key: this measures retrieval only, the "evidence
recall" of #66 / #67. For every category 1 to 4 question that LoCoMo labels
with evidence, are all of its evidence windows (6 turns, as in
`locomo_benchmark.rag_windows`), or any, among the 8 best? Ranked by:

  keyword            the memory's default ranker
  <model>            cosine of the question and window vectors
  <model>+keyword    half normalised keyword score, half cosine

for each model in MODELS. Also: install-free cost numbers (load time, ms per
text, ms per query over the windows), and the ranked top 8 of every question
for the best ranker, so the answer step (Haiku through Claude Code) can be run
afterwards on exactly those windows.

    pip install numpy model2vec fastembed
    python scripts/embed_bakeoff.py --out bakeoff.json

A model that cannot be loaded is reported and skipped, never faked. The
decision bar (ADR 0005): +5 points of all-evidence recall over keyword ranking
for a default, a smaller measured gain for opt-in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo_benchmark as lb  # noqa: E402
from locomo_enrich import sign_p, specs, top, wilson  # noqa: E402

from commonmemo.ranking import KeywordRanker  # noqa: E402

# name -> (library, model id). Revisions are recorded from the loaded model, not assumed.
MODELS = {
    "potion-base-8M": ("model2vec", "minishlab/potion-base-8M"),
    "potion-base-32M": ("model2vec", "minishlab/potion-base-32M"),
    "all-MiniLM-L6-v2": ("fastembed", "sentence-transformers/all-MiniLM-L6-v2"),
    "bge-small-en-v1.5": ("fastembed", "BAAI/bge-small-en-v1.5"),
}


def load(lib: str, model_id: str) -> Callable[[list[str]], Any]:
    if lib == "model2vec":
        from model2vec import StaticModel
        m = StaticModel.from_pretrained(model_id)
        return lambda texts: m.encode(texts)
    from fastembed import TextEmbedding
    m = TextEmbedding(model_id)
    return lambda texts: list(m.embed(texts))


def cosine_scores(np: Any, q: Any, mat: Any) -> list[float]:
    qn = q / (np.linalg.norm(q) or 1.0)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return (mn @ qn).tolist()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=None)
    p.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--out", default="bakeoff.json")
    args = p.parse_args()
    import numpy as np

    data = lb.load_data(args.data)
    kw = KeywordRanker()
    cats = lb.CATEGORIES
    rankers: dict[str, dict[str, Any]] = {"keyword": {}}
    embedders: dict[str, Callable[[list[str]], Any]] = {}
    report: dict[str, Any] = {"models": {}}
    for name in [m for m in args.models.split(",") if m]:
        lib, model_id = MODELS[name]
        t = time.time()
        try:
            embedders[name] = load(lib, model_id)
            embedders[name](["warm up"])
        except Exception as exc:  # noqa: BLE001 -- reported, never faked
            report["models"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"skip {name}: {exc}", file=sys.stderr)
            continue
        report["models"][name] = {"library": lib, "id": model_id, "load_s": round(time.time() - t, 2),
                                  "embed_ms_per_text": 0.0, "query_ms": 0.0, "texts": 0}
        rankers[name] = {}
        rankers[f"{name}+keyword"] = {}
    stats = {r: {c: {"n": 0, "all": 0, "any": 0} for c in cats} for r in rankers}
    paired: dict[str, list[int]] = {r: [0, 0] for r in rankers if r != "keyword"}
    ranked_out: dict[str, dict[str, list[int]]] = {r: {} for r in rankers}

    for ci in (int(c) for c in args.conversations.split(",")):
        conv = lb.disguise(data[ci], ci)
        windows = specs(conv)
        bodies = [w["body"] for w in windows]
        owner = {d: i for i, w in enumerate(windows) for d in w["ids"]}
        mats = {}
        for name, emb in embedders.items():
            t = time.time()
            mats[name] = np.asarray(emb(bodies), dtype="float32")
            info = report["models"][name]
            info["embed_ms_per_text"] += (time.time() - t) * 1000
            info["texts"] += len(bodies)
        for qi, q in enumerate(conv["qa"]):
            cat = q["category"]
            evidence = [e for e in q.get("evidence", []) if e in owner]
            if cat not in cats or "answer" not in q or not evidence:
                continue
            need = {owner[e] for e in evidence}
            kscores = asyncio.run(kw.score(q["question"], bodies))
            top_k = max(kscores) or 1.0
            kn = [s / top_k for s in kscores]
            scores = {"keyword": kscores}
            for name, emb in embedders.items():
                t = time.time()
                qv = np.asarray(emb([q["question"]]), dtype="float32")[0]
                cs = cosine_scores(np, qv, mats[name])
                report["models"][name]["query_ms"] += (time.time() - t) * 1000
                scores[name] = cs
                scores[f"{name}+keyword"] = [0.5 * a + 0.5 * b for a, b in zip(kn, cs)]
            hits = {}
            for r, s in scores.items():
                best = top(s)
                hits[r] = (need <= set(best), bool(need & set(best)))
                st = stats[r][cat]
                st["n"] += 1
                st["all"] += hits[r][0]
                st["any"] += hits[r][1]
                ranked_out[r][f"{ci}:{qi}"] = best
            for r in paired:
                paired[r][0] += hits[r][0] and not hits["keyword"][0]
                paired[r][1] += hits["keyword"][0] and not hits[r][0]

    n_q = {r: sum(stats[r][c]["n"] for c in cats) for r in rankers}
    for name, info in report["models"].items():
        if "error" in info:
            continue
        info["embed_ms_per_text"] = round(info["embed_ms_per_text"] / max(1, info["texts"]), 3)
        info["query_ms"] = round(info["query_ms"] / max(1, n_q["keyword"]), 3)

    print("\nAll the evidence among the 8 best windows / any of it\n")
    print("| Ranker | " + " | ".join(cats.values()) + " | all questions | vs keyword (only / only, p) |")
    print("|---|" + "---|" * (len(cats) + 2))
    result: dict[str, Any] = {"rankers": {}, "models": report["models"]}
    base = sum(stats["keyword"][c]["all"] for c in cats) / max(1, n_q["keyword"])
    for r in rankers:
        cells = [f"{stats[r][c]['all']}/{stats[r][c]['n']} ({100 * stats[r][c]['all'] / max(1, stats[r][c]['n']):.0f}%)"
                 for c in cats]
        ta = sum(stats[r][c]["all"] for c in cats)
        ty = sum(stats[r][c]["any"] for c in cats)
        n = n_q[r]
        lo, hi = wilson(ta, n) if n else (0, 0)
        cmp = "" if r == "keyword" else f"{paired[r][0]} / {paired[r][1]}, p = {sign_p(*paired[r]):.4f}"
        print(f"| {r} | " + " | ".join(cells) + f" | {ta}/{n} ({100 * ta / max(1, n):.1f}%, {100 * lo:.0f}-{100 * hi:.0f}%), any {100 * ty / max(1, n):.0f}% | {cmp} |")
        result["rankers"][r] = {"all": ta, "any": ty, "n": n, "points_over_keyword": round(100 * (ta / max(1, n) - base), 2),
                                "by_category": {cats[c]: stats[r][c] for c in cats},
                                **({"only": paired[r][0], "keyword_only": paired[r][1], "p": sign_p(*paired[r])} if r != "keyword" else {})}
    print("\nCost per model\n")
    for name, info in report["models"].items():
        print(f"- {name}: " + (info["error"] if "error" in info else
              f"load {info['load_s']} s, {info['embed_ms_per_text']} ms a window, {info['query_ms']} ms a question"))
    best = max((r for r in rankers), key=lambda r: result["rankers"][r]["all"])
    result["best"] = best
    result["ranked_top8"] = {best: ranked_out[best], "keyword": ranked_out["keyword"]}
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nbest: {best} ({result['rankers'][best]['points_over_keyword']:+.1f} points over keyword); results in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
