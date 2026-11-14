"""LongMemEval live benchmark: no memory vs naive retrieval vs memory as shipped, with Claude Code as the model (#173).

Not part of the test suite or CI -- it spends model tokens through the `claude` CLI.
LongMemEval (Wu et al., 2024) gives each question its own chat history of about 50
sessions (the `longmemeval_s` file: roughly 115k tokens) and asks a question about it.
The cleaned data is downloaded by hand (nothing is vendored):
huggingface.co/datasets/xiaowu0162/longmemeval-cleaned, file `longmemeval_s_cleaned.json`.

Conditions, each question in its own headless `claude -p` session (reusing `locomo_benchmark`):

| Condition | What the model has |
|---|---|
| `none` | nothing but today's date: the floor |
| `rag` | the history cut into windows of 6 turns, the 8 that share the most words with the question pasted in (the same keyword ranker memory uses). No memory, no tools |
| `memory` | CommonMemo as shipped (24 windows pushed with each turn and returned by `memory_recall`), every session recorded and closed, read through the hooks and the MCP tools, read-only, extraction off |

The whole history (about 115k tokens a question) is not a condition: it does not fit the budget.
Abstention questions (`_abs` ids) are left out. Contamination guard: the years 2022, 2023 and 2024
become 2033, 2034 and 2052 (same weekdays) in the history, the question date, the question and the
answer. That defeats lookup by date, not recognition of the question set.

    python scripts/longmemeval_benchmark.py --data longmemeval_s_cleaned.json --per-type 20 --budget 5
    python scripts/longmemeval_benchmark.py --data longmemeval_s_cleaned.json --ingest-only --per-type 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo_benchmark as lb  # noqa: E402  (reuses ingest, project, ask, rag_windows)
from commonmemo import said  # noqa: E402

CONDITIONS = ("none", "rag", "memory")
YEARS = {"2022": "2033", "2023": "2034", "2024": "2052"}


def sample(data: list[dict[str, Any]], per_type: int, seed: int) -> list[dict[str, Any]]:
    """A seeded sample, `per_type` questions of each type, abstention questions left out."""
    rng = random.Random(seed)
    out = []
    for t in sorted({x["question_type"] for x in data}):
        pool = [x for x in data if x["question_type"] == t and not x["question_id"].endswith("_abs")]
        rng.shuffle(pool)
        out += pool[:per_type]
    return out


def disguise(item: dict[str, Any]) -> dict[str, Any]:
    text = re.sub(r"\b(2022|2023|2024)\b", lambda m: YEARS[m.group(1)], json.dumps(item))
    return json.loads(text)


def locomo_date(date: str) -> str:
    """"2034/04/27 (Thu) 10:25" as "10:25 am on 27 April, 2034", the form `locomo_benchmark` reads."""
    t = time.strptime(re.sub(r" \(\w+\)", "", date), "%Y/%m/%d %H:%M")
    return time.strftime("%I:%M %p on %d %B, %Y", t).replace("AM", "am").replace("PM", "pm")


def as_conv(item: dict[str, Any]) -> dict[str, Any]:
    """The history in the shape `locomo_benchmark` reads: user is speaker_a, assistant speaker_b."""
    c: dict[str, Any] = {"speaker_a": "User", "speaker_b": "Assistant"}
    for n, (date, turns) in enumerate(zip(item["haystack_dates"], item["haystack_sessions"]), 1):
        c[f"session_{n}_date_time"] = locomo_date(date)
        c[f"session_{n}"] = [{"speaker": "User" if t["role"] == "user" else "Assistant", "text": t["content"]} for t in turns]
    return {"conversation": c}


def question_prompt(item: dict[str, Any], condition: str, windows: Optional[list[str]]) -> str:
    q = (f"Today is {item['question_date']}. Question about your earlier chats with this user: {item['question']}\n"
         "Answer briefly (a name, number, date or short phrase; for advice questions, a short recommendation "
         "that fits what they told you before). If you cannot tell, make your best guess.")
    if condition == "rag":
        return f"You have no tools. Here are excerpts from your earlier chats that may be relevant.\n\n{lb.rag_context(windows or [], item['question'])}\n\n{q}"
    if condition == "memory":
        return f"You have memory of your earlier chats with this user. Use it.\n\n{q}"
    return f"You have no tools and have not seen your earlier chats.\n\n{q}"


def judge(item: dict[str, Any], answer: str, model: Optional[str]) -> tuple[Optional[bool], float]:
    kind = ("The gold text describes what a good answer should take into account; the answer is CORRECT if it "
            "uses that information, WRONG if it ignores it or gives generic advice." if item["question_type"] == "single-session-preference"
            else "The answer is CORRECT if it contains the gold answer's information (extra detail is fine; for "
                 "counts of days, weeks or months an off-by-one error is fine). It is WRONG if it misses, contradicts "
                 "or says it doesn't know. For a knowledge-update question the answer must give the updated, latest value.")
    prompt = ("You grade answers to questions about a user's earlier chats with an assistant.\n"
              f"Question: {item['question']}\nGold answer: {item['answer']}\nAnswer to grade: {answer}\n\n"
              f"{kind} Reply with exactly one word: CORRECT or WRONG.")
    with tempfile.TemporaryDirectory(prefix="lme-judge-") as d:
        r = lb.ask(Path(d), prompt, model=model)
    verdict = r["reply"].strip().upper()
    return (None if r["error"] or verdict not in ("CORRECT", "WRONG") else verdict == "CORRECT"), r["cost"]


def sign_test(a: list[bool], b: list[bool]) -> dict[str, Any]:
    """Exact two-sided sign test of paired outcomes, a against b."""
    plus = sum(x and not y for x, y in zip(a, b))
    minus = sum(y and not x for x, y in zip(a, b))
    n = plus + minus
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(plus, minus) + 1)) / 2 ** n)
    return {"plus": plus, "minus": minus, "p": round(p, 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="longmemeval_s_cleaned.json")
    p.add_argument("--per-type", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--model", default="haiku")
    p.add_argument("--judge-model", default="haiku")
    p.add_argument("--budget", type=float, default=5.0, help="stop starting new calls past this many USD")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--work", default=None, help="working dir; reuse it to resume")
    p.add_argument("--out", default="longmemeval-results.json")
    p.add_argument("--ingest-only", action="store_true")
    p.add_argument("--types", default="", help="comma-separated question types to keep (default: all six)")
    p.add_argument("--line-chars", type=int, default=0,
                   help="characters kept of each message in a window (COMMONMEMO_LINE_CHARS; 0 leaves the default, 400)")
    args = p.parse_args()
    if args.line_chars:
        os.environ["COMMONMEMO_LINE_CHARS"] = str(args.line_chars)  # for the MCP server and hooks the questions start
        said.LINE_CHARS = args.line_chars  # for the ingest in this process
    conditions = [c for c in args.conditions.split(",") if c]
    if set(conditions) - set(CONDITIONS):
        p.error("unknown conditions")
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    keep = {t for t in args.types.split(",") if t}
    items = [disguise(x) for x in sample(data, args.per_type, args.seed) if not keep or x["question_type"] in keep]
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="commonmemo-lme-"))
    work.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False).stdout.strip()
    print(f"claude {version}; work dir {work}; {len(items)} questions", file=sys.stderr)

    projects: dict[str, Path] = {}
    if "memory" in conditions:
        for it in items:
            qid = it["question_id"]
            db = work / f"{qid}.db"
            lb.ingest(as_conv(it), db, None)
            if not args.ingest_only:
                projects[qid] = lb.project(work, qid, db, [])
                if args.line_chars:
                    mcp = projects[qid] / ".mcp.json"
                    cfg = json.loads(mcp.read_text())
                    cfg["mcpServers"]["memory"]["env"]["COMMONMEMO_LINE_CHARS"] = str(args.line_chars)
                    mcp.write_text(json.dumps(cfg, indent=2))
    if args.ingest_only:
        print("ingested", len(items), file=sys.stderr)
        return 0

    cache_file = work / "answers.jsonl"
    cached = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            r = json.loads(line)
            cached[f"{r['condition']}|{r['id']}"] = r
    bare = work / "bare"
    bare.mkdir(exist_ok=True)
    windows = {it["question_id"]: lb.rag_windows(as_conv(it)) for it in items} if "rag" in conditions else {}

    def run(task: tuple[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
        condition, it = task
        qid = it["question_id"]
        if f"{condition}|{qid}" in cached:
            return cached[f"{condition}|{qid}"]
        if lb.SPENT["usd"] >= args.budget:
            return None
        mem = condition == "memory"
        r = lb.ask(projects[qid] if mem else bare, question_prompt(it, condition, windows.get(qid)),
                   model=args.model, max_turns=8 if mem else 1, tools=mem)
        ok, jc = judge(it, r["reply"], args.judge_model) if not r["error"] else (None, 0.0)
        row = {"condition": condition, "id": qid, "type": it["question_type"], "gold": str(it["answer"]), "answer": r["reply"],
               "correct": ok, "cost": round(r["cost"], 5), "judge_cost": round(jc, 5), "seconds": r["seconds"],
               "recalls": sum(t.endswith("memory_recall") for t in r["tools"]), "error": r["error"]}
        with lb._lock:
            with cache_file.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"  [{condition} {qid} {it['question_type']}] {'ok' if ok else 'wrong' if ok is False else '??'} ${r['cost']:.3f}",
              file=sys.stderr)
        return row

    tasks = [(c, it) for it in items for c in conditions]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(run, tasks) if r]

    types = sorted({it["question_type"] for it in items})
    by = {(r["condition"], r["id"]): r for r in rows}
    summary: dict[str, Any] = {}
    for c in conditions:
        mine = [r for r in rows if r["condition"] == c]
        summary[c] = {"judged": sum(r["correct"] is not None for r in mine), "correct": sum(bool(r["correct"]) for r in mine),
                      "avg_cost_usd": round(sum(r["cost"] for r in mine) / max(1, len(mine)), 4),
                      "avg_judge_cost_usd": round(sum(r["judge_cost"] for r in mine) / max(1, len(mine)), 4),
                      "recalls_per_question": round(sum(r["recalls"] for r in mine) / max(1, len(mine)), 2),
                      "by_type": {t: {"correct": sum(bool(r["correct"]) for r in mine if r["type"] == t and r["correct"] is not None),
                                      "of": sum(r["correct"] is not None for r in mine if r["type"] == t)} for t in types}}
    tests: dict[str, Any] = {}
    for a, b in (("memory", "none"), ("memory", "rag"), ("rag", "none")):
        if a in conditions and b in conditions:
            ids = [it for it in items if by.get((a, it["question_id"]), {}).get("correct") is not None
                   and by.get((b, it["question_id"]), {}).get("correct") is not None]

            def pairs(sub: list[dict[str, Any]]) -> dict[str, Any]:
                return sign_test([by[(a, i["question_id"])]["correct"] for i in sub], [by[(b, i["question_id"])]["correct"] for i in sub])
            tests[f"{a} vs {b}"] = {"overall": pairs(ids), **{t: pairs([i for i in ids if i["question_type"] == t]) for t in types}}
    out = {"date": time.strftime("%Y-%m-%d"), "models": sorted(lb.MODELS), "seed": args.seed, "per_type": args.per_type,
           "judge_model": args.judge_model, "total_cost_usd": round(lb.SPENT["usd"], 4), "claude_calls": lb.SPENT["calls"],
           "summary": summary, "sign_tests": tests, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for c, s in summary.items():
        print(f"{c}: {s['correct']}/{s['judged']} ${s['avg_cost_usd']} " + " ".join(f"{t}={v['correct']}/{v['of']}" for t, v in s["by_type"].items()))
    print(json.dumps(tests))
    print(f"total ${lb.SPENT['usd']:.2f} over {lb.SPENT['calls']} calls; results in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
