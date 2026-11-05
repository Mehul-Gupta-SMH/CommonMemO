"""Entity and alias expansion for window retrieval (#146, Part A): do aliases find more of the evidence?

Once per LoCoMo conversation, Haiku lists the 50 to 200 most frequent entities with their aliases (cached in
`--work/aliases-<n>.json`). At query time, every entity named in the question adds its other names to the keyword
query; no model call per query. Reports the share of questions with all the evidence among the top k windows, by
category, against plain keyword ranking, with a paired sign test. Retrieval only; no answer run.

    python scripts/alias_retrieval.py --model haiku --budget 3 --out alias.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo_benchmark as lb
from locomo_enrich import sign_p, specs, top

from commonmemo.ranking import KeywordRanker

KS = (8, 16, 24)
ARMS = ("question", "question+aliases", "alias bonus")
PROMPT = ("Below is a long conversation between {a} and {b}. List the 50 to 200 entities that come up most often "
          "(people, pets, places, organisations, events, hobbies, works) and, for each, every other name, "
          "nickname, spelling, abbreviation or short description the conversation uses for it. Aliases only: no "
          "relations, no facts. Reply with JSON only: a list of {{\"name\": str, \"aliases\": [str, ...]}}.\n\n{text}")


def parse(reply: str) -> list[dict[str, Any]]:
    m = re.search(r"\[.*\]", reply, re.DOTALL)
    try:
        rows = json.loads(m.group(0)) if m else []
    except ValueError:
        return []
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("name"), str)]


def expand(question: str, entities: list[dict[str, Any]]) -> str:
    """The question plus the other names of every entity it mentions."""
    low = question.lower()
    extra: list[str] = []
    for e in entities:
        names = [e["name"], *[a for a in e.get("aliases", []) if isinstance(a, str)]]
        if any(n.strip() and re.search(rf"\b{re.escape(n.lower())}\b", low) for n in names):
            extra += names
    return f"{question} {' '.join(dict.fromkeys(extra))}" if extra else question


def extras(question: str, entities: list[dict[str, Any]]) -> str:
    """Only the other names of the entities the question mentions (no question words)."""
    return expand(question, entities)[len(question):].strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=None)
    p.add_argument("--model", default="haiku")
    p.add_argument("--budget", type=float, default=3.0)
    p.add_argument("--work", default="alias-work")
    p.add_argument("--out", default="alias.json")
    args = p.parse_args()
    data = lb.load_data(args.data)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bare = work / "bare"
    bare.mkdir(exist_ok=True)

    kw = KeywordRanker()
    hits: dict[int, dict[int, dict[str, list[bool]]]] = {c: {k: {m: [] for m in ARMS} for k in KS} for c in (0, *lb.CATEGORIES)}
    expanded = total = 0
    for ci in range(len(data)):
        conv = lb.disguise(data[ci], ci)
        windows = specs(conv)
        a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
        cache = work / f"aliases-{ci}.json"
        if cache.exists():
            entities = json.loads(cache.read_text())
        else:
            if lb.SPENT["usd"] >= args.budget:
                print(f"budget reached before conversation {ci}", file=sys.stderr)
                break
            text = "\n\n".join(w["text"] for w in windows)
            r = lb.ask(bare, PROMPT.format(a=a, b=b, text=text), model=args.model)
            entities = parse(r["reply"]) if not r["error"] else []
            if not entities:
                print(f"conversation {ci}: no entity list ({r['error']})", file=sys.stderr)
                continue
            cache.write_text(json.dumps(entities))
        print(f"conversation {ci}: {len(entities)} entities", file=sys.stderr)
        common = {a.lower(), b.lower()}
        qs = [q["question"].lower() for q in conv["qa"]]
        for e in entities:
            names = [e["name"], *[x for x in e.get("aliases", []) if isinstance(x, str)]]
            if sum(any(n.lower() in q for n in names) for q in qs) > 0.1 * len(qs):
                common.update(n.lower() for n in names)
        rare = [e for e in entities if e["name"].lower() not in common
                and not any(x.lower() in common for x in e.get("aliases", []) if isinstance(x, str))]
        owner = {d: i for i, w in enumerate(windows) for d in w["ids"]}
        bodies = [w["body"] for w in windows]
        for q in conv["qa"]:
            ev = [e for e in q.get("evidence", []) if e in owner]
            if q["category"] not in lb.CATEGORIES or "answer" not in q or not ev:
                continue
            need = {owner[e] for e in ev}
            text = expand(q["question"], entities)
            total += 1
            expanded += text != q["question"]
            bonus = extras(q["question"], rare)
            for arm, query in (("question", q["question"]), ("question+aliases", text), ("alias bonus", q["question"])):
                scores = asyncio.run(kw.score(query, bodies))
                if arm == "alias bonus" and bonus:
                    scores = [x + 0.5 * y for x, y in zip(scores, asyncio.run(kw.score(bonus, bodies)))]
                for k in KS:
                    ok = need <= set(top(scores, k))
                    for cat in (0, q["category"]):
                        hits[cat][k][arm].append(ok)
    print(f"{total} questions, {expanded} expanded; ${lb.SPENT['usd']:.2f} over {lb.SPENT['calls']} calls", file=sys.stderr)

    names = {0: "all", **lb.CATEGORIES}
    out: dict[str, Any] = {"n": total, "expanded": expanded, "cost_usd": round(lb.SPENT["usd"], 4), "results": {}}
    print("| k | category | n | question | +aliases | alias bonus | bonus vs question (+/-, p) |\n|---|---|---|---|---|---|---|")
    for k in KS:
        for cat, name in names.items():
            row = hits[cat][k]
            n = len(row["question"])
            if not n:
                continue
            w = sum(c and not b for b, c in zip(row["question"], row["alias bonus"]))
            l = sum(b and not c for b, c in zip(row["question"], row["alias bonus"]))
            pct = {arm: round(100 * sum(row[arm]) / n, 1) for arm in ARMS}
            out["results"][f"{name}@{k}"] = {"n": n, **pct, "gain": w, "loss": l, "p": round(sign_p(w, l), 4)}
            print(f"| {k} | {name} | {n} | {pct['question']}% | {pct['question+aliases']}% | {pct['alias bonus']}% | +{w}/-{l}, p={sign_p(w, l):.3f} |")
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
