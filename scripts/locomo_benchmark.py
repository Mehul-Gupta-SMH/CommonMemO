"""LoCoMo live benchmark: memory vs no memory vs the whole conversation, with Claude Code as the model (#59, part of #18).

Not part of the test suite or CI -- it spends model tokens through the
`claude` CLI. LoCoMo (Maharana et al., 2024) is 10 long conversations
(19 to 35 sessions each) with question-answer pairs. It is licensed
CC BY-NC 4.0, so nothing is vendored: the data is downloaded on first use
and cached under ~/.cache/commonmemo.

For one conversation, the same sampled questions are asked in four
conditions, each question in its own headless `claude -p` session:

| Condition | What the model has |
|---|---|
| `none` | nothing: the floor |
| `rag` | the simple alternative: the transcript cut into windows of 6 turns, the 8 that share the most words with the question (the same keyword ranker memory uses) pasted into the prompt. No memory, no tools |
| `said` | `memory` plus the user's past messages searched (#62): 8 windows pushed with each turn and 8 returned by `memory_recall`, the same windows `rag` pastes |
| `both` | `extracted` plus the same message windows |
| `memory` | CommonMemo as shipped: every turn recorded in a session per LoCoMo session, closed (the local engine's recaps), read through the hooks and the MCP tools, read-only |
| `extracted` | the same, with `Memory(extractor=ModelExtractor(...))`: a model call as each session closes writes its facts into the user profile (#60) |
| `full` | the whole conversation in the prompt: the ceiling |

**Contamination.** LoCoMo is public, and the model may have seen it: in a first
run it said it was answering "from what I recall of these conversations"
when its memory searches found nothing. So by default the data is disguised
before anything is asked: the speakers get other names (and nicknames), and
2022, 2023 and 2024 become 2033, 2034 and 2052, which have the same weekdays.
That defeats lookup by name and date, not recognition of the story, so treat
the numbers as an upper bound on what memory adds; `--no-disguise` turns it off.
In every condition the model is told to guess when it cannot tell, so the
`none` row measures how far guessing alone gets.

Answers are judged correct or wrong by a model call against the gold
answer. Categories 1 to 4 are asked (multi-hop, temporal, open-domain,
single-hop); category 5 (adversarial) is left out, as in the benchmark's own
papers. Results are cached per question, so a run can be stopped and resumed,
and a budget caps the spend.

    python scripts/locomo_benchmark.py --conversation 0 --per-category 8 --budget 10
    python scripts/locomo_benchmark.py --conditions none,full --per-category 2   # a cheap check
    python scripts/locomo_benchmark.py --ingest-only                            # offline: build the stores, no model calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_claude_code as live  # noqa: E402  (reuses its child environment and redaction)

from commonmemo import AlertPolicy, Memory, ModelExtractor  # noqa: E402
from commonmemo.ranking import KeywordRanker  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "commonmemo"
USER = "locomo"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
CONDITIONS = ("none", "rag", "memory", "extracted", "said", "both", "full")
STORE_OF = {"memory": "memory", "extracted": "extracted", "said": "memory", "both": "extracted"}
"""Which ingested store each memory condition reads; `said` and `both` add the user's past messages (#62)."""
SAID_FLAGS = ["--messages", "8", "--recall-messages", "8"]
RAG_WINDOW, RAG_TOP = 6, 8
"""The simple alternative a developer would build: windows of 6 turns, the 8 best against the question."""
CALL_BUDGET = "0.60"
"""Per-call cap passed to the CLI; the whole-conversation prompt is the big one."""

_lock = threading.Lock()
SPENT = {"usd": 0.0, "calls": 0}
MODELS: set[str] = set()


# -- data -------------------------------------------------------------------------


def load_data(path: Optional[str]) -> list[dict[str, Any]]:
    file = Path(path) if path else CACHE / "locomo10.json"
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}", file=sys.stderr)
        with urllib.request.urlopen(DATA_URL, timeout=60) as r:  # noqa: S310 -- a fixed https URL
            file.write_bytes(r.read())
    return json.loads(file.read_text(encoding="utf-8"))


def sessions_of(conv: dict[str, Any]) -> list[tuple[int, str, list[dict[str, Any]]]]:
    """`(number, date, turns)` for each session, in order."""
    c = conv["conversation"]
    numbers = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))
    return [(n, c.get(f"session_{n}_date_time", ""), c[f"session_{n}"]) for n in numbers]


def turn_text(turn: dict[str, Any]) -> str:
    text = turn["text"]
    if turn.get("blip_caption"):
        text += f" [shares a photo: {turn['blip_caption']}]"
    return f"{turn['speaker']}: {text}"


def transcript(conv: dict[str, Any]) -> str:
    out = []
    for n, date, turns in sessions_of(conv):
        out.append(f"## Session {n}, {date}\n" + "\n".join(turn_text(t) for t in turns))
    return "\n\n".join(out)


def sample(conv: dict[str, Any], index: int, per_category: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for cat in CATEGORIES:
        qs = [(i, q) for i, q in enumerate(conv["qa"]) if q["category"] == cat and "answer" in q]
        rng.shuffle(qs)
        out += [{"id": f"{index}:{i}", "category": cat, "question": q["question"], "gold": str(q["answer"])}
                for i, q in sorted(qs[:per_category])]
    return out


NAMES = [("Priya", "Dana"), ("Marcus", "Elena"), ("Tobias", "Ingrid"), ("Rohan", "Leila"), ("Soren", "Mira"),
         ("Anika", "Felix"), ("Hugo", "Nadia"), ("Zara", "Omar"), ("Lena", "Viktor"), ("Idris", "Petra")]
NICKNAMES = ("Mel", "Deb", "Cal", "Caro")
YEARS = {"2022": "2033", "2023": "2034", "2024": "2052"}


def disguise(conv: dict[str, Any], index: int) -> dict[str, Any]:
    """The conversation with its speakers renamed and its years moved to years with the same
    weekdays, in the dialogue, the questions and the gold answers alike."""
    text = json.dumps(conv)
    for old, new in zip((conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]), NAMES[index % len(NAMES)]):
        for alias in (a for a in NICKNAMES if old.startswith(a) and a != old):
            text = re.sub(rf"\b{alias}\b", new[: len(alias)], text)
        text = re.sub(rf"\b{old}(?=\b|s\b)", new, text)
    text = re.sub(r"\b(2022|2023|2024)\b", lambda m: YEARS[m.group(1)], text)
    return json.loads(text)


# -- claude -----------------------------------------------------------------------


def env_with_commonmemo() -> dict[str, str]:
    env = dict(live.CHILD_ENV)
    env["PATH"] = str(SHIM_DIR) + os.pathsep + env.get("PATH", "")
    return env


SHIM_DIR = Path(tempfile.mkdtemp(prefix="commonmemo-shim-"))
"""The hooks and the MCP config call `commonmemo`; this makes that command run this interpreter's copy."""
(SHIM_DIR / "commonmemo").write_text(f'#!/bin/sh\nexec "{sys.executable}" -m commonmemo "$@"\n')
(SHIM_DIR / "commonmemo").chmod(0o755)


def ask(cwd: Path, prompt: str, *, model: Optional[str], max_turns: int = 1, tools: bool = False) -> dict[str, Any]:
    """One headless Claude Code session in `cwd`; built-in tools are off, the memory MCP tools allowed."""
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--max-turns", str(max_turns),
           "--tools", "", "--session-id", str(uuid.uuid4()), "--no-session-persistence",
           "--max-budget-usd", CALL_BUDGET]
    if tools:
        cmd += ["--allowedTools", ",".join(live.ALLOWED)]
    if model:
        cmd += ["--model", model]
    started = time.time()
    proc = subprocess.run(cmd, input=prompt, cwd=cwd, env=env_with_commonmemo(), capture_output=True, text=True,
                          timeout=600, check=False)
    result: dict[str, Any] = {}
    called: list[str] = []
    for line in proc.stdout.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "assistant":
            called += [b["name"] for b in d.get("message", {}).get("content", []) if b.get("type") == "tool_use"]
        elif d.get("type") == "result":
            result = d
    cost = result.get("total_cost_usd") or 0.0
    with _lock:
        SPENT["usd"] += cost
        SPENT["calls"] += 1
        MODELS.update((result.get("modelUsage") or {}).keys())
    return {"reply": live.clip(result.get("result") or "", 20_000), "cost": cost, "turns": result.get("num_turns"),
            "seconds": round(time.time() - started, 1), "tools": called, "error": bool(result.get("is_error")) or not result,
            "stderr": live.clip(proc.stderr, 300)}


def judge(question: str, gold: str, answer: str, model: Optional[str]) -> tuple[Optional[bool], float]:
    prompt = (
        "You grade answers to questions about a long conversation between two people.\n"
        f"Question: {question}\nGold answer: {gold}\nAnswer to grade: {answer}\n\n"
        "The answer is CORRECT if it contains the gold answer's information (a date may be given another way, "
        "e.g. 'the week before 9 June' for 'around 2 June'; extra detail is fine). It is WRONG if it misses, "
        "contradicts or says it doesn't know. Reply with exactly one word: CORRECT or WRONG.")
    with tempfile.TemporaryDirectory(prefix="locomo-judge-") as d:
        r = ask(Path(d), prompt, model=model)
    verdict = r["reply"].strip().upper()
    return (None if r["error"] or verdict not in ("CORRECT", "WRONG") else verdict == "CORRECT"), r["cost"]


# -- ingestion --------------------------------------------------------------------


def extractor(model: Optional[str]) -> ModelExtractor:
    """CommonMemo's own extractor (#60), with Claude Code as the model."""
    def complete(prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="locomo-extract-") as d:
            return ask(Path(d), prompt, model=model)["reply"]

    return ModelExtractor(complete)


def ingest(conv: dict[str, Any], db: Path, extract: Optional[ModelExtractor]) -> dict[str, int]:
    """Record the conversation, one closed memory session per LoCoMo session; with an
    extractor, closing a session also extracts its facts (the product feature). A store
    already in the working directory is reused, so a resumed or extended run asks about the same facts."""
    # A bulk ingestion looks like a write spike to the default alerts; that is not what is measured here.
    built = db.exists()
    mem = Memory(f"sqlite:///{db}", project="locomo", extractor=extract, alerts=AlertPolicy(enabled=False))
    for n, date, turns in [] if built else sessions_of(conv):
        with mem.session(f"locomo-session-{n}", user_id=USER, agent="locomo") as s:
            for i, t in enumerate(turns):
                role = "user" if t["speaker"] == conv["conversation"]["speaker_a"] else "assistant"
                s.message(role, (f"On {date}. " if i == 0 else "") + turn_text(t))
    stats = {"sessions": len(sessions_of(conv)), "facts": len(mem.store.events(kinds=["user_fact"])),
             "events": len(mem.store.events())}
    asyncio.run(mem.aclose())
    return stats


def project(root: Path, name: str, db: Path, flags: list[str]) -> Path:
    """A scratch project wired to CommonMemo as a user would, but read-only: asking a question must not change memory."""
    d = root / f"project-{name}"
    (d / ".claude").mkdir(parents=True, exist_ok=True)
    store = f"sqlite:///{db}"
    out = subprocess.run([str(SHIM_DIR / "commonmemo"), "setup", "claude-code", "--user", USER,
                          "--project", "locomo", "--recall-tool", "--read-only", "--no-environment", "--store", store, *flags],
                         capture_output=True, text=True, check=True)
    settings = json.loads(out.stdout)
    settings["enableAllProjectMcpServers"] = True
    (d / ".claude" / "settings.json").write_text(json.dumps(settings, indent=2))
    (d / ".mcp.json").write_text(json.dumps({"mcpServers": {"memory": {
        "command": "commonmemo", "args": ["mcp", "--user", USER, "--agent", "claude-code", "--project", "locomo",
                                          "--read-only", *flags],
        "env": {"COMMONMEMO_STORE": store},
    }}}, indent=2))
    return d


# -- running ----------------------------------------------------------------------


def rag_windows(conv: dict[str, Any]) -> list[str]:
    out = []
    for n, date, turns in sessions_of(conv):
        for i in range(0, len(turns), RAG_WINDOW):
            out.append(f"[Session {n}, {date}]\n" + "\n".join(turn_text(t) for t in turns[i: i + RAG_WINDOW]))
    return out


def rag_context(windows: list[str], q: str) -> str:
    scores = asyncio.run(KeywordRanker().score(q, windows))
    best = sorted(range(len(windows)), key=lambda i: (-scores[i], i))[:RAG_TOP]
    return "\n\n".join(windows[i] for i in sorted(best))


def question_prompt(conv: dict[str, Any], q: str, condition: str, windows: Optional[list[str]] = None) -> str:
    a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
    ask_line = (f"Question about the conversations between {a} and {b} over several months: {q}\n"
                "Answer in one short sentence (a date, name, number or short phrase). "
                "If you cannot tell, make your best guess.")
    if condition == "full":
        return f"You have no tools. Here are all their conversations.\n\n{transcript(conv)}\n\n{ask_line}"
    if condition == "rag":
        return (f"You have no tools. Here are excerpts from their conversations that may be relevant.\n\n"
                f"{rag_context(windows or [], q)}\n\n{ask_line}")
    if condition in STORE_OF:
        return f"You have memory of their conversations. Use it.\n\n{ask_line}"
    return f"You have no tools and have not seen their conversations.\n\n{ask_line}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--conversation", type=int, default=0, help="which of the 10 conversations")
    p.add_argument("--per-category", type=int, default=8, help="questions per category (1 to 4)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-disguise", action="store_true", help="use the data as published (the model may have seen it)")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--model", default=None, help="answering/extraction model for claude --model (default: the CLI's); "
                                                 "e.g. haiku for the smallest")
    p.add_argument("--judge-model", default="haiku", help="model that grades the answers")
    p.add_argument("--budget", type=float, default=10.0, help="stop starting new calls past this many USD")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--data", default=None, help="a local locomo10.json (default: download and cache)")
    p.add_argument("--work", default=None, help="working dir; reuse it to resume (default: a new temp dir)")
    p.add_argument("--out", default="locomo-results.json")
    p.add_argument("--ingest-only", action="store_true", help="build the memory stores and stop (no model calls)")
    args = p.parse_args()

    conditions = [c for c in args.conditions.split(",") if c]
    bad = set(conditions) - set(CONDITIONS)
    if bad:
        p.error(f"unknown conditions {sorted(bad)}")
    data = load_data(args.data)
    conv = data[args.conversation]
    if not args.no_disguise:
        conv = disguise(conv, args.conversation)
    questions = sample(conv, args.conversation, args.per_category, args.seed)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="commonmemo-locomo-"))
    work.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False).stdout.strip()
    print(f"claude {version}; work dir {work}; conversation {args.conversation} "
          f"({conv['conversation']['speaker_a']} and {conv['conversation']['speaker_b']}, "
          f"{len(sessions_of(conv))} sessions); {len(questions)} questions", file=sys.stderr)

    dbs: dict[str, Path] = {}
    ingest_stats: dict[str, Any] = {}
    wanted = {STORE_OF[c] for c in conditions if c in STORE_OF} | ({"memory", "extracted"} if args.ingest_only else set())
    if "memory" in wanted:
        dbs["memory"] = work / "memory.db"
        ingest_stats["memory"] = ingest(conv, dbs["memory"], None)
    if "extracted" in wanted:
        dbs["extracted"] = work / "extracted.db"
        ingest_stats["extracted"] = ingest(conv, dbs["extracted"], None if args.ingest_only else extractor(args.model))
    print(f"ingested: {json.dumps(ingest_stats)}", file=sys.stderr)
    if args.ingest_only:
        return 0
    extraction_cost = SPENT["usd"]

    cache_file = work / "answers.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            r = json.loads(line)
            cached[f"{r['condition']}|{r['id']}"] = r
    projects = {c: project(work, c, dbs[STORE_OF[c]], SAID_FLAGS if c in ("said", "both") else [])
                for c in conditions if c in STORE_OF}
    windows = rag_windows(conv)
    bare = work / "bare"
    bare.mkdir(exist_ok=True)

    def run(task: tuple[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
        condition, q = task
        key = f"{condition}|{q['id']}"
        if key in cached:
            return cached[key]
        if SPENT["usd"] >= args.budget:
            return None
        memory = condition in STORE_OF
        r = ask(projects[condition] if memory else bare, question_prompt(conv, q["question"], condition, windows),
                model=args.model, max_turns=8 if memory else 1, tools=memory)
        ok, judge_cost = judge(q["question"], q["gold"], r["reply"], args.judge_model) if not r["error"] else (None, 0.0)
        row = {"condition": condition, **q, "answer": r["reply"], "correct": ok, "cost": round(r["cost"], 5),
               "judge_cost": round(judge_cost, 5), "seconds": r["seconds"], "turns": r["turns"],
               "recalls": sum(t.endswith("memory_recall") for t in r["tools"]),
               "tools": r["tools"], "error": r["error"], "stderr": r["stderr"]}
        with _lock:
            with cache_file.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"  [{condition} {q['id']} {CATEGORIES[q['category']]}] {'ok' if ok else 'wrong' if ok is False else '??'} "
              f"${r['cost']:.3f} {live.clip(r['reply'], 70)!r}", file=sys.stderr)
        return row

    tasks = [(c, q) for q in questions for c in conditions]
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(run, tasks) if r]

    summary: dict[str, Any] = {}
    for c in conditions:
        mine = [r for r in rows if r["condition"] == c]
        cell = {"questions": len(mine), "judged": sum(r["correct"] is not None for r in mine),
                "correct": sum(bool(r["correct"]) for r in mine),
                "avg_cost_usd": round(sum(r["cost"] for r in mine) / max(1, len(mine)), 4),
                "avg_seconds": round(sum(r["seconds"] for r in mine) / max(1, len(mine)), 1),
                "recall_calls_per_question": round(sum(r["recalls"] for r in mine) / max(1, len(mine)), 2),
                "by_category": {}}
        for cat, name in CATEGORIES.items():
            sub = [r for r in mine if r["category"] == cat and r["correct"] is not None]
            cell["by_category"][name] = {"correct": sum(bool(r["correct"]) for r in sub), "of": len(sub)}
        summary[c] = cell
    out = {"date": time.strftime("%Y-%m-%d"), "claude_version": version, "models": sorted(MODELS),
           "conversation": args.conversation, "seed": args.seed, "per_category": args.per_category, "disguised": not args.no_disguise,
           "judge_model": args.judge_model, "ingest": ingest_stats, "extraction_cost_usd": round(extraction_cost, 4),
           "total_cost_usd": round(SPENT["usd"], 4), "claude_calls": SPENT["calls"], "summary": summary,
           "rows": rows, "work_dir": str(work)}
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n| Condition | Correct | " + " | ".join(CATEGORIES.values()) + " | $/question | s/question | recalls/question |")
    print("|---|---|" + "---|" * (len(CATEGORIES) + 3))
    for c, cell in summary.items():
        cats = " | ".join(f"{v['correct']}/{v['of']}" for v in cell["by_category"].values())
        print(f"| {c} | {cell['correct']}/{cell['judged']} | {cats} | {cell['avg_cost_usd']} | {cell['avg_seconds']} "
              f"| {cell['recall_calls_per_question']} |")
    print(f"\ntotal ${SPENT['usd']:.2f} over {SPENT['calls']} claude calls "
          f"(extraction ${extraction_cost:.2f}); results in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
