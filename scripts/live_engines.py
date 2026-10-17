"""Live run: the Graphiti and Mem0 engines against real backends and a real LLM.

Not part of the test suite -- it spends model tokens. It runs one scenario
per engine through the public `Memory` API and checks what came back:

1. session 1: the user says where they live and how they like follow-ups;
2. session 2: the user says they moved;
3. a third session asks for context and must see the *new* city in the
   user's facts and retrieve the session where the move was mentioned;
4. `forget_user` must leave nothing behind.

Requirements:

    pip install -e ".[graphiti,mem0]" kuzu
    export OPENAI_API_KEY=...        # both engines' default LLM and embedder

Graphiti uses Neo4j when NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD are set,
otherwise an embedded Kuzu database in a temp directory (no server needed).
Mem0 uses a local, on-disk Qdrant in a temp directory.

    python scripts/live_engines.py [graphiti] [mem0]

Exit code 0 only if every check of every engine that ran passed; an engine
whose prerequisites are missing is reported as SKIPPED, never as passed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Both libraries phone home by default; a test run has no business doing so.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
os.environ.setdefault("MEM0_TELEMETRY", "False")

from commonmemo import Memory  # noqa: E402


def _graphiti_engine(tmp: Path):
    from graphiti_core import Graphiti

    from commonmemo.engines.graphiti import GraphitiEngine

    if os.environ.get("NEO4J_URI"):
        graphiti = Graphiti(os.environ["NEO4J_URI"], os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
        where = "neo4j"
    else:
        from graphiti_core.driver.kuzu_driver import KuzuDriver

        driver = KuzuDriver(db=str(tmp / "graph.kuzu"))
        # graphiti-core 0.31.1: KuzuDriver never sets `_database`, which
        # Graphiti._resolve_request_scope reads whenever a group_id is given,
        # so every add_episode raises AttributeError on Kuzu. Kuzu has one
        # database and GraphDriver.clone() returns the same driver, so any
        # value that differs from a group id is correct here.
        if not hasattr(driver, "_database"):
            driver._database = ""
        graphiti = Graphiti(graph_driver=driver)
        where = "kuzu (embedded)"
    return GraphitiEngine(graphiti=graphiti), where, graphiti


def _mem0_engine(tmp: Path):
    from commonmemo.engines.mem0 import Mem0Engine

    config = {
        "vector_store": {"provider": "qdrant", "config": {"path": str(tmp / "qdrant"), "on_disk": True}},
        "history_db_path": str(tmp / "mem0-history.db"),
    }
    return Mem0Engine(config=config), "qdrant (local)", None


async def scenario(name: str) -> list[tuple[str, bool, str]]:
    tmp = Path(tempfile.mkdtemp(prefix=f"commonmemo-live-{name}-"))
    builder = {"graphiti": _graphiti_engine, "mem0": _mem0_engine}[name]
    engine, where, graphiti = builder(tmp)
    if graphiti is not None:
        await graphiti.build_indices_and_constraints()
    mem = Memory(f"sqlite:///{tmp}/mem.db", engine=engine)
    print(f"\n== {name} on {where}  (data in {tmp})")
    checks: list[tuple[str, bool, str]] = []

    started = time.perf_counter()
    async with mem.session("live-1", user_id="ada", agent="support") as s:
        s.message("user", "Hi, I live in Pune. Please always follow up with me by email, not phone.")
        s.message("assistant", "Noted: email follow-ups. How can I help today?")
        s.message("user", "My ThinkPad laptop will not charge.")
        s.message("assistant", "Try a different USB-C charger and reseat the battery.")
    async with mem.session("live-2", user_id="ada", agent="support") as s:
        s.message("user", "Update: I moved from Pune to Bangalore last month.")
        s.message("assistant", "Thanks, I've updated your city to Bangalore.")
    print(f"   consolidated 2 sessions in {time.perf_counter() - started:.1f}s")

    async with mem.session("live-3", user_id="ada", agent="support") as s:
        ctx = await s.context("which city does the user live in now?")
    facts = [f.value for f in ctx.facts]
    print("   facts:", facts)
    print("   recaps:", [(e.session_id, round(e.score or 0, 2)) for e in ctx.recaps])
    print("   rendered context:\n" + "\n".join("     | " + line for line in ctx.render().splitlines()))

    text = " ".join(facts).lower()
    checks.append(("user facts mention Bangalore", "bangalore" in text, "; ".join(facts)))
    checks.append(("user facts mention email follow-ups", "email" in text, "; ".join(facts)))
    checks.append(("move session retrieved", any(e.session_id == "live-2" for e in ctx.recaps),
                   str([e.session_id for e in ctx.recaps])))
    checks.append(("every extracted fact has provenance",
                   all(f.source_event_ids for f in ctx.facts), ""))

    await mem.forget_user("ada")
    after = await mem.engine.user_profile("ada")
    checks.append(("forget_user leaves nothing", after == [], str([f.value for f in after])))
    await mem.aclose()
    return checks


def _prerequisites(name: str) -> str | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set"
    modules = {"graphiti": ["graphiti_core"] + ([] if os.environ.get("NEO4J_URI") else ["kuzu"]),
               "mem0": ["mem0"]}[name]
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            return f"{module} is not installed"
    return None


async def main(names: list[str]) -> int:
    failed = 0
    for name in names:
        missing = _prerequisites(name)
        if missing:
            print(f"\n== {name}: SKIPPED ({missing})")
            failed += 1
            continue
        try:
            checks = await scenario(name)
        except Exception as exc:  # report, don't hide: a crash is a failure
            print(f"   ERROR {type(exc).__name__}: {exc}")
            failed += 1
            continue
        for label, ok, detail in checks:
            print(f"   {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail and not ok else ""))
            failed += not ok
    print(f"\n{'all checks passed' if not failed else f'{failed} failed or skipped'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or ["graphiti", "mem0"])))
