"""Run scenarios with and without memory, and report."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Optional

from commonmemo import Memory, MemoryContext, Session

from .agent import ScriptedAgent


@dataclass
class Outcome:
    passed: bool
    detail: str
    memory_chars: int = 0
    """Characters memory added to the prompt (~4 per token)."""


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    """What it checks, in one line."""
    run: Callable[[bool], Outcome]
    baseline: bool = True
    """Whether a run without memory is meaningful. Safety checks (forgetting,
    framing, cross-user isolation) have no "without memory" to compare."""


@dataclass
class Result:
    scenario: Scenario
    with_memory: Outcome
    without_memory: Optional[Outcome]
    ms: float

    @property
    def memory_helped(self) -> Optional[bool]:
        """True when memory turned a failure into a success."""
        if self.without_memory is None:
            return None
        return self.with_memory.passed and not self.without_memory.passed


def prompt_for(session: Session, request: str, *, with_memory: bool) -> tuple[str, int]:
    """The prompt an agent would get, and how much of it is memory."""
    if not with_memory:
        return f"# Request\n\n{request}", 0
    ctx: MemoryContext = asyncio.run(session.context(request))
    rendered = ctx.render()
    return (f"{rendered}\n\n# Request\n\n{request}" if rendered else f"# Request\n\n{request}"), len(rendered)


def agent_for(memory: Memory, session: Session, request: str, tools: dict[str, str], *,
              with_memory: bool) -> tuple[ScriptedAgent, int]:
    prompt, chars = prompt_for(session, request, with_memory=with_memory)
    descriptions = dict(tools)
    if with_memory:
        for tool, notes in memory.tool_notes(session.agent).items():
            if tool in descriptions:
                descriptions[tool] += "\n\nNotes from past calls:\n" + "\n".join(f"- {n}" for n in notes)
                chars += sum(len(n) for n in notes)
    return ScriptedAgent(prompt=prompt, tool_descriptions=descriptions), chars


def run_all(scenarios: Optional[list[Scenario]] = None) -> list[Result]:
    from .scenarios import SCENARIOS

    out = []
    for sc in scenarios or SCENARIOS:
        start = time.perf_counter()
        on = sc.run(True)
        ms = (time.perf_counter() - start) * 1000
        off = sc.run(False) if sc.baseline else None
        out.append(Result(sc, on, off, ms))
    return out


def report(results: list[Result]) -> str:
    """A Markdown table, for the terminal and for docs/evals.md."""
    def mark(o: Optional[Outcome]) -> str:
        return "n/a" if o is None else ("pass" if o.passed else "fail")

    lines = [
        "| Scenario | Checks | With memory | Without | Memory added (chars) | Time (ms) |",
        "|---|---|---|---|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.scenario.name} | {r.scenario.question} | {mark(r.with_memory)} | {mark(r.without_memory)} | "
            f"{r.with_memory.memory_chars} | {r.ms:.0f} |"
        )
    passed = sum(r.with_memory.passed for r in results)
    helped = sum(1 for r in results if r.memory_helped)
    compared = sum(1 for r in results if r.without_memory is not None)
    lines += ["", f"{passed} of {len(results)} pass with memory; memory turned a failure into a success in "
                  f"{helped} of the {compared} scenarios that have a without-memory baseline."]
    details = [f"- **{r.scenario.name}:** {r.with_memory.detail}" for r in results]
    return "\n".join(lines + ["", "Details (with memory):", ""] + details)
