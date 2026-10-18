"""CommonMemo's evaluation suite (#18): does memory change what an agent does?

Each scenario runs twice, with memory and without, against a scripted
agent (`agent.py`) that decides only from the text it is given: the prompt
and the tools' descriptions. That isolates what memory actually delivers to
an agent, deterministically, with no LLM. A live mode against a real model,
and public long-memory benchmarks, come after #3.

    python -m evals                 # print the report
    python -m evals --md docs/evals.md

The suite runs in CI through `tests/test_evals.py`.
"""

from .harness import Result, Scenario, report, run_all

__all__ = ["Result", "Scenario", "report", "run_all"]
