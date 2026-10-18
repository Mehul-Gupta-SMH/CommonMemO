"""`python -m evals [--md PATH]`: run the suite and print (or write) the report."""

import argparse
import datetime
import platform
import sys

import commonmemo

from . import report, run_all

parser = argparse.ArgumentParser(prog="python -m evals")
parser.add_argument("--md", help="also write the report to this Markdown file")
args = parser.parse_args()

results = run_all()
table = report(results)
print(table)
if args.md:
    header = (
        "# Evaluation results\n\n"
        "Deterministic mode of the evaluation suite ([#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)): "
        "each scenario runs with and without memory against a scripted agent that decides only from the text "
        "it is shown (see `evals/`). Engine: `local`. Regenerate with `python -m evals --md docs/evals.md`.\n\n"
        f"commonmemo {commonmemo.__version__} · Python {platform.python_version()} · "
        f"{datetime.date.today().isoformat()}\n\n"
    )
    with open(args.md, "w") as f:
        f.write(header + table + "\n")
sys.exit(0 if all(r.with_memory.passed for r in results) else 1)
