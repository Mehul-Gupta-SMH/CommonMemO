"""How many strings the secret masker rewrites in text that is not a secret (#177).

    python scripts/mask_false_positives.py [--lme longmemeval_s_cleaned.json] [--locomo locomo10.json]

Counts `commonmemo.secrets.mask` rewrites over each corpus (message by message) and prints examples
with the surrounding text, so each can be judged by eye as a placeholder, an identifier or a real secret."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from commonmemo import secrets as S  # noqa: E402


def lme(path: str):
    for it in json.loads(Path(path).read_text(encoding="utf-8")):
        for session in it["haystack_sessions"]:
            for t in session:
                yield t["content"]


def locomo(path: str):
    for c in json.loads(Path(path).read_text(encoding="utf-8")):
        for v in c["conversation"].values():
            if isinstance(v, list):
                for t in v:
                    yield t["text"]


def repo_text():
    root = Path(__file__).resolve().parent.parent
    for pat in ("docs/**/*.md", "README.md", "CHANGELOG.md", "src/**/*.py", "examples/**/*.py"):
        for f in root.glob(pat):
            yield f.read_text(encoding="utf-8")


def report(name: str, texts) -> None:
    n = rewrites = 0
    shapes: dict[str, int] = {}
    examples: list[str] = []
    for t in texts:
        n += 1
        masked, k = S.mask(t)
        rewrites += k
        for s in S.find(t):
            shapes[s] = shapes.get(s, 0) + 1
        for m in re.finditer(r".{0,30}\[secret removed: [a-z ]+\].{0,12}", masked, re.S):
            examples.append(m.group(0).replace("\n", " "))
    print(f"{name}: {n} texts, {rewrites} strings rewritten {shapes}")
    for e in examples[:12]:
        print("   ", e)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lme")
    p.add_argument("--locomo")
    a = p.parse_args()
    if a.lme:
        report("LongMemEval-s messages", lme(a.lme))
    if a.locomo:
        report("LoCoMo messages", locomo(a.locomo))
    report("repo docs and source (tests excluded: they hold secret-shaped fixtures on purpose)", repo_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
