"""Time a crawl of N generated Markdown files: first run, unchanged re-run, 1% edited."""
import sys, tempfile, time
from pathlib import Path
from commonmemo import Memory

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
with tempfile.TemporaryDirectory() as t:
    root = Path(t) / "docs"
    for i in range(n):
        d = root / f"d{i % 50}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"p{i}.md").write_text((f"# Page {i}\n" + f"Paragraph about topic {i} and widget {i * 7}. " * 40 + "\n") * 2)
    mem = Memory(f"sqlite:///{t}/m.db", org="acme")
    for label in ("first crawl", "unchanged re-crawl"):
        r = mem.crawl(root)
        print(f"{label}: {r.seen} docs, {r.passages_added} passages, {r.seconds:.1f}s")
    for i in range(0, n, 100):
        (root / f"d{i % 50}" / f"p{i}.md").write_text(f"edited {i}\n")
    r = mem.crawl(root)
    print(f"1% edited: {r.changed} changed, {r.seconds:.1f}s")
