"""Render the inspector's real pages from the demo memory as static files (#87).

    python scripts/build_pages.py            # writes pages/*.html

`pages/` is the public sample (GitHub Pages). It used to be a hand-made mockup that could drift from the
product; now it is the product's own page code (`inspector_ui.py`) run over `scripts/demo_memory.py`'s
memory, with links rewritten to files and the parts that need a server (the filter box, the kind
filters, the `why` pages) left out.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from commonmemo import Memory
from commonmemo.inspector_ui import overview_page, review_page, session_page, user_page
from commonmemo.views import overview

sys.path.insert(0, str(Path(__file__).parent))
import demo_memory  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "pages"
NOTE = "<div class=banner><div class=grow><b>Sample data.</b> These are the inspector's real pages over a demo memory; " \
       "<code>commonmemo inspect</code> shows yours.</div></div>"


def link(path: str) -> str:
    path = path.replace("&amp;", "&")
    if path == "/":
        return "index.html"
    if path == "/review":
        return "review.html"
    for kind in ("session", "user"):
        if path.startswith(f"/{kind}/"):
            name = re.sub(r"[^A-Za-z0-9_.-]", "-", path.split("/", 2)[2].split("?")[0])
            return f"{kind}-{name}.html"
    return "index.html"


def static(page: str, name: str) -> str:
    page = re.sub(r"<form class=search.*?</form>", "", page, flags=re.S)
    page = re.sub(r"<div class=filters.*?</div>", "", page, flags=re.S)
    page = re.sub(r"<span title='([^']*)'>[^<]*</span>", r"\1", page)  # a frozen "4 min ago" would be wrong tomorrow
    page = re.sub(r"<a href='/why\?[^']*'>why</a>", "", page)
    page = re.sub(r"href='(/[^']*)'", lambda m: f"href='{link(m.group(1))}'", page)
    page = page.replace("<main id=main>", f"<main id=main>{NOTE}", 1)
    page = page.replace("<a class=skip", f"<a class=skip data-page='{name}'", 1)
    return page


def main() -> None:
    OUT.mkdir(exist_ok=True)
    db = Path(tempfile.mkdtemp()) / "demo.db"
    demo_memory.build(db)
    mem = Memory(f"sqlite:///{db}", project="shop")
    v = overview(mem)
    pages = {"index.html": overview_page(mem, mode="sample data"), "review.html": review_page(mem, None, mode="sample data")}
    for u in v["users"]:
        pages[link(f"/user/{u}")] = user_page(mem, u, "sample data")
    for s in v["sessions"]:
        page = session_page(mem, s["session_id"], mode="sample data")
        if page:
            pages[link(f"/session/{s['session_id']}")] = page
    for old in OUT.glob("*.html"):
        old.unlink()
    for name, page in pages.items():
        (OUT / name).write_text(static(page, name), encoding="utf-8")
        print(f"pages/{name}")


if __name__ == "__main__":
    main()
