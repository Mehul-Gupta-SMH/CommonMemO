"""Crawlers (#143): folders into the organisation's passages."""

from __future__ import annotations

import subprocess
import sys

import pytest

from commonmemo import Memory
from commonmemo.crawl import FolderSource, crawl


@pytest.fixture
def mem():
    return Memory("sqlite://:memory:", org="acme")


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    (d / "sub").mkdir(parents=True)
    (d / "a.md").write_text("# Handbook\nThe red binder holds the expenses key.\n")
    (d / "sub" / "c.md").write_text("Parking is on level two.\n")
    return d


def live(mem):
    return {r["source"] for r in mem.knowledge.passage_sources("org:acme")}


def test_crawl_adds_then_second_run_writes_nothing(mem, docs):
    first = mem.crawl(docs)
    assert (first.seen, first.added, first.unchanged) == (2, 2, 0)
    assert live(mem) == {"docs/a.md", "docs/sub/c.md"}
    second = mem.crawl(docs)
    assert (second.added, second.changed, second.unchanged, second.passages_added) == (0, 0, 2, 0)


def test_changed_file_replaces_its_passages_and_deleted_file_goes(mem, docs):
    mem.crawl(docs)
    (docs / "a.md").write_text("The blue folder holds the key now.\n")
    (docs / "sub" / "c.md").unlink()
    r = mem.crawl(docs)
    assert (r.changed, r.removed) == (1, 1)
    assert live(mem) == {"docs/a.md"}
    assert mem.knowledge.passage_sources("org:acme")[0]["source"] == "docs/a.md"


def test_secrets_binaries_and_large_files_are_skipped_and_reported(mem, docs):
    (docs / "s.md").write_text("api_key = abcdefghijklmnop1234567\n")
    (docs / "b.txt").write_bytes(b"x\0y")
    (docs / "big.md").write_text("x" * 200)
    r = mem.crawl(FolderSource(docs, max_bytes=100))
    assert r.skipped == {"secret": ["s.md"], "binary": ["b.txt"], "large": ["big.md"]}
    assert "docs/s.md" not in live(mem)


def test_gitignore_is_respected_without_git(mem, docs):
    (docs / ".gitignore").write_text("drafts/\n*.tmp.md\n")
    (docs / "drafts").mkdir()
    (docs / "drafts" / "x.md").write_text("draft text")
    (docs / "n.tmp.md").write_text("temp text")
    mem.crawl(docs)
    assert live(mem) == {"docs/a.md", "docs/sub/c.md"}


def test_gitignore_is_respected_through_git(mem, docs):
    subprocess.run(["git", "init", "-q"], cwd=docs, check=True)
    (docs / ".gitignore").write_text("secret_notes.md\n")
    (docs / "secret_notes.md").write_text("ignored text")
    mem.crawl(docs)
    assert live(mem) == {"docs/a.md", "docs/sub/c.md"}


def test_dry_run_writes_nothing(mem, docs):
    r = mem.crawl(docs, dry_run=True)
    assert (r.added, r.passages_added) == (2, 2) and r.dry_run
    assert live(mem) == set()


def test_prefix_isolates_two_crawls(mem, docs, tmp_path):
    other = tmp_path / "wiki"
    other.mkdir()
    (other / "w.md").write_text("Wiki page.\n")
    mem.crawl(docs)
    mem.crawl(other)
    (other / "w.md").unlink()
    r = mem.crawl(other)
    assert r.removed == 1
    assert live(mem) == {"docs/a.md", "docs/sub/c.md"}


def test_crawled_text_is_found(mem, docs):
    import asyncio

    from commonmemo import passages

    mem.crawl(docs)
    found, _ = asyncio.run(passages.search(mem, "expenses key", scopes=["org:acme"], limit=4, threshold=0.0))
    assert any("red binder" in p.text for p in found)


def test_needs_an_org(docs):
    with pytest.raises(ValueError):
        Memory("sqlite://:memory:").crawl(docs)


def test_cli(tmp_path, docs):
    db = f"sqlite:///{tmp_path / 'm.db'}"
    run = lambda *a: subprocess.run([sys.executable, "-m", "commonmemo.cli", "crawl", str(docs), "--store", db, *a],
                                    capture_output=True, text=True, cwd=tmp_path)
    assert run().returncode != 0  # no --org
    out = run("--org", "acme", "--dry-run")
    assert out.returncode == 0 and "would write" in out.stdout
    assert "wrote 2 passages" in run("--org", "acme").stdout
    assert "0 new" in run("--org", "acme").stdout
