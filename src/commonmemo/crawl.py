"""Crawlers: fill the organisation's memory from documents in bulk (#143, ADR 0006).

    report = mem.crawl("docs/")                  # Memory(org="acme"): writes passages to org:acme
    report = mem.crawl(FolderSource("docs/"), dry_run=True)

A `Source` yields `Document`s; `crawl` chunks them, skips what has not changed (by content hash),
replaces what has, and retracts the passages of documents that are gone. All writes go through the
Log in batches, so a second crawl of an unchanged folder writes nothing. Crawled text is data: it is
stored and shown as reference text, labelled by source, never as instructions. This version reads local
folders and git checkouts only; there is no network access.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Protocol, Sequence, Union

from . import passages
from .events import PassageAdded, PassageRetracted

if TYPE_CHECKING:
    from .memory import Memory

TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".adoc")
"""What a folder crawl reads when no include pattern is given."""

SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache"})
MAX_BYTES = 1_000_000
"""Larger files are skipped (and reported)."""
BATCH = 200
"""Documents per write: one Log transaction and one projection catch-up each."""

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{16,}"),
)
"""A document that matches is skipped whole and reported by name, never stored."""


@dataclass(frozen=True)
class Document:
    uri: str
    """Stable name: becomes the passages' `source` (a path relative to the crawl root, with the prefix)."""
    text: str
    kind: str = "text"
    modified: Optional[float] = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16])


class Source(Protocol):
    """Anything that yields documents. `prefix` names the part of the scope this source owns: documents
    under it that the source no longer yields are retracted."""

    prefix: str

    def documents(self) -> Iterator[Document]: ...

    def skipped(self) -> dict[str, list[str]]: ...


@dataclass
class CrawlReport:
    scope: str = ""
    dry_run: bool = False
    seen: int = 0
    """Documents the source yielded."""
    added: int = 0
    """Documents new to the scope."""
    changed: int = 0
    unchanged: int = 0
    removed: int = 0
    """Documents whose passages were retracted because the source no longer has them."""
    passages_added: int = 0
    passages_retracted: int = 0
    bytes: int = 0
    skipped: dict[str, list[str]] = field(default_factory=dict)
    """Reason -> names: `secret`, `binary`, `large`, `unreadable`."""
    seconds: float = 0.0

    def summary(self) -> str:
        skipped = ", ".join(f"{len(v)} {k}" for k, v in self.skipped.items() if v) or "none"
        verb = "would write" if self.dry_run else "wrote"
        return (f"{self.scope}: {self.seen} documents ({self.bytes:,} bytes): {self.added} new, {self.changed} changed, "
                f"{self.unchanged} unchanged, {self.removed} removed; {verb} {self.passages_added} passages and "
                f"retracted {self.passages_retracted}; skipped: {skipped}; {self.seconds:.2f}s")


class FolderSource:
    """A local folder or git checkout. Reads Markdown, text, reStructuredText and AsciiDoc by default
    (`include` globs replace that), skips `exclude` globs, binaries, files over `max_bytes`, files that
    look like they hold a secret, and what `.gitignore` ignores (through git when the folder is a checkout)."""

    def __init__(self, root: Union[str, Path], *, include: Sequence[str] = (), exclude: Sequence[str] = (),
                 max_bytes: int = MAX_BYTES, prefix: Optional[str] = None) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"{root} is not a folder")
        self.include, self.exclude, self.max_bytes = tuple(include), tuple(exclude), max_bytes
        self.prefix = f"{self.root.name}/" if prefix is None else prefix
        self._skipped: dict[str, list[str]] = {}

    def skipped(self) -> dict[str, list[str]]:
        return self._skipped

    def _skip(self, reason: str, name: str) -> None:
        self._skipped.setdefault(reason, []).append(name)

    def _wanted(self, rel: str) -> bool:
        name = rel.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in self.exclude):
            return False
        if self.include:
            return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in self.include)
        return name.lower().endswith(TEXT_SUFFIXES)

    def _git_files(self) -> Optional[list[str]]:
        """Tracked and untracked-but-not-ignored files, when the root is inside a git checkout."""
        try:
            out = subprocess.run(["git", "-C", str(self.root), "ls-files", "-z", "--cached", "--others",
                                  "--exclude-standard"], capture_output=True, timeout=60, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return sorted(f for f in out.decode("utf-8", "replace").split("\0") if f)

    def _walk(self) -> list[str]:
        ignores = _gitignore(self.root)
        found: list[str] = []
        for path in sorted(self.root.rglob("*")):
            rel = path.relative_to(self.root).as_posix()
            if any(part in SKIP_DIRS for part in path.relative_to(self.root).parts[:-1]):
                continue
            if path.is_file() and not path.is_symlink() and not _ignored(rel, ignores):
                found.append(rel)
        return found

    def documents(self) -> Iterator[Document]:
        files = self._git_files()
        files = self._walk() if files is None else [f for f in files if not set(f.split("/")[:-1]) & SKIP_DIRS]
        for rel in files:
            if not self._wanted(rel):
                continue
            path = self.root / rel
            try:
                if not path.is_file() or path.is_symlink():
                    continue  # a deleted tracked file, a link out of the tree
                size = path.stat().st_size
                if size > self.max_bytes:
                    self._skip("large", rel)
                    continue
                raw = path.read_bytes()
            except OSError:
                self._skip("unreadable", rel)
                continue
            if b"\0" in raw[:4096]:
                self._skip("binary", rel)
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._skip("binary", rel)
                continue
            if looks_secret(text):
                self._skip("secret", rel)
                continue
            if not text.strip():
                continue
            yield Document(uri=self.prefix + rel, text=text, kind=path.suffix.lstrip(".").lower() or "text",
                           modified=path.stat().st_mtime)


def looks_secret(text: str) -> bool:
    return any(p.search(text) for p in SECRET_PATTERNS)


def _gitignore(root: Path) -> list[str]:
    f = root / ".gitignore"
    if not f.is_file():
        return []
    return [ln.strip() for ln in f.read_text("utf-8", "replace").splitlines()
            if ln.strip() and not ln.startswith(("#", "!"))]


def _ignored(rel: str, patterns: Iterable[str]) -> bool:
    """A small subset of .gitignore: names, globs, `dir/`, and anchored paths (no negation)."""
    for pat in patterns:
        anchored = pat.startswith("/")
        pat = pat.strip("/")
        if not pat:
            continue
        if anchored:
            if fnmatch.fnmatch(rel, pat) or rel.startswith(pat + "/"):
                return True
        elif any(fnmatch.fnmatch(part, pat) for part in rel.split("/")) or fnmatch.fnmatch(rel, pat):
            return True
    return False


def crawl(memory: "Memory", source: Union[Source, str, Path], *, scope: Optional[str] = None,
          dry_run: bool = False) -> CrawlReport:
    """Make `scope` (the memory's org by default) hold the documents `source` yields, and nothing else
    under the source's prefix. Idempotent: the same documents again write nothing."""
    started = time.perf_counter()
    scope = scope or memory.org_scope
    if scope is None:
        raise ValueError("no org set: pass Memory(org=...) or scope='org:<name>'")
    if not hasattr(source, "documents"):
        source = FolderSource(source)  # type: ignore[arg-type]
    report = CrawlReport(scope=scope, dry_run=dry_run)

    live: dict[str, list] = {}
    for row in memory.knowledge.passage_sources(scope):
        live.setdefault(row["source"], []).append(row)

    now = time.time()
    batch: list = []
    in_batch = 0
    seen: set[str] = set()

    def flush() -> None:
        nonlocal batch, in_batch
        if batch and not dry_run:
            memory._append_many(batch)
        batch, in_batch = [], 0

    for doc in source.documents():  # type: ignore[union-attr]
        report.seen += 1
        report.bytes += len(doc.text.encode("utf-8"))
        seen.add(doc.uri)
        events = passages.plan(doc.uri, doc.text, live.get(doc.uri, ()), scope=scope, ts=now)
        if not events:
            report.unchanged += 1
            continue
        report.changed += 1 if doc.uri in live else 0
        report.added += 0 if doc.uri in live else 1
        report.passages_added += sum(isinstance(e, PassageAdded) for e in events)
        report.passages_retracted += sum(isinstance(e, PassageRetracted) for e in events)
        batch += events
        in_batch += 1
        if in_batch >= BATCH:
            flush()
    flush()

    gone = [uri for uri in live if uri.startswith(source.prefix) and uri not in seen]  # type: ignore[union-attr]
    for uri in gone:
        events = [PassageRetracted(passage_id=r["id"], reason="gone from the crawled source") for r in live[uri]]
        report.removed += 1
        report.passages_retracted += len(events)
        batch += events
        if len(batch) >= BATCH * 4:
            flush()
    flush()

    report.skipped = {k: list(v) for k, v in source.skipped().items()}  # type: ignore[union-attr]
    report.seconds = time.perf_counter() - started
    return report
