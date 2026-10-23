"""Procedural memory: how a type of task is done ([ADR 0003](../../docs/adr/0003-procedural-memory.md), #26).

Four kinds of entry, keyed by **task type**:

- **template** -- the shape of a good result;
- **rule** -- a constraint the result must follow;
- **procedure** -- the steps that produce it;
- **exemplar** -- an accepted result, kept as a reference.

Written procedures live in files next to the app's code and are loaded at
start; agents can add their own; every entry is versioned, so outcomes can
be compared per version:

    mem.load_procedures("procedures/")              # operator's entries, current
    async with mem.session(user_id="u1", task="weekly_report") as s:
        ctx = await s.push_turn(prompt)             # template + rules + step names
        ...
    mem.procedure_outcomes("weekly_report")         # runs and feedback per version

**How a request finds its procedure**, in order: the app names the task
(`mem.session(task=...)`, or `context(task=...)`); else the request is
matched against each entry's `use_when` line with the memory's ranker, and a
match scoring at least `MemoryPolicy.min_procedure_score` is pushed; else
the agent asks with `memory_get_procedure(task)`, whose description lists
the task names.

**Push:** templates and rules go in the turn half, a procedure as its step
names (the full body on pull), exemplars only on pull.

**Trust** (ADR 0002's rule): entries from files and application code are
current at once; entries an agent writes for everyone, or learned ones, are
proposals until an operator approves them (`confirm_procedure`), and a less
trusted source never replaces a more trusted entry. An agent may write an
entry for its own user (`user:<id>` scope: "my reports skip the risks
section") without review, as it may write that user's facts: it reaches no
one else.

**Privacy:** exemplars are the user's data. They are written in the user's
scope; one is shared beyond its user only when an operator approves it with
the personal data scrubbed (`confirm_procedure(..., scrubbed=True)`; known
profile values are redacted on the way too). `forget_user` removes the
user's entries, exemplars and proposals.

**Files** (`load_procedures`): a directory of Markdown files, one entry
each, with a front matter block, or JSON/YAML files listing entries:

    ---
    task: weekly_report            # default: the file's directory name
    kind: template                 # template | rule | procedure | exemplar
    name: default                  # default: the file name
    use_when: Writing the weekly status report for the team
    steps: gather_metrics, draft, review     # a procedure's step names (optional)
    ---
    ## Highlights
    ...

A procedure's steps default to its numbered lines ("1. Gather metrics: ...");
name them after `@mem.step` functions and the steps a run never finished are
recorded with the run (`ProcedureFollowed.skipped`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from .knowledge import ProcedureRecord

if TYPE_CHECKING:
    from .memory import Memory
    from .ranking import Ranker

KINDS = ("template", "rule", "procedure", "exemplar")
PUSHED = ("template", "rule", "procedure")
"""Kinds pushed for a matched task; exemplars are pull only (they're long)."""

_FIELDS = {"task", "kind", "name", "use_when", "steps", "body", "scope"}
_NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s+(.+?)\s*$")


def check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"commonmemo: procedure kind must be one of {', '.join(KINDS)}, not {kind!r}")
    return kind


def slug(text: str) -> str:
    """`Gather metrics` -> `gather_metrics`: how a step label names a step."""
    return "_".join(re.findall(r"[a-z0-9]+", text.lower())) or text


def parse_steps(body: str) -> tuple[str, ...]:
    """A procedure's step names from its numbered lines: the text before a
    colon ("1. gather_metrics: pull the numbers"), else the whole line."""
    out = []
    for line in _NUMBERED.findall(body):
        label = line.split(":", 1)[0].strip().strip("*`_ ")
        out.append(slug(label[:60]))
    return tuple(out)


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    """`key: value` lines between `---` fences, then the body. Plain lines,
    so no YAML library is needed."""
    if not text.startswith("---"):
        return {}, text
    head, sep, body = text[3:].partition("\n---")
    if not sep:
        return {}, text
    meta = {}
    for line in head.strip().splitlines():
        line = line.split(" #", 1)[0].rstrip()
        if not line.strip():
            continue
        key, colon, value = line.partition(":")
        if not colon:
            raise ValueError(f"commonmemo: can't read front matter line {line!r} (expected key: value)")
        meta[key.strip()] = value.strip()
    return meta, body.lstrip("\n")


def _steps(value: Any) -> Optional[tuple[str, ...]]:
    if value is None or value == "":
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return tuple(slug(str(s).strip()) for s in items if str(s).strip())


@dataclass(frozen=True)
class EntrySpec:
    """One entry read from a file, before it is written."""

    task: str
    kind: str
    name: str
    body: str
    use_when: str = ""
    steps: tuple[str, ...] = ()
    scope: Optional[str] = None


def spec(data: dict[str, Any], *, task: Optional[str] = None, name: Optional[str] = None) -> EntrySpec:
    extra = set(data) - _FIELDS
    if extra:
        raise ValueError(f"commonmemo: unknown procedure fields {sorted(extra)} (expected {sorted(_FIELDS)})")
    task = data.get("task") or task
    name = data.get("name") or name
    if not task or not name or "kind" not in data:
        raise ValueError(f"commonmemo: a procedure entry needs a task, a kind and a name: {data!r}")
    kind = check_kind(str(data["kind"]))
    body = str(data.get("body") or "").strip()
    steps = _steps(data.get("steps"))
    if steps is None:
        steps = parse_steps(body) if kind == "procedure" else ()
    return EntrySpec(str(task), kind, str(name), body, str(data.get("use_when") or "").strip(), steps,
                     data.get("scope"))


def read(source: Union[str, Path, dict[str, Any], Sequence[dict[str, Any]]]) -> list[EntrySpec]:
    """The entries in a directory, a file, a dict (`{"entries": [...]}`) or a list."""
    if isinstance(source, dict):
        return [spec(d) for d in source.get("entries") or []]
    if isinstance(source, (list, tuple)):
        return [spec(d) for d in source]
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in (".md", ".json", ".yaml", ".yml"))
        return [s for f in files for s in _read_file(f, path)]
    return _read_file(path, path.parent)


def _read_file(path: Path, root: Path) -> list[EntrySpec]:
    text = path.read_text(encoding="utf-8")
    # A file in a subdirectory belongs to the task the directory is named after.
    task = path.parent.name if path.parent != root else None
    if path.suffix.lower() == ".md":
        meta, body = _front_matter(text)
        return [spec({**meta, "body": body}, task=task, name=path.stem)]
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("commonmemo: loading YAML needs PyYAML (pip install pyyaml), or use JSON or Markdown") from e
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    entries = data.get("entries", []) if isinstance(data, dict) else data
    return [spec(d, task=(data.get("task") if isinstance(data, dict) else None) or task) for d in entries]


@dataclass(frozen=True)
class ProcedureImport:
    entries: int
    written: int
    unchanged: int


def load(memory: "Memory", source: Any, *, scope: Optional[str] = None) -> ProcedureImport:
    """Write the entries in `source` with an operator's rank, skipping any
    whose current version already matches (so loading at every start
    doesn't churn the history)."""
    specs = read(source)
    written = unchanged = 0
    for s in specs:
        where = s.scope or scope or memory.project_scope
        current = memory.knowledge.procedure_entries([where], s.task)
        same = [c for c in current if (c.kind, c.name, c.scope) == (s.kind, s.name, where)
                and (c.body, c.use_when, c.steps) == (s.body, s.use_when, s.steps)]
        if same:
            unchanged += 1
            continue
        memory.save_procedure(s.task, s.kind, s.name, s.body, use_when=s.use_when, steps=s.steps, scope=where,
                              source="file")
        written += 1
    return ProcedureImport(len(specs), written, unchanged)


# -- reading ---------------------------------------------------------------------


@dataclass
class TaskProcedures:
    """The entries for one task as one session sees them: the most specific
    scope wins per entry (user over project over global)."""

    task: str
    entries: list[ProcedureRecord] = field(default_factory=list)
    how: str = "named"
    """How the task was found: `named` by the app, `matched` by description, `pulled` by the agent."""
    score: Optional[float] = None
    """The match score, for a matched task."""

    def of(self, kind: str) -> list[ProcedureRecord]:
        return [e for e in self.entries if e.kind == kind]

    @property
    def pushed(self) -> list[ProcedureRecord]:
        return [e for e in self.entries if e.kind in PUSHED]

    def __bool__(self) -> bool:
        return bool(self.entries)

    def turn_lines(self, *, max_chars: int = 1500, tool: bool = False) -> list[str]:
        """The pushed part: rules and templates in full (each clipped), a
        procedure as its step names, and, when the agent has
        `memory_get_procedure` (`tool`), what more it can pull."""
        out = []
        for e in self.of("rule"):
            out.append(f"Rule ({_label(e)}): {_clip(e.body, max_chars)}")
        for e in self.of("template"):
            out.append(f"Template ({_label(e)}):\n{_indent(_clip(e.body, max_chars))}")
        for e in self.of("procedure"):
            steps = " -> ".join(e.steps) if e.steps else _clip(e.body.splitlines()[0] if e.body else "", 120)
            out.append(f"Procedure ({_label(e)}): {steps}")
        more = []
        if self.of("procedure"):
            more.append("the full procedure")
        n = len(self.of("exemplar"))
        if n:
            more.append(f"{n} accepted example{'s' if n > 1 else ''}")
        if more and tool:
            out.append(f'For {" and ".join(more)}, call memory_get_procedure("{self.task}").')
        return out

    def full_text(self, max_chars: int = 6000) -> str:
        """Everything, for `memory_get_procedure`."""
        parts = []
        for kind in KINDS:
            for e in self.of(kind):
                head = f"## {kind.capitalize()}: {e.name} ({_label(e)})"
                if e.use_when:
                    head += f"\nUse when: {e.use_when}"
                parts.append(f"{head}\n{_clip(e.body, max_chars)}")
        return f"# How {self.task} is done\n\n" + "\n\n".join(parts) if parts else ""


def _label(e: ProcedureRecord) -> str:
    whose = "your own" if e.scope.startswith("user:") else {"global": "everyone's"}.get(e.scope, "this project's")
    return f"{e.name}, v{e.version}, {whose}"


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else "" for line in text.splitlines())


def task_text(task: str, entries: Sequence[ProcedureRecord]) -> str:
    """What a request is matched against: the task's name and `use_when` lines."""
    lines = sorted({e.use_when for e in entries if e.use_when})
    return f"{task.replace('_', ' ')}: " + "; ".join(lines)


async def match(ranker: "Ranker", query: Optional[str], entries: Sequence[ProcedureRecord],
                threshold: float) -> Optional[tuple[str, float]]:
    """The task the request is about, and its score, if any scores at least
    `threshold`: the share of the task's description (its name and `use_when`
    lines) found in the request, or their cosine similarity with an embedding
    ranker. Only tasks with a `use_when` line can be matched."""
    by_task: dict[str, list[ProcedureRecord]] = {}
    for e in entries:
        by_task.setdefault(e.task, []).append(e)
    tasks = [t for t, es in by_task.items() if any(e.use_when for e in es)]
    if not query or not tasks:
        return None
    # How much of each task's description the request contains, not how much
    # of the request the description contains: a long request ("...We shipped
    # the login page; the payments API is late...") would dilute the latter,
    # and a short one sharing two common words would pass it (#51).
    scores = [(await ranker.score(task_text(t, by_task[t]), [query]))[0] for t in tasks]
    best = max(zip(scores, tasks), key=lambda p: p[0])
    return (best[1], best[0]) if best[0] >= threshold else None


def scrub(text: str, values: Sequence[str]) -> str:
    """`text` with each of `values` (a user's profile values) redacted."""
    values = sorted({v for v in values if len(v.strip()) >= 3}, key=len, reverse=True)
    if not values:
        return text
    pattern = re.compile("|".join(re.escape(v) for v in values), re.IGNORECASE)
    return pattern.sub("[redacted]", text)


__all__ = ["KINDS", "EntrySpec", "ProcedureImport", "TaskProcedures", "load", "match", "parse_steps", "read",
           "scrub", "slug"]
