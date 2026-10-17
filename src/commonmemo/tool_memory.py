"""Tool memory: what calling each tool has taught us.

Built from `ToolInvoked` events (an `@mem.track`-ed call, or a CommonADK
`tool_call`), per `(agent, tool)`:

- **reliability** -- calls, failures, typical duration;
- **error signatures** -- the same failure seen again, normalized so
  "order 4411 not found" and "order 9 not found" count as one;
- **fixes** -- a failed call followed, in the same session, by a successful
  call of the same tool: what changed in the arguments' *shape*
  ("date: dmy_date -> iso_date") is the fix the next caller should apply.

Tool memory is shared across users, so it must never hold a user's data.
It stores argument **shapes** (`iso_date`, `email`, `int`, `text`...), never
values, and error signatures with emails, numbers, ids and quoted values
replaced by placeholders. `notes()` turns a profile into short advice lines,
meant for the tool's own description (see the CommonADK integration).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

_PATTERNS = [
    ("iso_datetime", re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")),
    ("iso_date", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("dmy_or_mdy_date", re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")),
    ("email", re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")),
    ("url", re.compile(r"^https?://", re.I)),
    ("uuid", re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)),
    ("int_string", re.compile(r"^-?\d+$")),
    ("decimal_string", re.compile(r"^-?\d+\.\d+$")),
]


def shape(value: Any) -> str:
    """The shape of one argument value -- never the value itself."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, tuple)):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return "object"
    text = str(value).strip()
    if not text:
        return "empty_string"
    for name, pattern in _PATTERNS:
        if pattern.match(text):
            return name
    return "text" if len(text) <= 80 else "long_text"


def arguments_shape(arguments: dict[str, Any]) -> dict[str, str]:
    return {k: shape(v) for k, v in sorted(arguments.items())}


_SCRUB = [
    (re.compile(r"[^@\s'\"]+@[^@\s'\"]+\.[^@\s'\"]+"), "<email>"),
    (re.compile(r"https?://\S+", re.I), "<url>"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<uuid>"),
    (re.compile(r"'[^']*'|\"[^\"]*\""), "'<value>'"),
    (re.compile(r"\d+(?:[./:-]\d+)*"), "<n>"),
]


def error_signature(error: str, max_len: int = 160) -> str:
    """`error` with anything user-specific replaced, so equal failures group
    together and no user data leaves the user's scope."""
    text = " ".join(error.split())
    for pattern, placeholder in _SCRUB:
        text = pattern.sub(placeholder, text)
    return text[:max_len]


def shape_change(before: dict[str, str], after: dict[str, str]) -> str:
    """Human-readable difference between two argument shapes."""
    parts = []
    for key in sorted(set(before) | set(after)):
        a, b = before.get(key), after.get(key)
        if a == b:
            continue
        if a is None:
            parts.append(f"add {key} ({b})")
        elif b is None:
            parts.append(f"drop {key}")
        else:
            parts.append(f"{key}: {a} -> {b}")
    return "; ".join(parts)


@dataclass
class ToolError:
    signature: str
    count: int
    last_seen: float
    fix: Optional[str] = None
    """The argument-shape change that turned this failure into a success."""


@dataclass
class ToolProfile:
    """Everything tool memory knows about one tool (for one agent, or for
    all agents when `agent` is None)."""

    tool: str
    agent: Optional[str]
    calls: int = 0
    failures: int = 0
    total_ms: float = 0.0
    timed_calls: int = 0
    errors: list[ToolError] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    @property
    def avg_ms(self) -> Optional[float]:
        return self.total_ms / self.timed_calls if self.timed_calls else None

    def notes(self, *, min_calls: int = 3, max_errors: int = 3) -> list[str]:
        """Advice lines, most useful first. Empty until there is evidence:
        one failed call is an anecdote, not a pattern."""
        out = []
        for err in sorted(self.errors, key=lambda e: (e.fix is None, -e.count))[:max_errors]:
            if err.fix:
                out.append(f"Known failure '{err.signature}' (seen {err.count}x) was fixed by: {err.fix}.")
            elif err.count >= 2:
                out.append(f"Recurring failure (seen {err.count}x): '{err.signature}'.")
        if self.calls >= min_calls and self.failure_rate >= 0.2:
            out.append(f"Fails {self.failure_rate:.0%} of {self.calls} calls; check arguments before calling.")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "agent": self.agent,
            "calls": self.calls,
            "failures": self.failures,
            "failure_rate": round(self.failure_rate, 3),
            "avg_ms": self.avg_ms,
            "errors": [e.__dict__ for e in self.errors],
        }


def dumps_shape(s: dict[str, str]) -> str:
    return json.dumps(s, sort_keys=True)


__all__ = [
    "shape",
    "arguments_shape",
    "error_signature",
    "shape_change",
    "ToolProfile",
    "ToolError",
]
