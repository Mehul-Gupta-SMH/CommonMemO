"""Likely secrets: one detector for the crawler (skip the file), capture (mask the value before
the event is written) and `commonmemo scan` (report what an earlier Log already holds).

Best effort: token shapes and `KEY=value` forms, not a guarantee. A secret in an unusual shape,
or split across messages, gets through."""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional

PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?:.*?-----END [A-Z ]*PRIVATE KEY-----)?", re.S)),
    ("aws key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("api key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{24,}\b")),
    ("google key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("stripe key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("credential", re.compile(
        r"(?i)(\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"]?)[A-Za-z0-9/+_\-]{16,}")),
)

_ENVELOPE = frozenset({"id", "kind", "ts", "session_id", "user_id", "agent", "run_id", "role", "name"})


def looks_secret(text: str) -> bool:
    return any(p.search(text) for _, p in PATTERNS)


def find(text: str) -> list[str]:
    """Names of the secret shapes in `text` (never the values)."""
    return [name for name, p in PATTERNS if p.search(text)]


def mask(text: str) -> tuple[str, int]:
    """`text` with each likely secret replaced by `[secret removed: <shape>]`, and how many."""
    n = 0
    for name, p in PATTERNS:
        def sub(m: "re.Match[str]", name: str = name) -> str:
            nonlocal n
            n += 1
            head = m.group(1) if m.re.groups else ""
            return f"{head}[secret removed: {name}]"
        text = p.sub(sub, text)
    return text, n


def _walk(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, dict):
        out, total = {}, 0
        for k, v in value.items():
            out[k], c = _walk(v)
            total += c
        return out, total
    if isinstance(value, (list, tuple)):
        items = [_walk(v) for v in value]
        return type(value)(i for i, _ in items), sum(c for _, c in items)
    return value, 0


def mask_event(event: Any) -> tuple[Any, int]:
    """`event` with secrets masked in its content fields (not its envelope), and the count."""
    changes: dict[str, Any] = {}
    total = 0
    for f in dataclasses.fields(event):
        if f.name in _ENVELOPE:
            continue
        new, c = _walk(getattr(event, f.name))
        if c:
            changes[f.name], total = new, total + c
    return (dataclasses.replace(event, **changes) if changes else event), total


def scan_text(text: Optional[str]) -> list[str]:
    return find(text) if text else []
