"""World-fact keys: an optional registry, lifetimes, pinning, bulk import (#24).

World facts are a small set of curated operational facts, managed like
configuration ("the refund window is 30 days"), not a document store: prose
that needs retrieval belongs in the app's RAG. The registry says which keys
exist and how they behave. It is optional; without one, world facts work as
before.

    mem = Memory(fact_keys=[
        FactKey("refund_window", "How long customers have to return an order",
                lifetime="90d", pinned=True),
        FactKey("plan_tier", "The customer's plan", allowed=("free", "pro")),
    ])
    mem.load_facts("facts.yaml")          # keys and values, versioned with the app

With a registry:

- **Agents reuse keys.** `memory_note_fact` lists the registered keys. An
  agent's (or extractor's) write to an unregistered key, a near-duplicate of
  a registered one (`refund_period` for `refund_window`), or a value outside
  `allowed` stays a **proposal** with a note saying why, even when
  `world_fact_approval` is off. An operator writing a value outside
  `allowed` gets a `ValueError`.
- **Lifetimes.** A registered key's `lifetime` sets `valid_until` when the
  write doesn't, so facts go stale by default instead of by opt-in.
- **Pinned facts** go in the stable half of the push (it rarely changes, so
  prompt caching keeps working). The rest are ranked by the request and go in
  the per-turn half, or are left to `memory_recall`: the stable half stays
  small however many facts there are.

`load_facts` reads JSON or YAML (YAML needs PyYAML):

    scope: project:shop          # optional; default the project, else global
    keys:
      - key: refund_window
        description: How long customers have to return an order
        lifetime: 90d            # seconds, or 30m / 12h / 90d / 2w
        pinned: true
    facts:
      refund_window: 30 days     # or a list of {key, value, scope, ttl}

Loading is idempotent: a fact whose current value already matches is not
written again, so loading at every start doesn't churn the history.
Imported facts have `source="import"` (an operator's rank).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

if TYPE_CHECKING:
    from .memory import Memory

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}


def parse_duration(value: Union[None, int, float, str]) -> Optional[float]:
    """Seconds from a number or a string like "90d", "12h", "30m", "2w"."""
    if value is None or isinstance(value, (int, float)):
        return None if value is None else float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhdw]?)\s*", str(value))
    if not m:
        raise ValueError(f"commonmemo: can't read {value!r} as a duration (use seconds or 30m, 12h, 90d, 2w)")
    return float(m.group(1)) * _UNITS[m.group(2) or "s"]


@dataclass(frozen=True)
class FactKey:
    """One registered world-fact key."""

    key: str
    description: str = ""
    allowed: Optional[tuple[str, ...]] = None
    """The values it may take; None for any."""
    lifetime: Union[None, float, str] = None
    """How long a value stays true when the write doesn't say (seconds, or "90d")."""
    pinned: bool = False
    """Always pushed, in the stable half; unpinned facts are ranked by the request."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "lifetime", parse_duration(self.lifetime))
        if self.allowed is not None:
            object.__setattr__(self, "allowed", tuple(str(v) for v in self.allowed))

    def line(self) -> str:
        """How the key is listed to the agent."""
        out = self.key + (f": {self.description}" if self.description else "")
        if self.allowed:
            out += f" (one of: {', '.join(self.allowed)})"
        return out


def _words(key: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", key.lower()) if w}


@dataclass
class FactRegistry:
    """The registered keys (`Memory.fact_keys`). Empty means no registry."""

    keys: dict[str, FactKey] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.keys)

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def get(self, key: str) -> Optional[FactKey]:
        return self.keys.get(key)

    def register(self, *keys: FactKey) -> None:
        for k in keys:
            self.keys[k.key] = k

    def pinned(self, key: str) -> bool:
        if key.startswith("env."):  # environment facts (environment.py), registered or not
            return True
        spec = self.keys.get(key)
        return bool(spec and spec.pinned)

    def near(self, key: str) -> Optional[str]:
        """The registered key `key` most likely duplicates, if any: very
        similar spelling, or the same leading word (`refund_period` and
        `refund_window`)."""
        if key in self.keys or not self.keys:
            return None
        best = difflib.get_close_matches(key, list(self.keys), n=1, cutoff=0.75)
        if best:
            return best[0]
        words = _words(key)
        overlap = [(len(words & _words(k)), k) for k in self.keys]
        score, found = max(overlap)
        head = key.lower().split("_")[0]
        return found if score and (score >= 2 or found.lower().startswith(head + "_")) else None

    def check(self, key: str, value: str) -> Optional[str]:
        """Why an agent's write of `key = value` should wait for review, or None."""
        if not self.keys:
            return None
        spec = self.keys.get(key)
        if spec is None:
            near = self.near(key)
            return f"unregistered key; did you mean {near!r}?" if near else "unregistered key"
        if spec.allowed and value not in spec.allowed:
            return f"{value!r} is not an allowed value for {key} ({', '.join(spec.allowed)})"
        return None

    def agent_lines(self, limit: int = 40) -> list[str]:
        return [k.line() for k in sorted(self.keys.values(), key=lambda k: k.key)[:limit]]


@dataclass(frozen=True)
class ImportResult:
    keys: int
    written: int
    unchanged: int


def _read(source: Union[str, Path, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("commonmemo: loading YAML needs PyYAML (pip install pyyaml), or use JSON") from e
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"commonmemo: {path} must hold a mapping with 'keys' and/or 'facts'")
    return data


_KEY_FIELDS = {"key", "description", "allowed", "lifetime", "pinned"}


def _facts(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict):
        return [{"key": k, "value": v} for k, v in data.items()]
    return list(data or [])


def load(memory: "Memory", source: Union[str, Path, dict[str, Any]]) -> ImportResult:
    """Register the keys and write the facts in `source` (see the module docstring)."""
    data = _read(source)
    unknown = set(data) - {"scope", "keys", "facts"}
    if unknown:
        raise ValueError(f"commonmemo: unknown sections {sorted(unknown)} (expected scope, keys, facts)")
    keys = []
    for entry in data.get("keys") or []:
        extra = set(entry) - _KEY_FIELDS
        if extra:
            raise ValueError(f"commonmemo: unknown fields {sorted(extra)} for key {entry.get('key')!r}")
        keys.append(FactKey(**entry))
    memory.fact_keys.register(*keys)
    default_scope = data.get("scope") or memory.project_scope
    written = unchanged = 0
    for fact in _facts(data.get("facts")):
        key, value = fact["key"], str(fact["value"])
        scope = fact.get("scope") or default_scope
        current = {(w.scope, w.key): w.value for w in memory.knowledge.world_facts([scope])}
        if current.get((scope, key)) == value:
            unchanged += 1
            continue
        memory.learn(key, value, scope=scope, ttl=parse_duration(fact.get("ttl")), source="import")
        written += 1
    return ImportResult(keys=len(keys), written=written, unchanged=unchanged)


__all__ = ["FactKey", "FactRegistry", "ImportResult", "load", "parse_duration"]
