"""Engine registry. Each engine's dependencies are imported lazily, only
when that engine is requested -- `import commonmemo` never needs any of them."""

from __future__ import annotations

from typing import Any

from .base import Engine, Recap, Fact
from .local import LocalEngine

_PLANNED: dict[str, str] = {}
"""Engines with a reserved name but no implementation yet (none today)."""


def known_engines() -> list[str]:
    return ["local", "graphiti", "mem0"]


def engine_from_name(name: str, **options: Any) -> Engine:
    """Build an engine by name. A planned-but-unshipped engine raises
    `NotImplementedError` (a real target, not there yet); an unknown name
    raises `ValueError` -- two different problems, two different errors."""
    if name == "local":
        return LocalEngine(**options)
    if name == "graphiti":
        from .graphiti import GraphitiEngine

        return GraphitiEngine(**options)
    if name == "mem0":
        from .mem0 import Mem0Engine

        return Mem0Engine(**options)
    if name in _PLANNED:
        raise NotImplementedError(
            f"commonmemo: engine {name!r} is planned but not shipped yet -- "
            f"{_PLANNED[name]} lands in a later release (see docs/design.md, 'Roadmap'). "
            f"Shipped engines: {known_engines()}"
        )
    raise ValueError(
        f"commonmemo: unknown engine {name!r} (shipped: {known_engines()}, planned: {sorted(_PLANNED)})"
    )


__all__ = ["Engine", "Recap", "Fact", "LocalEngine", "engine_from_name", "known_engines"]
