"""Shared by engines that keep a local index and rank it by external hits."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .base import Episode


def rank_by_hits(
    hits: Iterable[tuple[str, Iterable[str]]],
    local: dict[str, Episode],
    *,
    limit: int,
    label: str,
) -> list[Episode]:
    """Order local episodes by the hits that point at them.

    `hits` is best first: `(text, keys)`, where `keys` name the episodes a
    hit came from (in `local`'s key space). An episode ranks where its first
    hit ranks, and every distinct hit text pointing at it is appended to its
    summary under `label` -- so a reader sees *why* it was retrieved. Hits
    pointing at no known episode are ignored. Empty when nothing matched.
    """
    order: list[str] = []
    texts: dict[str, list[str]] = {}
    for text, keys in hits:
        for key in keys:
            if key not in local:
                continue
            if key not in texts:
                order.append(key)
                texts[key] = []
            if text not in texts[key]:
                texts[key].append(text)
    return [
        replace(
            local[key],
            summary=f"{local[key].summary} {label}: {'; '.join(texts[key])}.",
            score=1.0 / (rank + 1),
        )
        for rank, key in enumerate(order[:limit])
    ]
