"""Stores for the tiers CommonMemo owns: the Log and Working memory."""

from __future__ import annotations

from .base import ClaimRecord, SessionRecord, Store
from .sqlite import SQLiteStore


def store_from_url(url: str) -> Store:
    """Build a store from a URL: `sqlite:///path/to/mem.db` or `sqlite://:memory:`.

    Only SQLite ships today; another backend is a new `Store` subclass plus
    a scheme here.
    """
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError(f"commonmemo: store URL {url!r} has no scheme (expected 'sqlite:///...')")
    if scheme == "sqlite":
        path = rest[1:] if rest.startswith("/") else rest
        return SQLiteStore(path or ":memory:")
    raise ValueError(f"commonmemo: unknown store scheme {scheme!r} (known: 'sqlite')")


__all__ = ["Store", "SessionRecord", "ClaimRecord", "SQLiteStore", "store_from_url"]
