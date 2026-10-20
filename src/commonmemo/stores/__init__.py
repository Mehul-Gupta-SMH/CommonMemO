"""Stores for the tiers CommonMemo owns: the Log and Working memory."""

from __future__ import annotations

from ..sqldb import POSTGRES_SCHEMES
from .base import ClaimRecord, EventCount, SessionRecord, Store
from .sql import PostgresStore, SQLiteStore, SqlStore


def store_from_url(url: str) -> Store:
    """Build a store from a URL: `sqlite:///path/to/mem.db`, `sqlite://:memory:`,
    or `postgresql://user@host:5432/db` (add `?schema=name` for a schema of
    its own; needs `commonmemo[postgres]`)."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError(f"commonmemo: store URL {url!r} has no scheme (expected 'sqlite:///...' or 'postgresql://...')")
    if scheme == "sqlite":
        path = rest[1:] if rest.startswith("/") else rest
        return SQLiteStore(path or ":memory:")
    if scheme in POSTGRES_SCHEMES:
        return PostgresStore(url)
    raise ValueError(f"commonmemo: unknown store scheme {scheme!r} (known: 'sqlite', 'postgresql')")


__all__ = ["Store", "SessionRecord", "ClaimRecord", "EventCount", "SqlStore", "SQLiteStore", "PostgresStore",
           "store_from_url"]
