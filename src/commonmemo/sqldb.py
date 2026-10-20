"""One SQL connection for SQLite or Postgres (ADR 0001, #11).

The Log store, the projections (`knowledge.py`) and the local engine all
write plain SQL in one dialect that both databases accept: `?`
parameters, `INSERT ... ON CONFLICT ... DO NOTHING / DO UPDATE`, `REAL`
and `INTEGER` columns. `connect` returns a `Db` that runs it on either:

    connect("mem.db")                                  # SQLite file (or ":memory:")
    connect("postgresql://u@host/db?schema=shop")      # Postgres; schema optional

On Postgres, `?` becomes `%s`, `REAL` becomes `DOUBLE PRECISION` (Postgres'
`REAL` is single precision, too coarse for timestamps), `INTEGER` becomes
`BIGINT`, and numeric aggregates come back as `int` or `float`, never
`Decimal`.

**Write transactions:** `execute("BEGIN IMMEDIATE")` takes the write lock,
then `commit()` or `rollback()` ends it, as in SQLite. On Postgres it opens a
transaction and takes a transaction-scoped advisory lock named by `lock`, so
writers sharing a lock name run one at a time across processes. Outside such
a transaction, each Postgres statement commits on its own (`commit()` is
then a no-op), so a multi-statement change that must be atomic opens one.

**Setup:** creating schemas, tables and indexes runs inside `setup()`. On
Postgres that holds one database-wide lock, because `CREATE ... IF NOT
EXISTS` is not safe when several processes start at once: two of them can
both find the table missing and one then fails.

`?schema=` puts every table in that Postgres schema (created if missing):
one database can hold several independent memories.
"""

from __future__ import annotations

import re
import sqlite3
import zlib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, ContextManager, Iterable, Iterator, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

POSTGRES_SCHEMES = ("postgresql", "postgres")
SETUP_LOCK = zlib.crc32(b"commonmemo:setup") - 2**31


def is_postgres(target: str) -> bool:
    return target.split("://", 1)[0] in POSTGRES_SCHEMES and "://" in target


class Db:
    """What `sqlite3.Connection` offers that CommonMemo uses, for both databases."""

    dialect: str

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any: ...
    def setup(self) -> ContextManager[None]:
        """Hold while creating or changing tables (see the module docstring)."""
        return nullcontext()

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None: ...
    def executescript(self, ddl: str) -> None: ...
    def columns(self, table: str) -> set[str]: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


class SQLiteDb(Db):
    dialect = "sqlite"

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params: Any = ()) -> Any:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        self.conn.executemany(sql, rows)

    def executescript(self, ddl: str) -> None:
        self.conn.executescript(ddl)

    def columns(self, table: str) -> set[str]:
        return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        self.conn.close()


_PARAM = re.compile(r"\?")
_REAL = re.compile(r"\bREAL\b")
_INTEGER = re.compile(r"\bINTEGER\b")


def _pg_ddl(ddl: str) -> str:
    return _INTEGER.sub("BIGINT", _REAL.sub("DOUBLE PRECISION", ddl))


def _pg_sql(sql: str) -> str:
    # Our SQL never holds a literal '?' or '%', so this is a plain swap.
    return _PARAM.sub("%s", sql)


class _Rows:
    """A finished Postgres result, shaped like a sqlite3 cursor."""

    def __init__(self, rows: list[dict[str, Any]], rowcount: int) -> None:
        self._rows, self.rowcount = rows, rowcount

    def fetchone(self) -> Optional[dict[str, Any]]:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)


def _split(url: str) -> tuple[str, Optional[str]]:
    """The connection URL without our `schema` parameter, and that schema."""
    parts = urlsplit(url)
    query = parse_qsl(parts.query)
    schema = next((v for k, v in query if k == "schema"), None)
    rest = urlencode([(k, v) for k, v in query if k != "schema"])
    return urlunsplit(parts._replace(query=rest)), schema


class PostgresDb(Db):
    dialect = "postgres"

    def __init__(self, url: str, *, lock: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from psycopg.types.numeric import NumericLoader
        except ImportError as e:  # pragma: no cover -- exercised without the extra
            raise ImportError('commonmemo: Postgres needs psycopg: pip install "commonmemo[postgres]"') from e

        class Number(NumericLoader):
            # SUM over integers is NUMERIC in Postgres: give back int or float.
            def load(self, data: Any) -> Any:
                value = super().load(data)
                return int(value) if value == value.to_integral_value() else float(value)

        dsn, self.schema = _split(url)
        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        self.conn.adapters.register_loader("numeric", Number)
        self._lock_key = zlib.crc32(f"commonmemo:{self.schema or ''}:{lock}".encode()) - 2**31
        self._in_tx = False
        if self.schema:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", self.schema):
                raise ValueError(f"commonmemo: invalid Postgres schema name {self.schema!r}")
            with self.setup():
                self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            self.conn.execute(f'SET search_path TO "{self.schema}"')

    @contextmanager
    def setup(self) -> Iterator[None]:
        self.conn.execute("SELECT pg_advisory_lock(%s)", (SETUP_LOCK,))
        try:
            yield
        finally:
            self.conn.execute("SELECT pg_advisory_unlock(%s)", (SETUP_LOCK,))

    def execute(self, sql: str, params: Any = ()) -> Any:
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            self.conn.execute("BEGIN")
            self._in_tx = True
            self.conn.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
            return _Rows([], 0)
        cur = self.conn.execute(_pg_sql(sql), tuple(params))
        rows = cur.fetchall() if cur.description is not None else []
        return _Rows(rows, cur.rowcount)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        rows = [tuple(r) for r in rows]
        if rows:
            with self.conn.cursor() as cur:
                cur.executemany(_pg_sql(sql), rows)

    def executescript(self, ddl: str) -> None:
        for statement in (s.strip() for s in _pg_ddl(ddl).split(";")):
            if statement:
                self.conn.execute(statement)

    def columns(self, table: str) -> set[str]:
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s "
            "AND table_schema = current_schema()", (table,)
        ).fetchall()
        return {r["column_name"] for r in rows}

    def commit(self) -> None:
        if self._in_tx:
            self._in_tx = False
            self.conn.execute("COMMIT")

    def rollback(self) -> None:
        if self._in_tx:
            self._in_tx = False
            self.conn.execute("ROLLBACK")

    def close(self) -> None:
        self.conn.close()


def connect(target: str | Path, *, lock: str = "default") -> Db:
    """A `Db` for a SQLite path or a `postgresql://` URL. `lock` names the
    write lock `BEGIN IMMEDIATE` takes on Postgres."""
    target = str(target)
    return PostgresDb(target, lock=lock) if is_postgres(target) else SQLiteDb(target)


__all__ = ["Db", "POSTGRES_SCHEMES", "PostgresDb", "SQLiteDb", "connect", "is_postgres"]
