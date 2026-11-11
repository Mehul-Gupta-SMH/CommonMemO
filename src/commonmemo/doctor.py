"""`commonmemo doctor`: a read-only health check of a SQLite Log (#163).

Opens the file read-only (opening a `Memory` can rebuild projections, so it is not used), and reports
each check as ok / warn / fail with what to do. Postgres is not covered yet."""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path
from typing import Optional

from .knowledge import SCHEMA_VERSION


def check(path: str) -> list[tuple[str, str, str]]:
    """(level, name, detail) rows; level is `ok`, `warn` or `fail`."""
    rows: list[tuple[str, str, str]] = []
    add = lambda level, name, detail: rows.append((level, name, detail))  # noqa: E731
    p = Path(path)
    if not p.exists():
        add("fail", "log file", f"{path} does not exist (nothing has been recorded, or the --store path is wrong)")
        return rows
    try:
        con = sqlite3.connect(f"file:{p.resolve()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        add("fail", "log file", f"cannot open read-only: {e}")
        return rows
    try:
        add("ok", "log file", f"{p} ({p.stat().st_size / 1e6:.1f} MB)")
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        add("ok" if integrity == "ok" else "fail", "integrity", integrity if integrity != "ok" else "sqlite integrity_check passed")
        try:
            n, head, last = con.execute("SELECT count(*), max(seq), max(ts) FROM events").fetchone()
        except sqlite3.Error as e:
            add("fail", "events table", str(e))
            return rows
        add("ok", "events", f"{n} in the Log" + (f", last {(time.time() - last) / 3600:.1f} h ago" if last else ""))
        try:
            from .memory import Memory

            raw = Memory.RAW_KINDS
            total, rawb = (con.execute(f"SELECT COALESCE(SUM(LENGTH(body)),0), COALESCE(SUM(CASE WHEN kind IN "
                                       f"({','.join('?' * len(raw))}) THEN LENGTH(body) END),0) FROM events", raw).fetchone())
            add("ok", "size", f"{total / 1e6:.1f} MB of event text, {rawb / 1e6:.1f} MB of it raw (messages, tool calls, "
                              "pushes) and so what --retain-days / --max-log-mb can drop")
        except sqlite3.Error:
            pass
        meta = {}
        try:
            meta = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM knowledge_meta")}
        except sqlite3.Error:
            pass
        schema = meta.get("schema")
        if schema is None:
            add("warn", "projections", "none built yet; they are built on the next open")
        elif schema != SCHEMA_VERSION:
            add("warn", "schema", f"file is v{schema}, this commonmemo is v{SCHEMA_VERSION}: projections rebuild "
                                 "from the Log on the next open (nothing is lost)")
        else:
            add("ok", "schema", f"v{schema}")
            behind = (head or 0) - (meta.get("applied_seq") or 0)
            add("ok" if behind <= 0 else "warn", "projections in step",
                "up to date" if behind <= 0 else f"{behind} event(s) not applied yet; they apply on the next open")
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        if mode != "wal":
            add("warn", "concurrency", f"journal_mode={mode}: two harnesses writing at once wait on each other "
                                       "(SQLite default 5 s busy timeout)")
        else:
            add("ok", "concurrency", "WAL")
    finally:
        con.close()
    if shutil.which("commonmemo") is None:
        add("warn", "command", "`commonmemo` is not on PATH, so hooks that call it will fail")
    else:
        add("ok", "command", "`commonmemo` is on PATH")
    return rows


def main(path: str, out=print) -> int:
    rows = check(path)
    for level, name, detail in rows:
        out(f"{level.upper():4}  {name}: {detail}")
    return 1 if any(level == "fail" for level, _, _ in rows) else 0


__all__ = ["check", "main"]
