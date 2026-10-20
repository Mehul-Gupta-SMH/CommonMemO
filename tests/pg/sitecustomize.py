"""Run the test suite on Postgres (#11): loaded at interpreter start, in the
test process and in every subprocess a test starts (the CLI, the MCP
server, hooks), when `tests/conftest.py` puts this directory on PYTHONPATH.

Every SQLite target becomes a Postgres schema in the test database: a file
path always maps to the same schema (so reopening a file from another
process reopens the same memory), and each ":memory:" gets a fresh one.
"""

from __future__ import annotations

import importlib
import itertools
import os
import zlib

DSN = os.environ.get("COMMONMEMO_TEST_PG_DSN")

if DSN:
    from commonmemo import sqldb

    _connect = sqldb.connect
    _fresh = itertools.count()

    def connect(target, *, lock="default"):
        target = str(target)
        if sqldb.is_postgres(target):
            return _connect(target, lock=lock)
        if target == ":memory:":
            schema = f"m{os.getpid()}_{next(_fresh)}"
        else:
            schema = "f" + format(zlib.crc32(os.path.abspath(target).encode()), "08x")
        return _connect(f"{DSN}{'&' if '?' in DSN else '?'}schema={schema}", lock=lock)

    for name in ("commonmemo.sqldb", "commonmemo.stores.sql", "commonmemo.knowledge", "commonmemo.engines.local"):
        importlib.import_module(name).connect = connect
