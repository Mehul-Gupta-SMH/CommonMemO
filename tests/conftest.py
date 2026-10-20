"""Postgres mode (#11): `COMMONMEMO_TEST_POSTGRES=postgresql://user@host:port/db pytest`
runs the whole suite on Postgres, in a database of its own that is created
here and dropped afterwards (see `tests/pg/sitecustomize.py`). Tests that
are about SQLite itself are marked `sqlite_only` and skipped in that mode."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

BASE = os.environ.get("COMMONMEMO_TEST_POSTGRES")
_DB = f"commonmemo_test_{uuid.uuid4().hex[:8]}"

if BASE:
    import psycopg

    with psycopg.connect(BASE, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{_DB}"')
    dsn = BASE.rsplit("/", 1)[0] + "/" + _DB
    os.environ["COMMONMEMO_TEST_PG_DSN"] = dsn
    here = str(Path(__file__).parent / "pg")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [here, os.environ.get("PYTHONPATH")]))
    # Apply the mapping in this process too. By file, not `import
    # sitecustomize`: Python has already imported a system one by that name.
    import importlib.util

    spec = importlib.util.spec_from_file_location("commonmemo_pg_mode", Path(here) / "sitecustomize.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    from commonmemo import sqldb

    assert sqldb.connect.__module__ == "commonmemo_pg_mode", "Postgres mode did not take effect"


def pytest_configure(config):
    config.addinivalue_line("markers", "sqlite_only: about SQLite itself; skipped when running on Postgres")


def pytest_collection_modifyitems(config, items):
    if BASE:
        skip = pytest.mark.skip(reason="about SQLite itself (COMMONMEMO_TEST_POSTGRES is set)")
        for item in items:
            if "sqlite_only" in item.keywords:
                item.add_marker(skip)


def pytest_sessionfinish(session, exitstatus):
    if BASE:
        import gc

        gc.collect()  # close what the tests left open before dropping the database
        with psycopg.connect(BASE, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{_DB}" WITH (FORCE)')
