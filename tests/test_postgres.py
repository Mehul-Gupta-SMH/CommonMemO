"""Postgres-only behaviour (#11): run with COMMONMEMO_TEST_POSTGRES=postgresql://... (the rest
of the suite then runs on Postgres too, see conftest.py)."""

from __future__ import annotations

import multiprocessing
import os
import threading
import uuid

import pytest

from commonmemo import Memory
from commonmemo.events import Message
from commonmemo.sqldb import connect
from commonmemo.stores import PostgresStore, store_from_url

DSN = os.environ.get("COMMONMEMO_TEST_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set COMMONMEMO_TEST_POSTGRES to run against Postgres")


def url(schema: str) -> str:
    return f"{DSN}?schema={schema}"


def fresh() -> str:
    return url("t" + uuid.uuid4().hex[:10])


def _append(target: str, worker: int, n: int) -> None:
    store = PostgresStore(target)
    for i in range(n):
        store.append(Message(role="user", content=f"{worker}-{i}"))
    store.close()


def test_a_postgres_url_gives_a_postgres_store_and_memory():
    target = fresh()
    assert isinstance(store_from_url(target), PostgresStore)
    mem = Memory(target)
    assert mem.knowledge._conn.dialect == "postgres" and mem.engine._conn.dialect == "postgres"


def test_schemas_keep_memories_apart():
    a, b = Memory(fresh()), Memory(fresh())
    a.learn("refund_window", "30 days")
    assert b.world_facts() == [] and a.world_facts()[0].value == "30 days"


def test_concurrent_processes_never_hide_an_event_from_a_reader():
    """Log positions commit in order, so a reader that resumes after the last
    position it saw never misses an event, even with writers in other processes."""
    target = fresh()
    store = PostgresStore(target)  # creates the tables first
    ctx = multiprocessing.get_context("spawn")
    writers = [ctx.Process(target=_append, args=(target, w, 150)) for w in range(4)]
    seen: list[int] = []
    ids: set[str] = set()
    for p in writers:
        p.start()
    while any(p.is_alive() for p in writers) or len(ids) < 600:
        for seq, event in store.events_after(seen[-1] if seen else 0):
            seen.append(seq)
            ids.add(event.id)
        if not any(p.is_alive() for p in writers) and len(ids) < 600 and store.position() == (seen[-1] if seen else 0):
            break  # everything committed was read; anything still missing was skipped
    for p in writers:
        p.join()
        assert p.exitcode == 0
    assert seen == sorted(seen) and len(ids) == 600 == len({e.id for e in store.events()})


def test_concurrent_threads_append_each_event_once():
    target = fresh()
    store = PostgresStore(target)
    event = Message(role="user", content="once")
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(PostgresStore(target).append(event))) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and len(store.events()) == 1


def test_sums_come_back_as_numbers_not_decimals():
    db = connect(fresh())
    db.executescript("CREATE TABLE n (v INTEGER)")
    db.executemany("INSERT INTO n VALUES (?)", [(1,), (2,)])
    total = db.execute("SELECT SUM(v) AS s FROM n").fetchone()["s"]
    assert total == 3 and type(total) is int


def test_an_invalid_schema_name_is_refused():
    with pytest.raises(ValueError):
        connect(url('x"; DROP TABLE events; --'))
