import dataclasses
import time

import pytest

from commonmemo import Memory, MemoryPolicy
from commonmemo.events import Message, UserFact


def fill(mem, n, days_old=0.0):
    for i in range(n):
        with mem.session(session_id=f"s{i}-{days_old}", user_id="ada") as s:
            e = s.message("user", f"message number {i} about gardening and tomatoes " * 4)
            if days_old:  # backdate: the Log is append-only, so write an old copy directly
                mem.store.prune(older_than=0)  # no-op, keeps API honest
                old = dataclasses.replace(e, id=f"old-{i}-{days_old}", ts=time.time() - days_old * 86400)
                mem.store.append(old)


def test_age_drops_raw_events_but_keeps_facts_and_recall(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    mem = Memory(url)
    with mem.session(session_id="f", user_id="ada") as s:
        s.record(UserFact(key="city", value="Lisbon"))
    old_fact = dataclasses.replace(mem.store.events(kinds=["user_fact"])[0], id="oldfact", ts=time.time() - 90 * 86400)
    mem.store.append(old_fact)
    fill(mem, 5, days_old=60)
    msgs = lambda: [e for e in mem.store.events(kinds=["message"])]  # noqa: E731
    assert any(e.id.startswith("old-") for e in msgs())
    mem = Memory(url, policy=MemoryPolicy(retain_days=30))
    removed = mem.apply_retention()
    assert removed >= 5 and not any(e.id.startswith("old-") for e in msgs())
    assert mem.store.events(kinds=["user_fact"])  # kept, however old
    assert mem.store.events(kinds=["retention_applied"])[0].removed == removed
    mem.rebuild_knowledge()  # a rebuild still has the fact
    assert [e.value for e in mem.store.events(kinds=["user_fact"])].count("Lisbon") >= 1


@pytest.mark.sqlite_only  # reads the SQLite windows table directly
def test_size_bound_and_no_dangling_windows(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    mem = Memory(url)
    with mem.session(session_id="f", user_id="ada") as s:
        s.record(UserFact(key="city", value="Lisbon"))
    fill(mem, 60)
    assert mem.store.log_bytes() > 20_000
    mem = Memory(url, policy=MemoryPolicy(max_log_mb=0.02))
    mem.apply_retention()
    assert mem.store.log_bytes() <= 20_000 + 2_000  # the RetentionApplied event itself
    assert mem.store.events(kinds=["user_fact"])
    alive = {e.id for e in mem.store.events(kinds=["message"])}
    import json

    import sqlite3

    con = sqlite3.connect(f"{tmp_path}/m.db")  # read the projection table directly: the Projections interface hides it
    rows = con.execute("SELECT event_ids FROM message_windows").fetchall()
    con.close()
    assert all(json.loads(r[0])[0] in alive for r in rows)


def test_off_by_default_and_runs_at_close(tmp_path):
    mem = Memory(f"sqlite:///{tmp_path}/m.db")
    fill(mem, 3)
    assert mem.apply_retention() == 0
    mem2 = Memory(f"sqlite:///{tmp_path}/m2.db", policy=MemoryPolicy(max_log_mb=0.005))
    fill(mem2, 20)  # each close applies the policy
    assert mem2.store.log_bytes(Memory.RAW_KINDS) < 8_000
    assert isinstance(mem2.store.events(kinds=["message"])[0], Message)
