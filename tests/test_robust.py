"""Property and fault tests (#167): the projections are a pure function of the Log, forgetting leaves
nothing behind, and two processes' worth of writers on one SQLite file do not lose events."""
import sqlite3
import threading

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from commonmemo import Memory
from commonmemo.agent_tools import run_sync
from commonmemo.events import UserFact

WORDS = st.text(alphabet="abcdefghij klm", min_size=3, max_size=30)
STEP = st.one_of(
    st.tuples(st.just("msg"), st.sampled_from(["ada", "bo"]), WORDS),
    st.tuples(st.just("fact"), st.sampled_from(["ada", "bo"]), st.sampled_from(["city", "pet"]), WORDS),
)


def dump(path):
    con = sqlite3.connect(path)
    out = {}
    for (t,) in con.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' "
                            "and name not like '%fts%' and name not in ('meta')"):
        cols = [c[1] for c in con.execute(f"pragma table_info({t})") if c[1] not in ("rowid", "id")]
        if not cols:
            continue
        out[t] = sorted(map(repr, con.execute(f"select {','.join(cols)} from {t}")))
    con.close()
    return out


def play(mem, steps):
    for i, step in enumerate(steps):
        user = step[1]
        with mem.session(session_id=f"s{i}", user_id=user) as s:
            if step[0] == "msg":
                s.message("user", step[2])
            else:
                s.record(UserFact(key=step[2], value=step[3]))


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(steps=st.lists(STEP, min_size=1, max_size=8))
def test_rebuilt_projections_equal_incremental(steps, tmp_path_factory):
    path = tmp_path_factory.mktemp("p") / "m.db"
    mem = Memory(f"sqlite:///{path}")
    play(mem, steps)
    before = dump(path)
    mem.rebuild_knowledge()
    assert dump(path) == before


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(steps=st.lists(STEP, min_size=1, max_size=6))
def test_forgotten_user_text_is_nowhere(steps, tmp_path_factory):
    path = tmp_path_factory.mktemp("f") / "m.db"
    mem = Memory(f"sqlite:///{path}")
    play(mem, steps)
    secret = "zzqxjv-unique-marker"
    with mem.session(session_id="gone", user_id="eve") as s:
        s.message("user", f"my number is {secret}")
        s.record(UserFact(key="number", value=secret))
    run_sync(mem.forget_user("eve"))
    mem.rebuild_knowledge()
    con = sqlite3.connect(path)
    blob = "\n".join(repr(r) for (t,) in con.execute("select name from sqlite_master where type='table'")
                     for r in con.execute(f"select * from {t}"))
    con.close()
    assert secret not in blob


def test_a_failed_append_leaves_log_and_projections_consistent(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    mem = Memory(f"sqlite:///{path}")
    play(mem, [("msg", "ada", "first words")])
    before = dump(path)
    real = mem.store.append

    def boom(event):
        raise OSError("disk full")

    monkeypatch.setattr(mem.store, "append", boom)
    with pytest.raises(OSError):
        with mem.session(session_id="x", user_id="ada") as s:
            s.message("user", "lost words")
    monkeypatch.setattr(mem.store, "append", real)
    after = dump(path)
    assert after["events"] == before["events"]  # nothing half-written (an empty session row may remain)
    mem.rebuild_knowledge()
    assert dump(path)["events"] == before["events"]


def test_two_writers_on_one_file_lose_nothing(tmp_path):
    path = tmp_path / "m.db"
    a, b = Memory(f"sqlite:///{path}"), Memory(f"sqlite:///{path}")
    errors = []

    def write(mem, tag):
        try:
            for i in range(25):
                with mem.session(session_id=f"{tag}{i}", user_id=tag) as s:
                    s.message("user", f"{tag} message {i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=write, args=(m, t)) for m, t in ((a, "a"), (b, "b"))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:1]
    assert len(list(Memory(f"sqlite:///{path}").store.events(kinds=["message"]))) == 50
