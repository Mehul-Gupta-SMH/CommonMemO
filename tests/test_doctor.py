import sqlite3

from commonmemo import Memory
from commonmemo.cli import main
from commonmemo.doctor import check


def levels(path):
    return {name: level for level, name, _ in check(path)}


def test_healthy_log(tmp_path):
    db = tmp_path / "m.db"
    mem = Memory(f"sqlite:///{db}")
    with mem.session(user_id="u") as s:
        s.message("user", "hello there")
    got = levels(str(db))
    assert got["integrity"] == "ok" and got["schema"] == "ok" and got["projections in step"] == "ok"


def test_missing_file_fails_without_creating_it(tmp_path, capsys):
    db = tmp_path / "none.db"
    assert main(["doctor", "--store", f"sqlite:///{db}"]) == 1
    assert not db.exists() and "does not exist" in capsys.readouterr().out


def test_doctor_does_not_write(tmp_path):
    db = tmp_path / "m.db"
    mem = Memory(f"sqlite:///{db}")
    with mem.session(user_id="u") as s:
        s.message("user", "hello there")
    before = db.read_bytes()
    con = sqlite3.connect(db)
    con.execute("UPDATE knowledge_meta SET value = 1 WHERE key = 'schema'")  # an older schema
    con.commit()
    con.close()
    stale = db.read_bytes()
    assert levels(str(db))["schema"] == "warn"
    assert db.read_bytes() == stale and stale != before
