import pytest

from commonmemo import Memory
from commonmemo.cli import main
from commonmemo.events import UserFact
from commonmemo.transfer import export_log, import_log


def seeded(path):
    mem = Memory(f"sqlite:///{path}")
    with mem.session(session_id="s", user_id="ada") as s:
        s.message("user", "I live in Lisbon")
        s.record(UserFact(key="city", value="Lisbon"))
    return mem


def test_round_trip_is_idempotent(tmp_path):
    src = seeded(tmp_path / "a.db")
    out = tmp_path / "x.jsonl"
    n = export_log(src, out)
    dst = Memory(f"sqlite:///{tmp_path}/b.db")
    assert import_log(dst, out) == (n, 0)
    assert import_log(dst, out) == (0, n)
    assert [e.id for e in dst.store.events()] == [e.id for e in src.store.events()]
    assert dst.store.events(kinds=["user_fact"])[0].value == "Lisbon"


def test_tampered_or_truncated_file_writes_nothing(tmp_path):
    out = tmp_path / "x.jsonl"
    export_log(seeded(tmp_path / "a.db"), out)
    lines = out.read_text().splitlines()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(lines[:-2] + [lines[-1]]) + "\n")
    dst = Memory(f"sqlite:///{tmp_path}/b.db")
    with pytest.raises(ValueError):
        import_log(dst, bad)
    assert dst.store.events() == []


def test_cli(tmp_path, capsys):
    seeded(tmp_path / "a.db")
    f = tmp_path / "x.jsonl"
    assert main(["export", str(f), "--store", f"sqlite:///{tmp_path}/a.db"]) == 0
    assert main(["import", str(f), "--store", f"sqlite:///{tmp_path}/b.db"]) == 0
    assert "added" in capsys.readouterr().out


def test_imported_sessions_can_be_forgotten(tmp_path):
    from commonmemo.agent_tools import run_sync

    out = tmp_path / "x.jsonl"
    export_log(seeded(tmp_path / "a.db"), out)
    dst = Memory(f"sqlite:///{tmp_path}/b.db")
    import_log(dst, out)
    assert dst.store.session_ids("ada") == ["s"]
    run_sync(dst.forget_user("ada"))
    assert dst.store.events(user_id="ada") == []
