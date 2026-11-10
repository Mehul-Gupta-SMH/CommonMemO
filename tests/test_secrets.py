from commonmemo import Memory, MemoryPolicy
from commonmemo.cli import main
from commonmemo.events import ToolInvoked
from commonmemo.secrets import find, mask

KEY = "sk-ant-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4"
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


def test_shapes_found_and_masked():
    text = f"key {KEY} aws {AWS} password=hunter2hunter2hunter2 -----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"
    assert {"api key", "aws key", "credential", "private key"} <= set(find(text))
    out, n = mask(text)
    assert n == 4 and KEY not in out and AWS not in out and "hunter2" not in out and "abc" not in out
    assert "password=[secret removed: credential]" in out


def test_ordinary_text_is_untouched():
    assert mask("I like sk-8 and tokens in general, the password is short")[1] == 0


def test_masked_before_the_log(tmp_path):
    mem = Memory(f"sqlite://{tmp_path}/m.db", policy=MemoryPolicy(mask_secrets=True))
    with mem.session(user_id="u") as s:
        s.message("user", f"use {KEY} please")
        s.record(ToolInvoked(name="t", arguments={"env": f"TOKEN={AWS}"}, result=f"ok {KEY}"))
    dump = repr(list(mem.store.events()))
    assert KEY not in dump and AWS not in dump and "[secret removed: api key]" in dump
    assert mem.secrets_masked == 3


def test_off_by_default_for_the_library():
    mem = Memory()
    with mem.session(user_id="u") as s:
        s.message("user", KEY)
    assert KEY in repr(list(mem.store.events()))


def test_scan_reports_without_printing(tmp_path, capsys):
    db = f"sqlite://{tmp_path}/m.db"
    mem = Memory(db)
    with mem.session(user_id="u") as s:
        s.message("user", KEY)
    assert main(["scan", "--store", db]) == 0
    out = capsys.readouterr().out
    assert "1 event(s) hold likely secrets" in out and KEY not in out
