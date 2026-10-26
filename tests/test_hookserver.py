"""The resident hook server (#92): same answers as a hook process, without the process."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.skipif(not hasattr(os, "getuid"), reason="Unix sockets"), pytest.mark.sqlite_only]

ARGS = ["--harness", "claude-code", "--user", "ada", "--no-environment"]


@pytest.fixture
def env(tmp_path):
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    e = dict(os.environ, XDG_RUNTIME_DIR=str(run), COMMONMEMO_STORE=f"sqlite:///{tmp_path}/memory.db",
             COMMONMEMO_HOOK_IDLE="8", COMMONMEMO_HOOK_LOG=str(tmp_path / "server.log"))
    yield e
    for pid in _servers(run):
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def _servers(run: Path) -> list[int]:
    out = subprocess.run(["ps", "-wwo", "pid,comm,args", "-e"], capture_output=True, text=True).stdout
    return [int(line.split()[0]) for line in out.splitlines()
            if "python" in line.split()[1] and "commonmemo.hookserver" in line]


def hook(env, event, *extra, sid="t1", **payload):
    p = json.dumps({"hook_event_name": event, "session_id": sid, "cwd": ".", **payload})
    r = subprocess.run([sys.executable, "-m", "commonmemo", "hook", *ARGS, *extra], input=p, text=True,
                       capture_output=True, env=env, timeout=60)
    return r.stdout, r.stderr, r.returncode


def wait_for_socket(env, seconds=20):
    sock = Path(env["XDG_RUNTIME_DIR"]) / "commonmemo" / "hook.sock"
    end = time.time() + seconds
    while time.time() < end and not sock.exists():
        time.sleep(0.1)
    return sock


def test_the_client_imports_no_library():
    code = "import sys, commonmemo.hookclient; print([m for m in sys.modules if m.startswith('commonmemo.') and m != 'commonmemo.hookclient'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert out.strip() == "[]"


def test_resident_answers_like_a_hook_process_and_starts_itself(env):
    before = set(_servers(Path(env["XDG_RUNTIME_DIR"])))  # servers other runs left behind
    hook(env, "UserPromptSubmit", prompt="We deploy with make ship, never with scp.")
    hook(env, "Stop", last_assistant_message="ok")
    ask = dict(prompt="How do we deploy this?", sid="t2")  # another session: what was said before is pushed
    plain = hook(env, "UserPromptSubmit", **ask)
    first = hook(env, "UserPromptSubmit", "--resident", **ask)  # no server yet: answered in-process, one is started
    assert first[:2] == plain[:2] and "make ship" in plain[0]
    sock = wait_for_socket(env)
    assert sock.exists() and oct(sock.stat().st_mode & 0o777) == "0o600"
    time.sleep(0.5)
    again = hook(env, "UserPromptSubmit", "--resident", **ask)  # now through the server
    assert again[:2] == plain[:2]
    assert len(set(_servers(sock.parent.parent)) - before) == 1  # one server, not one per hook


def test_resident_records_events_into_the_same_store(env):
    hook(env, "UserPromptSubmit", "--resident", prompt="The wifi password is kiwi-42.")
    wait_for_socket(env)
    time.sleep(0.5)
    hook(env, "Stop", "--resident", last_assistant_message="Noted.")
    out, _, _ = hook(env, "UserPromptSubmit", sid="t2", prompt="what is the wifi password?")  # a plain process reads it
    assert "kiwi-42" in out


def test_a_failed_server_falls_back_and_the_server_exits_when_idle(env):
    sock = Path(env["XDG_RUNTIME_DIR"]) / "commonmemo" / "hook.sock"
    sock.parent.mkdir(mode=0o700)
    sock.write_text("not a socket")  # a stale file where the socket goes
    out, err, code = hook(env, "UserPromptSubmit", "--resident", prompt="hello")
    assert code == 0 and err == ""
    deadline = time.time() + 20
    while time.time() < deadline and not _servers(sock.parent.parent):
        time.sleep(0.2)
    deadline = time.time() + 20
    while time.time() < deadline and _servers(sock.parent.parent):
        time.sleep(0.5)  # COMMONMEMO_HOOK_IDLE=8
    assert _servers(sock.parent.parent) == []
