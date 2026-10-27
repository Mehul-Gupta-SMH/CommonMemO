"""Seeing what memory did: the CLI, the views and the inspector (#25)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from commonmemo import Memory
from commonmemo.agent_tools import build_agent_tools
from commonmemo.inspector import serve
from commonmemo.views import session_view, user_view

ATTACK = "<script>alert(1)</script>"


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    mem = Memory(url)
    mem.learn("refund_window", "30 days")
    with mem.session("s1", user_id="ada", agent="claude-code") as s:
        s.remember("preferred_contact", "email")
        s.never("call me", reason="hates calls")
        s.message("user", "the charger firmware rollback bricked unit 7")
    with mem.session("s2", user_id="ada", agent="codex") as s:
        asyncio.run(s.push("charger firmware rollback", part="all"))  # both halves, as an app delivers them
        s.checkpoint("halfway", {"next": "c.py"})
    with mem.session("s3", user_id="mallory", agent="support") as s:
        s.message("user", ATTACK)
        {t.__name__: t for t in build_agent_tools(s)}["memory_note_fact"]("refund_window", f"{ATTACK} 90 days")
    asyncio.run(mem.aclose())
    return url


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "commonmemo", *args], capture_output=True, text=True, timeout=60)


def test_views_never_reopen_a_closed_session(db):
    mem = Memory(db)
    view = session_view(mem, "s2")
    assert view["closed"] and view["resume"] == ['Last checkpoint \'halfway\': {"next": "c.py"}']
    assert not mem.store.get_session("s2").is_open
    user = user_view(mem, "ada")
    assert [f["key"] for f in user["facts"]] == ["preferred_contact"] and user["sessions"] == ["s1", "s2"]
    assert user["negatives"][0]["statement"] == "call me"
    asyncio.run(mem.aclose())


def test_cli_show_and_trace(db):
    user = cli("show", "user", "ada", "--store", db)
    assert user.returncode == 0 and "preferred_contact: email" in user.stdout and "call me (hates calls)" in user.stdout
    trace = cli("trace", "s2", "--store", db)
    assert "turn push" in trace.stdout and "negative: Never call me." in trace.stdout and "fact: preferred_contact" in trace.stdout
    why = cli("trace", "s2", "--why", "preferred_contact", "--store", db)
    assert why.returncode == 0 and "from user_fact" in why.stdout and "in session s1" in why.stdout
    assert cli("trace", "s2", "--why", "nope", "--store", db).returncode == 1
    assert cli("show", "session", "nope", "--store", db).returncode == 1


@pytest.fixture
def inspector(db):
    mem = Memory(db)
    server = serve(mem, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    asyncio.run(mem.aclose())


def get(url: str) -> tuple[int, dict, str]:
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def test_inspector_pages(inspector):
    status, headers, body = get(inspector + "/")
    assert status == 200 and "default-src 'none'" in headers["Content-Security-Policy"]
    assert "s1" in body and "Waiting for review" in body
    status, _, body = get(inspector + "/session/s2")
    assert status == 200 and "turn push" in body and "Never call me." in body and "halfway" in body
    assert "/why?session=s2&amp;key=preferred_contact" in body
    status, _, body = get(inspector + "/why?session=s2&key=preferred_contact")
    assert status == 200 and "user_fact" in body and "/session/s1" in body
    status, _, body = get(inspector + "/user/ada")
    assert status == 200 and "preferred_contact" in body and "hates calls" in body


def test_inspector_escapes_stored_text(inspector):
    for path in ("/", "/session/s3"):
        _, _, body = get(inspector + path)
        assert ATTACK not in body and "&lt;script&gt;alert(1)&lt;/script&gt;" in body


def test_inspector_is_read_only_and_has_json(inspector):
    request = urllib.request.Request(inspector + "/", method="POST", data=b"")
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(request)
    assert refused.value.code == 405
    assert get(inspector + "/session/nope")[0] == 404
    status, headers, body = get(inspector + "/api/why?session=s2&key=preferred_contact")
    data = json.loads(body)
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert data["item"]["key"] == "preferred_contact" and data["sources"][0]["session_id"] == "s1"
    assert json.loads(get(inspector + "/api/user/ada")[2])["facts"][0]["key"] == "preferred_contact"


def test_the_demo_memory_builds_and_every_event_reads_as_a_sentence(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("demo_memory", Path(__file__).parent.parent / "scripts" / "demo_memory.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    demo.build(tmp_path / "demo.db")
    mem = Memory(f"sqlite:///{tmp_path / 'demo.db'}", project="shop")
    view = session_view(mem, "support-ada-2")
    assert not [t["line"] for t in view["timeline"] if "{" in t["line"]]  # no raw payload dumps
    assert "Use calendar days, not working days." in view["labels"].values()
    assert len(mem.pending_procedures()) == 2 and len(mem.pending_world_facts()) == 1
    asyncio.run(mem.aclose())


def test_the_overview_filters_and_the_session_filters_by_kind(inspector):
    _, _, everything = get(inspector + "/")
    _, _, only = get(inspector + "/?q=mallory")
    assert "s1" in everything and "s3" in only and "s1" not in only.split("Recent sessions")[1].split("World facts")[0]
    assert "Clear" in only and "Filter" in everything
    _, _, tools = get(inspector + "/session/s3?show=messages")
    assert "user:" in tools and "memory pushed" not in tools
    _, _, none = get(inspector + "/session/s3?show=tools")
    assert "No events in this view" in none
    assert get(inspector + "/session/s3?show=bogus")[0] == 200  # an unknown group is ignored


def test_pages_have_landmarks_a_current_page_and_the_review_count(inspector):
    _, _, body = get(inspector + "/")
    assert "<main id=main>" in body and "Skip to content" in body and "aria-current=page" in body
    assert "<title>Overview · CommonMemo</title>" in body and "waiting for review" in body
    _, _, session = get(inspector + "/session/s2")
    assert "Overview</a>" in session and "<th scope=col" in session
    assert "<script" not in session.lower()


def test_empty_states_say_what_to_do(tmp_path):
    mem = Memory(f"sqlite:///{tmp_path / 'empty.db'}")
    server = serve(mem, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _, _, body = get(f"http://127.0.0.1:{server.server_address[1]}/")
        assert "No sessions yet" in body and "mem.learn(key, value)" in body and "Nothing is waiting" in body
    finally:
        server.shutdown()
        server.server_close()
        asyncio.run(mem.aclose())
