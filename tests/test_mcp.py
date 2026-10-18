"""The MCP server, driven over real stdio: raw JSON-RPC, and the official
MCP client when it is installed."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from commonmemo import Memory
from commonmemo.mcp import PROTOCOL_VERSIONS


def _serve(tmp_path, messages, *args):
    """Run the server on `messages`, close stdin, return (responses, process)."""
    db = f"sqlite:///{tmp_path / 'mem.db'}"
    proc = subprocess.run(
        [sys.executable, "-m", "commonmemo.mcp", "--store", db, *args],
        input="".join(json.dumps(m) + "\n" for m in messages),
        capture_output=True, text=True, timeout=60, cwd=tmp_path,
    )
    responses = {r["id"]: r for r in map(json.loads, proc.stdout.splitlines())}
    return responses, proc, db


def _req(i, method, params=None):
    return {"jsonrpc": "2.0", "id": i, "method": method, **({"params": params} if params is not None else {})}


INIT = _req(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "test", "version": "0"}})
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def _call(i, name, **arguments):
    return _req(i, "tools/call", {"name": name, "arguments": arguments})


def test_full_session_over_stdio(tmp_path):
    seed = Memory(f"sqlite:///{tmp_path / 'mem.db'}")
    with seed.session(user_id="ada") as s:
        s.never("suggest phone calls")
    asyncio.run(seed.aclose())

    r, proc, db = _serve(tmp_path, [
        _req(0, "server/discover"), INIT, INITIALIZED, _req(2, "tools/list"),
        _call(3, "memory_remember", key="editor", value="vim"),
        _call(4, "memory_recall", query="editor"),
        _call(5, "memory_remember", key="editor"),           # missing an argument
        _call(6, "no_such_tool"),
        _req(7, "resources/list"), _req(8, "resources/read", {"uri": "memory://context"}),
        _req(9, "ping"),
    ], "--user", "ada", "--agent", "claude-code", "--session", "task-1")
    assert proc.returncode == 0, proc.stderr

    assert r[0]["error"]["code"] == -32601  # tells a 2026 client to use the handshake
    init = r[1]["result"]
    assert init["protocolVersion"] == "2025-06-18"
    assert "suggest phone calls" in init["instructions"]  # stable memory is pushed
    tools = {t["name"]: t for t in r[2]["result"]["tools"]}
    assert set(tools) == {"memory_recall", "memory_remember", "memory_note_fact",
                          "memory_avoid", "memory_save_progress"}
    note = tools["memory_note_fact"]["inputSchema"]
    assert note["required"] == ["key", "value"]
    assert note["properties"]["valid_for_seconds"] == {
        "type": "integer", "description": "How long the fact stays true; 0 for no expiry.", "default": 0}

    assert r[3]["result"]["isError"] is False
    assert "vim" in r[4]["result"]["content"][0]["text"]
    assert r[5]["result"]["isError"] is True
    assert r[6]["error"]["code"] == -32602
    assert r[7]["result"]["resources"][0]["uri"] == "memory://context"
    assert "editor: vim" in r[8]["result"]["contents"][0]["text"]
    assert r[9]["result"] == {}

    mem = Memory(db)
    [fact] = mem.store.events(session_id="task-1", kinds=["user_fact"])
    assert (fact.source, fact.key, fact.value) == ("agent", "editor", "vim")
    record = mem.store.get_session("task-1")
    assert (record.user_id, record.agent, record.is_open) == ("ada", "claude-code", False)  # closed on EOF
    asyncio.run(mem.aclose())


def test_read_only_offers_only_recall(tmp_path):
    r, proc, _ = _serve(tmp_path, [INIT, _req(2, "tools/list"), _call(3, "memory_remember", key="k", value="v")],
                        "--user", "ada", "--read-only")
    assert proc.returncode == 0, proc.stderr
    assert [t["name"] for t in r[2]["result"]["tools"]] == ["memory_recall"]
    assert r[3]["error"]["code"] == -32602
    assert "memory_remember" not in r[1]["result"]["instructions"]


def test_agent_world_facts_are_proposals(tmp_path):
    r, proc, db = _serve(tmp_path, [INIT, _call(2, "memory_note_fact", key="refund_window", value="90 days")],
                         "--user", "mallory")
    assert proc.returncode == 0, proc.stderr
    mem = Memory(db)
    assert mem.world_facts() == []
    assert [p.value for p in mem.pending_world_facts()] == ["90 days"]
    asyncio.run(mem.aclose())


def test_another_users_session_is_refused(tmp_path):
    _, first, _ = _serve(tmp_path, [INIT], "--user", "ada", "--session", "s1")
    assert first.returncode == 0, first.stderr
    r, proc, _ = _serve(tmp_path, [INIT], "--user", "bob", "--session", "s1")
    assert proc.returncode != 0
    assert r == {}
    assert "belongs to another user" in proc.stderr


def test_config_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONMEMO_USER", "ada")
    monkeypatch.setenv("COMMONMEMO_READ_ONLY", "1")
    r, proc, _ = _serve(tmp_path, [INIT, _req(2, "tools/list")])
    assert proc.returncode == 0, proc.stderr
    assert "for user ada" in r[1]["result"]["instructions"]
    assert [t["name"] for t in r[2]["result"]["tools"]] == ["memory_recall"]


def test_unknown_protocol_version_gets_the_newest(tmp_path):
    init = _req(1, "initialize", {"protocolVersion": "1999-01-01", "capabilities": {},
                                  "clientInfo": {"name": "t", "version": "0"}})
    r, _, _ = _serve(tmp_path, [init, "not json"])
    assert r[1]["result"]["protocolVersion"] == PROTOCOL_VERSIONS[-1]


def test_parse_error_keeps_serving(tmp_path):
    db = f"sqlite:///{tmp_path / 'mem.db'}"
    proc = subprocess.run([sys.executable, "-m", "commonmemo.mcp", "--store", db],
                          input="garbage\n" + json.dumps(_req(2, "ping")) + "\n",
                          capture_output=True, text=True, timeout=60, cwd=tmp_path)
    lines = [json.loads(line) for line in proc.stdout.splitlines()]
    assert lines[0]["error"]["code"] == -32700 and lines[1]["result"] == {}


def test_cli_entry_point(tmp_path):
    db = f"sqlite:///{tmp_path / 'mem.db'}"
    proc = subprocess.run([sys.executable, "-m", "commonmemo.cli", "mcp", "--store", db, "--user", "ada"],
                          input=json.dumps(INIT) + "\n", capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["result"]["serverInfo"]["name"] == "commonmemo"


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_official_mcp_client(tmp_path, mode):
    mcp = pytest.importorskip("mcp")
    if not hasattr(mcp, "Client"):
        pytest.skip("needs an mcp SDK with mcp.Client")
    from mcp import StdioServerParameters

    db = f"sqlite:///{tmp_path / 'mem.db'}"
    params = StdioServerParameters(command=sys.executable, cwd=str(tmp_path),
                                   args=["-m", "commonmemo.mcp", "--store", db, "--user", "ada",
                                         "--agent", "sdk", "--session", "s-sdk"])

    async def scenario():
        async with mcp.Client(params, mode=mode) as client:
            tools = await client.list_tools()
            assert "memory_recall" in {t.name for t in tools.tools}
            saved = await client.call_tool("memory_remember", {"key": "editor", "value": "vim"})
            assert not saved.is_error if hasattr(saved, "is_error") else not saved.isError
            found = await client.call_tool("memory_recall", {"query": "editor"})
            assert "vim" in found.content[0].text
            ctx = await client.read_resource("memory://context")
            assert "editor: vim" in ctx.contents[0].text

    asyncio.run(scenario())
    mem = Memory(db)
    [fact] = mem.store.events(session_id="s-sdk", kinds=["user_fact"])
    assert fact.source == "agent"
    asyncio.run(mem.aclose())
