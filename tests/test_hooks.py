"""Harness hook adapters, driven with recorded payloads (no live harness)."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from commonmemo import Memory
from commonmemo.hooks import MAX_TEXT, run

FIXTURES = Path(__file__).parent / "fixtures" / "hooks"
CLAUDE = ("session-start", "user-prompt-submit", "post-tool-use", "post-tool-use-failure", "stop",
          "pre-compact", "session-end")
CODEX = ("session-start", "user-prompt-submit", "post-tool-use", "stop", "pre-compact", "session-end")


def payload(harness: str, name: str, **changes) -> dict:
    return {**json.loads((FIXTURES / harness / f"{name}.json").read_text()), **changes}


def hook(db: str, harness: str, data, *args: str):
    """Run one hook in-process: (parsed stdout or None, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    stdin = io.StringIO(data if isinstance(data, str) else json.dumps(data))
    assert run(["--harness", harness, "--store", db, *args], stdin, out, err) == 0  # never blocks
    return (json.loads(out.getvalue()) if out.getvalue() else None), err.getvalue()


def context_of(output) -> str:
    return output["hookSpecificOutput"]["additionalContext"]


def open_memory(db: str) -> Memory:
    return Memory(db)


def close(mem: Memory) -> None:
    asyncio.run(mem.aclose())


@pytest.fixture
def db(tmp_path):
    return f"sqlite:///{tmp_path / 'mem.db'}"


# -- Claude Code ---------------------------------------------------------------


def test_claude_code_session(db):
    seed = open_memory(db)
    with seed.session(user_id="ada") as s:
        s.never("suggest phone calls")
    close(seed)

    outputs = {name: hook(db, "claude-code", payload("claude-code", name), "--user", "ada") for name in CLAUDE}
    assert all(not err for _, err in outputs.values()), outputs
    start = outputs["session-start"][0]
    assert start["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "suggest phone calls" in context_of(start)  # stable memory is pushed at start
    assert outputs["post-tool-use"][0] is None and outputs["stop"][0] is None

    mem = open_memory(db)
    sid = "claude-code-" + payload("claude-code", "session-start")["session_id"]
    events = mem.store.events(session_id=sid)
    kinds = [e.kind for e in events]
    assert kinds.count("message") == 2 and kinds.count("tool_invoked") == 2 and "checkpoint" in kinds
    user_msg, reply = [e for e in events if e.kind == "message"]
    assert (user_msg.role, user_msg.content) == ("user", "Fix the date parsing in search_flights")
    assert reply.role == "assistant" and reply.content.startswith("Fixed:")
    ok, failed = [e for e in events if e.kind == "tool_invoked"]
    assert ok.name == "Bash" and ok.error is None and "3 passed" in ok.result and ok.duration_ms == 812
    assert failed.error.startswith("Exit code 1") and failed.result is None
    assert all(e.agent == "claude-code" and e.user_id == "ada" for e in events)
    assert not mem.store.get_session(sid).is_open  # SessionEnd closed and consolidated it
    assert (mem.tool_profile("Bash").calls, mem.tool_profile("Bash").failures) == (2, 1)  # the calls fed tool memory
    close(mem)


def test_prompt_gets_memory_for_the_turn(db):
    seed = open_memory(db)
    with seed.session(user_id="ada", agent="codex") as s:  # yesterday, in another harness
        s.message("user", "The search_flights date parsing breaks on slashes")
    close(seed)
    out, _ = hook(db, "claude-code", payload("claude-code", "user-prompt-submit"), "--user", "ada")
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "search_flights" in context_of(out) and "codex" in context_of(out)  # its recap, and whose it was


# -- Codex -----------------------------------------------------------------------


def _check(schema: dict, data: dict) -> None:
    """The parts of JSON Schema the Codex hook schemas use."""
    props = schema.get("properties", {})
    missing = set(schema.get("required", [])) - set(data)
    assert not missing, f"{schema['title']}: missing {missing}"
    if schema.get("additionalProperties") is False:
        extra = set(data) - set(props)
        assert not extra, f"{schema['title']}: unexpected {extra}"
    for key, value in data.items():
        spec = props.get(key, {})
        if not isinstance(spec, dict):  # `true`: any value
            continue
        if "const" in spec:
            assert value == spec["const"], key
        if "enum" in spec:
            assert value in spec["enum"], key
        ref = spec.get("$ref") or next((a["$ref"] for a in spec.get("allOf", []) if "$ref" in a), None)
        target = schema.get("definitions", {}).get(ref.rsplit("/", 1)[-1]) if ref else None
        if isinstance(value, dict) and target and target.get("type") == "object":
            _check({**target, "title": f"{schema['title']}.{key}", "definitions": schema.get("definitions")}, value)


def _schema(name: str) -> dict:
    return json.loads((FIXTURES / "codex-schema" / f"{name}.json").read_text())


@pytest.mark.parametrize("name", CODEX)
def test_codex_fixtures_match_codex_schemas(name):
    _check(_schema(f"{name}.command.input"), payload("codex", name))


def test_codex_session_and_output_schema(db):
    outputs = {name: hook(db, "codex", payload("codex", name), "--user", "ada")[0] for name in CODEX}
    for name in ("session-start", "user-prompt-submit"):
        if outputs[name] is not None:
            _check(_schema(f"{name}.command.output"), outputs[name])
    mem = open_memory(db)
    sid = "codex-" + payload("codex", "session-start")["session_id"]
    [tool] = [e for e in mem.store.events(session_id=sid) if e.kind == "tool_invoked"]
    assert tool.name == "shell" and tool.arguments["command"][-1] == "sed -n 1,40p c.py"
    assert all(e.agent == "codex" for e in mem.store.events(session_id=sid))
    assert not mem.store.get_session(sid).is_open
    close(mem)


# -- one session across harnesses -------------------------------------------------


def test_session_continues_from_claude_code_in_codex(db):
    shared = ("--user", "ada", "--session", "task-9")
    hook(db, "claude-code", payload("claude-code", "session-start"), *shared)
    hook(db, "claude-code", payload("claude-code", "user-prompt-submit", prompt="Refactor a.py, b.py and c.py"),
         *shared)
    hook(db, "claude-code", payload("claude-code", "pre-compact"), *shared)
    hook(db, "claude-code", payload("claude-code", "session-end"), *shared)

    out, err = hook(db, "codex", payload("codex", "session-start"), *shared)
    assert not err
    pushed = context_of(out)
    assert "Refactor a.py, b.py and c.py" in pushed and "context compacted" in pushed  # where it stood
    hook(db, "codex", payload("codex", "stop"), *shared)

    mem = open_memory(db)
    agents = [e.agent for e in mem.store.events(session_id="task-9")]
    assert agents[0] == "claude-code" and agents[-1] == "codex"
    close(mem)


def test_another_users_session_is_refused_without_blocking(db):
    hook(db, "claude-code", payload("claude-code", "session-start"), "--user", "ada", "--session", "s1")
    out, err = hook(db, "codex", payload("codex", "session-start"), "--user", "bob", "--session", "s1")
    assert out is None and "belongs to another user" in err


# -- robustness --------------------------------------------------------------------


@pytest.mark.parametrize("data", ["not json", "[]", '{"hook_event_name": "Stop"}'])
def test_bad_input_never_blocks(db, data):
    out, err = hook(db, "claude-code", data)
    assert out is None and err.startswith("commonmemo hook:")


def test_unknown_events_are_ignored(db):
    hook(db, "claude-code", payload("claude-code", "session-end"))
    out, err = hook(db, "claude-code", payload("claude-code", "stop", hook_event_name="Notification"))
    assert out is None and not err
    mem = open_memory(db)
    sid = "claude-code-" + payload("claude-code", "stop")["session_id"]
    assert not mem.store.get_session(sid).is_open  # a late, unrelated event doesn't reopen it
    close(mem)


def test_memory_tool_calls_and_long_output(db):
    sid = payload("claude-code", "post-tool-use")["session_id"]
    hook(db, "claude-code", payload("claude-code", "post-tool-use", tool_name="mcp__memory__memory_recall"))
    hook(db, "claude-code", payload("claude-code", "post-tool-use", tool_name="Read",
                                    tool_input={"file_path": "big.txt"}, tool_response="x" * (MAX_TEXT * 3)))
    mem = open_memory(db)
    [tool] = [e for e in mem.store.events(session_id=f"claude-code-{sid}") if e.kind == "tool_invoked"]
    assert tool.name == "Read" and len(tool.result) < MAX_TEXT + 100
    close(mem)


def test_read_only_pushes_but_records_nothing(db):
    for name in CLAUDE:
        hook(db, "claude-code", payload("claude-code", name), "--read-only")
    mem = open_memory(db)
    sid = "claude-code-" + payload("claude-code", "session-start")["session_id"]
    assert mem.store.events(session_id=sid) == []
    close(mem)


def test_hook_over_real_stdio(tmp_path, db):
    proc = subprocess.run(
        [sys.executable, "-m", "commonmemo", "hook", "--harness", "claude-code", "--store", db],
        input=json.dumps(payload("claude-code", "user-prompt-submit")), capture_output=True, text=True,
        timeout=60, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    mem = open_memory(db)
    assert [e.kind for e in mem.store.events()] == ["message", "context_pushed"]
    close(mem)


# -- setup -------------------------------------------------------------------------


def _setup(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "commonmemo", "setup", *args],
                          capture_output=True, text=True, timeout=60)


def test_setup_claude_code_and_codex():
    claude = json.loads(_setup("claude-code", "--user", "ada lovelace").stdout)["hooks"]
    assert set(claude) == {"SessionStart", "UserPromptSubmit", "PostToolUse", "PostToolUseFailure", "Stop",
                           "PreCompact", "SessionEnd"}
    command = claude["PostToolUse"][0]["hooks"][0]["command"]
    assert command == "commonmemo hook --harness claude-code --user 'ada lovelace'"

    codex = json.loads(_setup("codex").stdout)["hooks"]
    for event in codex:  # every event Codex is configured for exists in Codex
        name = "".join("-" + c.lower() if c.isupper() else c for c in event).lstrip("-")
        assert (FIXTURES / "codex-schema" / f"{name}.command.input.json").exists(), event


def test_setup_opencode_prints_the_plugin():
    proc = _setup("opencode")
    assert "export const CommonMemo" in proc.stdout and ".opencode/plugin" in proc.stderr


# -- OpenCode --------------------------------------------------------------------


DRIVER = """
import { CommonMemo } from "./plugin.mjs";
const hooks = await CommonMemo({ directory: "/work/shop" });
const sessionID = "ses_7f3a";
await hooks["chat.message"]({ sessionID, agent: "build" },
  { message: {}, parts: [{ type: "text", text: "Why does search_flights fail?" }] });
await hooks["tool.execute.after"]({ tool: "bash", sessionID, callID: "call_1", args: { command: "pytest" } },
  { title: "pytest", output: "1 failed", metadata: {} });
await hooks["tool.execute.after"]({ tool: "bash", sessionID, callID: "call_2", args: { command: "deploy" } },
  { title: "deploy", output: "charger firmware rollback started on unit 7", metadata: {} });
const output = { system: ["You are OpenCode."] };
await hooks["experimental.chat.system.transform"]({ sessionID, model: {} }, output);
const compact = { context: [] };
await hooks["experimental.session.compacting"]({ sessionID }, compact);
await hooks.event({ event: { type: "session.deleted", properties: { info: { id: sessionID } } } });
console.log(JSON.stringify({ system: output.system, compact: compact.context }));
"""


def test_opencode_plugin(tmp_path, db):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    seed = open_memory(db)
    with seed.session(user_id="ada") as s:
        s.never("suggest phone calls")
    with seed.session(user_id="ada", agent="opencode") as s:
        s.message("user", "the charger firmware rollback bricked unit 7; flash it over serial instead")
    close(seed)
    shutil.copy(Path(__file__).parents[1] / "src" / "commonmemo" / "opencode_plugin.js", tmp_path / "plugin.mjs")
    (tmp_path / "driver.mjs").write_text(DRIVER)
    command = [sys.executable, "-m", "commonmemo", "hook", "--harness", "opencode", "--store", db, "--user", "ada"]
    proc = subprocess.run([node, "driver.mjs"], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "COMMONMEMO_COMMAND": json.dumps(command)})
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["system"][0] == "You are OpenCode."
    assert any("suggest phone calls" in s for s in result["system"][1:])  # pushed into the system prompt
    assert any("suggest phone calls" in s for s in result["compact"])  # and kept through compaction
    assert any("flash it over serial" in s for s in result["system"])  # refreshed after the deploy result

    mem = open_memory(db)
    events = mem.store.events(session_id="opencode-ses_7f3a")
    assert [e.kind for e in events] == ["message", "context_pushed", "tool_invoked", "tool_invoked",
                                        "context_pushed", "checkpoint"]
    assert events[2].name == "bash" and events[2].result == "1 failed"
    assert events[4].reason == "refresh"
    assert all(e.agent == "opencode" for e in events)
    assert not mem.store.get_session("opencode-ses_7f3a").is_open
    close(mem)


# -- mid-run refresh (#22) ---------------------------------------------------------


def _seed_old_session(db):
    seed = open_memory(db)
    with seed.session(user_id="ada", agent="claude-code") as s:
        s.message("user", "the charger firmware rollback bricked unit 7; flash it over serial instead")
    close(seed)


def test_tool_result_refreshes_memory_mid_run(db):
    _seed_old_session(db)
    args = ("--user", "ada")
    hook(db, "claude-code", payload("claude-code", "user-prompt-submit", prompt="Ship release 4.2"), *args)
    quiet, _ = hook(db, "claude-code", payload("claude-code", "post-tool-use"), *args)
    assert quiet is None  # "3 passed" brings up nothing
    out, err = hook(db, "claude-code", payload(
        "claude-code", "post-tool-use",
        tool_response={"stdout": "deploy: charger firmware rollback started on unit 7", "stderr": ""}), *args)
    assert not err and out["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "flash it over serial" in context_of(out)


def test_codex_refresh_output_matches_codex_schema(db):
    _seed_old_session(db)
    hook(db, "codex", payload("codex", "user-prompt-submit", prompt="Ship release 4.2"), "--user", "ada")
    out, _ = hook(db, "codex", payload("codex", "post-tool-use", tool_response={
        "output": "charger firmware rollback started on unit 7", "exit_code": 0}), "--user", "ada")
    assert out is not None
    _check(_schema("post-tool-use.command.output"), out)


def test_read_only_never_refreshes(db):
    _seed_old_session(db)
    out, _ = hook(db, "claude-code", payload(
        "claude-code", "post-tool-use", tool_response="charger firmware rollback started on unit 7"),
        "--user", "ada", "--read-only")
    assert out is None
