"""Environment facts (#57)."""

from __future__ import annotations

import asyncio
import json

import pytest

from commonmemo import FactKey, Memory
from commonmemo.environment import _toml, detect
from commonmemo.hooks import handle

needs_toml = pytest.mark.skipif(_toml is None, reason="needs tomllib or tomli")

PYPROJECT = """
[project]
name = "app"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=1.10,<2",
    "httpx[http2] >= 0.27 ; python_version >= '3.11'",
    "private @ https://user:token@example.com/private.whl",
]

[project.optional-dependencies]
dev = ["pytest"]
"""

UV_LOCK = """
version = 1

[[package]]
name = "pydantic"
version = "1.10.13"

[[package]]
name = "httpx"
version = "0.27.2"

[[package]]
name = "anyio"
version = "4.6.0"
"""


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def no_active_venv(monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    (tmp_path / "uv.lock").write_text(UV_LOCK)
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion_info = 3.12.4.final.0\n")
    site = venv / "lib" / "python3.12" / "site-packages"
    for name, version in (("pydantic", "1.10.9"), ("httpx", "0.27.2"), ("pytest", "8.3.3")):
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return tmp_path


@needs_toml
def test_python_pins_at_the_locked_version_with_the_installed_one_where_it_differs(project):
    facts = detect(project)
    assert facts["env.python"] == "3.12.4 (requires >=3.11)"
    # direct dependencies only (not anyio), sorted; the URL requirement is skipped
    assert facts["env.python_packages"] == "httpx 0.27.2, pydantic 1.10.13 (installed 1.10.9), pytest 8.3.3"
    assert "token" not in json.dumps(facts)
    assert facts["env.os"]


def test_requirements_files_and_python_version_file_without_a_venv(tmp_path):
    (tmp_path / ".python-version").write_text("3.11.9\n")
    (tmp_path / "requirements.txt").write_text("# pinned\n-r base.txt\nDjango==4.2.16  # LTS\n-e .\n")
    (tmp_path / "requirements-dev.txt").write_text("ruff\ngit+https://github.com/x/y.git\n")
    facts = detect(tmp_path)
    assert facts["env.python"] == "3.11.9"
    assert facts["env.python_packages"].startswith("Django 4.2.16")  # an exact pin reads as a lock
    assert "ruff" in facts["env.python_packages"] and "git" not in facts["env.python_packages"]
    assert "env.node" not in facts


@needs_toml
def test_without_this_interpreter_only_the_projects_files_count(tmp_path):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    facts = detect(tmp_path, this_interpreter=False)
    assert facts["env.python"] == "unknown (requires >=3.11)"
    assert facts["env.python_packages"] == "httpx >=0.27, pydantic >=1.10,<2, pytest"


def test_node_pins(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"react": "^18.2.0"},
                                                       "devDependencies": {"vite": "^5.0.0", "x": "file:../x"}}))
    (tmp_path / "package-lock.json").write_text(json.dumps({"packages": {"node_modules/react": {"version": "18.3.1"}}}))
    (tmp_path / "node_modules" / "vite").mkdir(parents=True)
    (tmp_path / "node_modules" / "vite" / "package.json").write_text(json.dumps({"version": "5.4.10"}))
    assert detect(tmp_path)["env.node_packages"] == "react 18.3.1, vite 5.4.10 (requires ^5.0.0)"


@needs_toml
def test_recording_is_trusted_pinned_idempotent_and_follows_changes(project):
    mem = Memory(project="app", fact_keys=[FactKey("refund_window", "How long returns take")])
    first = mem.detect_environment(project)
    assert sorted(first.written) == ["env.os", "env.python", "env.python_packages"]
    assert mem.detect_environment(project).written == []  # unchanged: nothing written again
    with mem.session(user_id="ada") as s:
        stable = run(s.context("anything")).stable_text()
    # pinned in the stable half, even though the registry doesn't list env.* keys
    assert "env.python_packages: httpx 0.27.2, pydantic 1.10.13 (installed 1.10.9)" in stable
    assert "tool result" in stable
    # an agent's guess doesn't replace it
    mem.learn("env.python", "3.9", source="agent")
    assert {w.key: w.value for w in mem.world_facts()}["env.python"].startswith("3.12.4")
    # an upgrade supersedes, with history; a removed pin set is retracted
    (project / "uv.lock").write_text(UV_LOCK.replace("1.10.13", "1.10.15"))
    assert mem.detect_environment(project).written == ["env.python_packages"]
    (project / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    assert mem.detect_environment(project).retracted == ["env.python_packages"]
    assert "env.python_packages" not in {w.key for w in mem.world_facts()}


@needs_toml
def test_a_stale_fact_is_rewritten(project):
    mem = Memory(project="app")
    mem.detect_environment(project, lifetime=0.01)
    import time

    time.sleep(0.05)
    assert "env.os" in mem.detect_environment(project).written


@needs_toml
def test_the_hook_records_them_at_session_start_unless_turned_off(project):
    mem = Memory(project="app")
    payload = {"hook_event_name": "SessionStart", "session_id": "h1", "cwd": str(project)}
    out = handle(payload, mem, harness="claude-code", user="ada")
    assert "env.python: 3.12.4" in out["hookSpecificOutput"]["additionalContext"]
    quiet = Memory(project="app")
    handle(payload, quiet, harness="claude-code", user="ada", environment=False)
    assert quiet.world_facts() == []


def test_without_a_toml_reader_lockfiles_are_still_read(tmp_path, monkeypatch):
    from commonmemo import environment

    monkeypatch.setattr(environment, "_toml", None)  # Python 3.10 without tomli
    (tmp_path / "requirements.txt").write_text("pydantic>=1.10\n")
    (tmp_path / "uv.lock").write_text(UV_LOCK)
    assert detect(tmp_path, this_interpreter=False)["env.python_packages"] == "pydantic 1.10.13"
