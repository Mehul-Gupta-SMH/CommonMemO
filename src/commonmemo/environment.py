"""Environment facts: the interpreter, OS and package versions a project runs on (#57).

Agents guess these when nobody says, and in coding work a wrong guess about
the Python version or a library's major version is a common, avoidable
failure. `detect_environment` reads them from the machine and the project's
files and records them as world facts:

    mem.detect_environment(".")       # also run by the harness hooks at SessionStart

| Key | Value, for example | From |
|---|---|---|
| `env.python` | `3.13.1 (requires >=3.10)` | the project's virtualenv (`.venv`, `venv`, `$VIRTUAL_ENV`), else `.python-version`, else this interpreter (see below); `requires-python` |
| `env.os` | `Linux 6.8.0 (x86_64)` | this machine |
| `env.python_packages` | `httpx 0.27.2, pydantic 1.10.13 (installed 1.10.9)` | the project's pins (`pyproject.toml`, `requirements*.txt`), at the version locked in `uv.lock` / `poetry.lock`, and the installed version where it differs |
| `env.node` | `v22.11.0` | `node --version`, only when there is a `package.json` |
| `env.node_packages` | `react 18.3.1, vite 5.4.10` | `package.json`, locked in `package-lock.json`, installed in `node_modules` |

- **Trusted, and they refresh.** They are written with `source="tool"`
  (checked against the machine), so an agent's guess can't replace them; it
  is kept as disputed. Each lasts `lifetime` (7 days by default) and is
  rewritten when it changes or has gone stale; an unchanged value isn't
  written again, so detecting at every start doesn't churn the history. A
  key that is no longer detected (the last pin removed) is retracted.
- **Pinned.** `env.*` keys always go in the stable half of the push: they
  rarely change, so prompt caching keeps working. Packages are one line per
  ecosystem, direct dependencies only (at most `MAX_PACKAGES`), so the
  stable half stays small.
- **Whose interpreter.** Called from an app (`this_interpreter=True`, the
  default), the running interpreter is the project's, so with no virtualenv
  in the project its version and installed packages are used. The hooks
  run in commonmemo's own interpreter, which may not be the project's, so
  they pass `this_interpreter=False` and use only what the project's files
  and virtualenv say.
- **Nothing secret.** Only the values above are read: never environment
  variables, and requirement lines with a URL (which can carry a token) are
  skipped.

Reading `pyproject.toml` and the TOML lockfiles needs `tomllib` (Python
3.11+) or `tomli`; without either, on 3.10, `requirements*.txt` is still read
and the lockfiles are read line by line.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from .facts import parse_duration

if TYPE_CHECKING:
    from .memory import Memory

try:
    import tomllib as _toml
except ImportError:  # Python 3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ImportError:
        _toml = None  # type: ignore[assignment]

log = logging.getLogger("commonmemo")

PREFIX = "env."
"""World-fact keys under this prefix are environment facts, always pinned."""

MAX_PACKAGES = 60
"""Most packages listed per ecosystem; the rest are counted ("and 4 more")."""

LIFETIME = "7d"

_SOURCE = "tool"


def is_environment_key(key: str) -> bool:
    return key.startswith(PREFIX)


@dataclass(frozen=True)
class EnvironmentResult:
    """What `detect_environment` found and wrote."""

    facts: dict[str, str]
    written: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    retracted: list[str] = field(default_factory=list)


# -- names and requirements ----------------------------------------------------


def _norm(name: str) -> str:
    """A Python distribution's name as PEP 503 compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;]*)")


def _requirement(line: str) -> Optional[tuple[str, str]]:
    """`(name, specifier)` from a PEP 508 line, or None for what isn't a
    plain named requirement (URLs, paths, options, `name @ url`)."""
    line = line.split(" #", 1)[0].strip()
    if not line or line.startswith(("#", "-", ".", "/")) or "://" in line or "@" in line:
        return None
    m = _REQ.match(line)
    if not m:
        return None
    return m.group(1), m.group(2).strip().replace(" ", "")


def _read_toml(path: Path) -> Optional[dict[str, Any]]:
    if _toml is None:
        return None
    try:
        return _toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("commonmemo: couldn't read %s: %s", path, e)
        return None


def _poetry_spec(value: Any) -> str:
    spec = value.get("version", "") if isinstance(value, dict) else str(value)
    return "" if spec == "*" else spec


def _python_pins(root: Path) -> tuple[dict[str, tuple[str, str]], Optional[str]]:
    """`{normalized name: (name as written, specifier)}` and `requires-python`."""
    pins: dict[str, tuple[str, str]] = {}
    requires: Optional[str] = None

    def add(name: str, spec: str) -> None:
        pins.setdefault(_norm(name), (name, spec))

    pyproject = root / "pyproject.toml"
    data = _read_toml(pyproject) if pyproject.is_file() else None
    if data:
        project = data.get("project") or {}
        requires = project.get("requires-python")
        lines = list(project.get("dependencies") or [])
        for extra in (project.get("optional-dependencies") or {}).values():
            lines += extra
        for group in (data.get("dependency-groups") or {}).values():
            lines += [x for x in group if isinstance(x, str)]
        for line in lines:
            req = _requirement(line)
            if req:
                add(*req)
        poetry = (data.get("tool") or {}).get("poetry") or {}
        sections = [poetry.get("dependencies") or {}, poetry.get("dev-dependencies") or {}]
        sections += [g.get("dependencies") or {} for g in (poetry.get("group") or {}).values()]
        for section in sections:
            for name, value in section.items():
                if name == "python":
                    requires = requires or _poetry_spec(value)
                elif not (isinstance(value, dict) and ("path" in value or "url" in value or "git" in value)):
                    add(name, _poetry_spec(value))
    for req_file in sorted(root.glob("requirements*.txt")):
        try:
            text = req_file.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            req = _requirement(line)
            if req:
                add(*req)
    return pins, requires


_LOCK_PACKAGE = re.compile(r'(?ms)^\[\[package\]\]\s*\nname\s*=\s*"([^"]+)"\s*\nversion\s*=\s*"([^"]+)"')


def _python_locked(root: Path) -> dict[str, str]:
    """`{normalized name: version}` from `uv.lock` or `poetry.lock`."""
    for name in ("uv.lock", "poetry.lock"):
        path = root / name
        if not path.is_file():
            continue
        data = _read_toml(path)
        if data is not None:
            return {_norm(p["name"]): str(p["version"]) for p in data.get("package") or []
                    if "name" in p and "version" in p}
        try:  # no TOML reader: both formats start each package with name, then version
            return {_norm(n): v for n, v in _LOCK_PACKAGE.findall(path.read_text(encoding="utf-8"))}
        except OSError:
            return {}
    return {}


def _venv(root: Path) -> Optional[Path]:
    for candidate in (root / ".venv", root / "venv"):
        if (candidate / "pyvenv.cfg").is_file():
            return candidate
    active = os.environ.get("VIRTUAL_ENV")
    if active and (Path(active) / "pyvenv.cfg").is_file():
        return Path(active)
    return None


def _site_packages(venv: Path) -> list[Path]:
    return [p for p in [*venv.glob("lib/python*/site-packages"), venv / "Lib" / "site-packages"] if p.is_dir()]


def _python_installed(venv: Optional[Path], this_interpreter: bool) -> dict[str, str]:
    """`{normalized name: version}` installed in the project's virtualenv,
    else (with `this_interpreter`) in this interpreter's environment."""
    from importlib import metadata

    if venv:
        dists = metadata.distributions(path=[str(p) for p in _site_packages(venv)])
    elif this_interpreter:
        dists = metadata.distributions()
    else:
        return {}
    out: dict[str, str] = {}
    for d in dists:
        name = d.metadata.get("Name") if d.metadata else None
        if name:
            out.setdefault(_norm(name), d.version)
    return out


def _python_version(root: Path, venv: Optional[Path], this_interpreter: bool) -> Optional[str]:
    if venv:
        cfg = dict(line.split("=", 1) for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()
                   if "=" in line)
        cfg = {k.strip(): v.strip() for k, v in cfg.items()}
        version = cfg.get("version_info") or cfg.get("version")
        if version:
            return ".".join(version.split(".")[:3])
    pinned = root / ".python-version"
    if pinned.is_file():
        first = pinned.read_text(encoding="utf-8").strip().splitlines()
        if first:
            return first[0].strip()
    return platform.python_version() if this_interpreter else None


def _packages_line(pins: dict[str, tuple[str, str]], locked: dict[str, str], installed: dict[str, str]) -> str:
    items = []
    for key, (name, spec) in sorted(pins.items()):
        lock, have = locked.get(key), installed.get(key)
        exact = re.fullmatch(r"===?([^,*]+)", spec)
        lock = lock or (exact.group(1) if exact else None)  # `==1.10.13` is as good as a lock
        if lock:
            items.append(f"{name} {lock}" + (f" (installed {have})" if have and have != lock else ""))
        elif have:
            items.append(f"{name} {have}" + (f" (requires {spec})" if spec else ""))
        else:
            items.append(f"{name} {spec}".strip())
    more = len(items) - MAX_PACKAGES
    return ", ".join(items[:MAX_PACKAGES]) + (f", and {more} more" if more > 0 else "")


# -- node ----------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _node(root: Path) -> dict[str, str]:
    package = _read_json(root / "package.json")
    if package is None:
        return {}
    out: dict[str, str] = {}
    try:
        run = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if run.returncode == 0 and run.stdout.strip():
            out["env.node"] = run.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    pins = {name: (name, str(spec)) for section in ("dependencies", "devDependencies")
            for name, spec in (package.get(section) or {}).items()
            if isinstance(spec, str) and "://" not in spec and not spec.startswith(("file:", "link:", "git"))}
    if pins:
        lock = (_read_json(root / "package-lock.json") or {}).get("packages") or {}
        locked = {n: str(lock[f"node_modules/{n}"].get("version")) for n in pins
                  if isinstance(lock.get(f"node_modules/{n}"), dict) and lock[f"node_modules/{n}"].get("version")}
        installed = {}
        for n in pins:
            meta = _read_json(root / "node_modules" / n / "package.json")
            if meta and meta.get("version"):
                installed[n] = str(meta["version"])
        out["env.node_packages"] = _packages_line(pins, locked, installed)
    return out


# -- detection -----------------------------------------------------------------


def _os() -> str:
    system = platform.system()
    if system == "Darwin":
        system, release = "macOS", platform.mac_ver()[0]
    else:
        release = platform.release()
    return f"{system} {release} ({platform.machine()})".replace("  ", " ")


def detect(path: Union[str, Path] = ".", *, this_interpreter: bool = True) -> dict[str, str]:
    """The environment facts for the project at `path`, without writing them."""
    root = Path(path).resolve()
    venv = _venv(root)
    pins, requires = _python_pins(root)
    python = _python_version(root, venv, this_interpreter)
    facts = {"env.os": _os()}
    if python or requires:
        facts["env.python"] = (python or "unknown") + (f" (requires {requires})" if requires else "")
    if pins:
        installed = _python_installed(venv, this_interpreter)
        facts["env.python_packages"] = _packages_line(pins, _python_locked(root), installed)
    facts.update(_node(root))
    return facts


def record(memory: "Memory", path: Union[str, Path] = ".", *, scope: Optional[str] = None,
           lifetime: Union[float, str] = LIFETIME, this_interpreter: bool = True) -> EnvironmentResult:
    """Detect, then write what changed or went stale; retract what is gone."""
    facts = detect(path, this_interpreter=this_interpreter)
    scope = scope or memory.project_scope
    ttl = parse_duration(lifetime)
    current = {w.key: w for w in memory.knowledge.world_facts([scope]) if is_environment_key(w.key)}
    written, unchanged, retracted = [], [], []
    for key, value in facts.items():
        if key in current and current[key].value == value and current[key].source == _SOURCE:
            unchanged.append(key)
            continue
        memory.learn(key, value, scope=scope, ttl=ttl, source=_SOURCE)
        written.append(key)
    for key, w in current.items():
        if key not in facts and w.source == _SOURCE:
            memory.retract_fact(key, scope=scope, reason="no longer detected")
            retracted.append(key)
    return EnvironmentResult(facts, written, unchanged, retracted)


__all__ = ["EnvironmentResult", "MAX_PACKAGES", "PREFIX", "detect", "is_environment_key", "record"]
