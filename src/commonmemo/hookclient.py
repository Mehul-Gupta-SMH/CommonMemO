"""The thin side of a resident hook (#92): send the event to a running `hookserver`, or start one.

`commonmemo hook` costs about 250 ms because every event starts a Python that imports the whole
library. With `--resident` the process imports almost nothing (json, os, socket), hands the event
to a server that already has it loaded, and prints what the server answers.

If no server runs, this one starts one (detached) and handles the event itself in the ordinary way,
so a first hook is never slower than a hook without `--resident`. Anything that goes wrong with the
server falls back the same way: a hook never fails the harness.

Nothing here may import the library at module level: `import commonmemo` is lazy for this reason.
"""

from __future__ import annotations

import os
import sys

try:
    import _socket as socket  # the C module: `socket` itself imports enum, selectors and re, about 10 ms
except ImportError:  # pragma: no cover -- an interpreter without it
    import socket  # type: ignore[no-redef]

TIMEOUT = 30.0
"""Seconds a hook waits for the server's answer (a push on a big Postgres log can take a while)."""


def runtime_dir() -> str:
    """The per-user directory holding the socket: owned by the user, mode 0700."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base and os.path.isdir(base):
        path = os.path.join(base, "commonmemo")
    else:
        import tempfile

        path = os.path.join(tempfile.gettempdir(), f"commonmemo-{os.getuid()}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    st = os.stat(path)
    if st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise PermissionError(f"{path} is not private to this user")
    return path


def socket_path() -> str:
    return os.path.join(runtime_dir(), "hook.sock")


def fingerprint() -> str:
    """Names this installation: a server started by other code is not talked to, it is replaced."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        stamp = max(os.stat(os.path.join(here, f)).st_mtime_ns for f in ("hooks.py", "memory.py", "hookserver.py"))
    except OSError:
        stamp = 0
    return f"{sys.version_info[:2]}:{stamp}"


def encode(argv: list[str], text: str) -> bytes:
    """The request: NUL-separated fields (no JSON: importing json costs 6 ms a hook)."""
    env = [f"{k}={v}" for k, v in os.environ.items() if k.startswith("COMMONMEMO_")]
    return "\0".join([fingerprint(), os.getcwd(), str(len(argv)), *argv, str(len(env)), *env, text]).encode()


def request(argv: list[str], text: str, path: str | None = None) -> tuple[int, str, str] | None:
    """`(exit code, stdout, stderr)` from the server, or None when there is no (usable) server."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(TIMEOUT)
            s.connect(path or socket_path())
            s.sendall(encode(argv, text))
            s.shutdown(socket.SHUT_WR)
            data = b""
            while chunk := s.recv(65536):
                data += chunk
        finally:
            s.close()
        header, _, rest = data.partition(b"\n")
        code, size = (int(x) for x in header.split())
        return code, rest[:size].decode(), rest[size:].decode()
    except (OSError, ValueError):  # no server, a refusal ("restart"), or a reply that is not ours
        return None


def start_server() -> None:
    import subprocess

    log = os.environ.get("COMMONMEMO_HOOK_LOG")  # where the server writes what goes wrong (default: nowhere)
    err = open(log, "a") if log else subprocess.DEVNULL
    subprocess.Popen([sys.executable, "-m", "commonmemo.hookserver"], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=err, start_new_session=True, close_fds=True)


def main(argv: list[str]) -> int:
    argv = [a for a in argv if a != "--resident"]
    text = sys.stdin.read()
    answer = None
    if hasattr(socket, "AF_UNIX") and os.environ.get("COMMONMEMO_RESIDENT_OFF", "").lower() not in ("1", "true"):
        try:
            answer = request(argv, text)
            if answer is None:
                start_server()
        except Exception:  # noqa: BLE001 -- never fail the harness over the server
            answer = None
    if answer is not None:
        code, out, err = answer
        sys.stdout.write(out)
        sys.stderr.write(err)
        return code
    import io

    from . import hooks  # the ordinary hook, this once

    return hooks.run(argv, io.StringIO(text), sys.stdout, sys.stderr)
