"""The resident side of a resident hook (#92): hooks handled by a process that stays loaded.

    python -m commonmemo.hookserver        # normally started by `commonmemo hook --resident`

Listens on a Unix socket in a private per-user directory, handles one hook event per connection
exactly as `commonmemo hook` would (`hooks.run`), and keeps one `Memory` open per distinct
configuration, so an event costs a push, not an interpreter start. It serves its own user only,
handles one event at a time, exits after `COMMONMEMO_HOOK_IDLE` idle seconds (default 600), and
steps aside when a client runs different code (a new version replaces it).
"""

from __future__ import annotations

import io
import os
import socket
import socketserver
import struct
import sys
import threading
import time
from typing import Any

from .hookclient import fingerprint, socket_path

IDLE = float(os.environ.get("COMMONMEMO_HOOK_IDLE", "600"))
MAX_KEPT = 8


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str) -> None:
        self.fingerprint = fingerprint()
        self.kept: dict[Any, Any] = {}
        self.busy = threading.Lock()  # one event at a time: the environment overlay below is process-wide
        self.last = time.monotonic()
        super().__init__(path, Handler)
        os.chmod(path, 0o600)


class Handler(socketserver.StreamRequestHandler):
    server: Server

    def handle(self) -> None:
        if not self._ours():
            return
        try:
            request = self._read()
        except (ValueError, IndexError):
            return
        if request["fingerprint"] != self.server.fingerprint:
            self._reply(b"restart\n")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        with self.server.busy:
            self.server.last = time.monotonic()
            code, out, err = self._run(request)
            body = out.encode()
            self._reply(f"{code} {len(body)}\n".encode() + body + err.encode())
            self.server.last = time.monotonic()

    def _read(self) -> dict[str, Any]:
        """The client's NUL-separated fields (see `hookclient.encode`)."""
        fields = self.rfile.read().decode().split("\0")
        fingerprint_, cwd, n = fields[0], fields[1], int(fields[2])
        argv = fields[3: 3 + n]
        m = int(fields[3 + n])
        env = dict(f.split("=", 1) for f in fields[4 + n: 4 + n + m])
        return {"fingerprint": fingerprint_, "cwd": cwd, "argv": argv, "env": env, "stdin": "\0".join(fields[4 + n + m:])}

    def _ours(self) -> bool:
        """Only the user who started the server may use it (where the OS says who is asking)."""
        try:
            pid, uid, gid = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        except (AttributeError, OSError):
            return True  # no peer credentials here: the 0700 directory is the guard
        return uid == os.getuid()

    def _reply(self, answer: bytes) -> None:
        try:
            self.wfile.write(answer)
        except OSError:
            pass

    def _run(self, request: dict[str, Any]) -> tuple[int, str, str]:
        from . import hooks

        env = {k: v for k, v in (request.get("env") or {}).items() if str(k).startswith("COMMONMEMO_")}
        argv = [str(a) for a in request.get("argv") or []]
        key = (tuple(argv), tuple(sorted(env.items())))
        out, err = io.StringIO(), io.StringIO()
        saved = {k: os.environ.get(k) for k in {*env, *(k for k in os.environ if k.startswith("COMMONMEMO_"))}}
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        cwd = os.getcwd()
        try:
            if request.get("cwd"):
                os.chdir(request["cwd"])  # a relative store path (`.commonmemo/memory.db`) means the client's directory
            while len(self.server.kept) >= MAX_KEPT and key not in self.server.kept:
                hooks.run_sync(self.server.kept.pop(next(iter(self.server.kept))).aclose())
            code = hooks.run(argv, io.StringIO(request.get("stdin") or ""), out, err, kept=self.server.kept, key=key)
        except BaseException as e:  # noqa: BLE001 -- never take the server down over one event
            err.write(f"commonmemo hook server: {type(e).__name__}: {e}\n")
            code = 0
        finally:
            try:
                os.chdir(cwd)
            except OSError:
                pass
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        return code, out.getvalue(), err.getvalue()


def _bind(path: str) -> Server | None:
    try:
        return Server(path)
    except OSError:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(path)
            return None  # another server is running: leave it be
        except OSError:
            os.unlink(path)  # nobody listens: a stale socket file from a server that died
            return Server(path)
        finally:
            probe.close()


def main() -> int:
    server = _bind(socket_path())
    if server is None:
        return 0

    def watch() -> None:
        while True:
            time.sleep(min(5.0, max(0.5, IDLE / 4)))
            if time.monotonic() - server.last > IDLE and not server.busy.locked():
                server.shutdown()
                return

    threading.Thread(target=watch, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        from . import hooks

        for memory in list(server.kept.values()):
            try:
                hooks.run_sync(memory.aclose())
            except Exception:  # noqa: BLE001
                pass
        server.server_close()
        try:
            os.unlink(socket_path())
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
