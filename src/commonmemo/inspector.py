"""`commonmemo inspect`: a local web inspector over a memory file (#25), read-only by default.

    commonmemo inspect --store sqlite:///.commonmemo/memory.db     # http://127.0.0.1:8765/
    commonmemo inspect --review --reviewer ada                     # also approve and reject (#45)

One command, no server or account: the standard library's HTTP server,
bound to 127.0.0.1, answering GET only unless `--review`. It shows

- **Overview:** recent sessions, users, world facts and proposals waiting for review.
- **User page:** the profile with where each fact came from, constraints,
  what was forgotten, past sessions, feedback.
- **Session timeline:** events in order, where a resume would start, and
  each turn trace: what memory pushed, what it left out, and for any item,
  why (the events it came from).

- **Review** (`/review`): world facts and procedure entries waiting for
  review, each with what it would replace. With `--review`, each has
  Approve (a world fact optionally corrected) and Reject; an exemplar
  candidate is approved for its user, and shared with everyone only with
  "personal data checked" (`confirm_procedure(..., scrubbed=True)`).

The same views are JSON under `/api/` (`/api/overview`, `/api/user/<id>`,
`/api/session/<id>`, `/api/why?session=&key=`, `/api/review`).

Stored memory is untrusted text, so every value is HTML-escaped and the
pages run no scripts at all (a strict Content-Security-Policy says so to
the browser). Review actions are plain HTML form posts, and each must carry
the token this server made at start, from a page with this server's own
Host and Origin: another site the reviewer has open can neither post an
approval (CSRF) nor read pages through a rebound DNS name. Every action
runs under `run_id="review:<reviewer>"`, so it is in the Log and
`revert(run_id=...)` undoes a review session.
"""

from __future__ import annotations

import argparse
import hmac
import json
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from .agent_tools import run_sync
from .cli import memory_options
from .inspector_ui import _e, _page, _q, overview_page, review_page, session_page, user_page, why_page
from .memory import Memory
from .traces import TurnTrace
from .views import overview, review_queue, session_view, user_view

CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; form-action 'self'; frame-ancestors 'none'"
"""No script, no network. `form-action 'self'` is for the overview's filter, a GET form; the read-only server answers
any POST with 405, and a review server only takes a POST that carries its token."""
CSP_REVIEW = CSP
MAX_FORM = 64 * 1024

def _jsonable(value: Any) -> Any:
    if isinstance(value, TurnTrace):
        return {k: getattr(value, k) for k in value.__dataclass_fields__}
    raise TypeError(type(value).__name__)


class InspectorHandler(BaseHTTPRequestHandler):
    """GET, and with review on, POST of the review forms; nothing else."""

    memory: Memory
    lock: threading.Lock
    review: bool = False
    reviewer: str = "local"
    token: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass

    def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", CSP_REVIEW if self.review else CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def _hosts(self) -> set[str]:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _foreign(self) -> bool:
        """A request whose Host isn't this server's: a rebound DNS name."""
        return self.headers.get("Host", "") not in self._hosts()

    def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
        if self._foreign():
            return self._send(403, _page("Forbidden", "<h1>Forbidden</h1>"))
        url = urlsplit(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        with self.lock:
            self.memory.knowledge.catch_up(self.memory.store)  # see what other processes wrote since
            if parts[:1] == ["api"]:
                api = parts[1:]
                value: Any = None
                if api == ["overview"]:
                    value = overview(self.memory)
                elif len(api) == 2 and api[0] == "user":
                    value = user_view(self.memory, api[1])
                elif len(api) == 2 and api[0] == "session":
                    value = session_view(self.memory, api[1])
                elif api == ["why"]:
                    found = self.memory.why(query.get("key", ""), session_id=query.get("session"))
                    value = None if found is None else {
                        "item": found.item, "trace": _jsonable(found.trace),
                        "sources": [e.to_dict() for e in found.sources]}
                elif api == ["review"]:
                    value = review_queue(self.memory)
                if value is None:
                    return self._send(404, json.dumps({"error": "not found"}), "application/json")
                return self._send(200, json.dumps(value, default=_jsonable), "application/json")
            mode = f"review, as {self.reviewer}" if self.review else "read-only"
            routes: dict[str, Callable[[], Optional[str]]] = {
                "": lambda: overview_page(self.memory, query.get("q", ""), mode) if not parts else None,
                "user": lambda: user_page(self.memory, parts[1], mode) if len(parts) == 2 else None,
                "session": lambda: session_page(self.memory, parts[1], query.get("show", ""), mode) if len(parts) == 2 else None,
                "why": lambda: why_page(self.memory, query.get("session", ""), query.get("key", ""), mode),
                "review": lambda: review_page(self.memory, self.token if self.review else None,
                                              query.get("done", ""), mode) if len(parts) == 1 else None,
            }
            page = routes.get(parts[0] if parts else "", lambda: None)()
        if page is None:
            return self._send(404, _page("Not found", "<h1>Not found</h1><p><a href='/'>Overview</a></p>"))
        self._send(200, page)

    def _refuse(self) -> None:
        self._send(405, _page("Read-only", "<h1>Read-only</h1><p>The inspector only shows memory "
                                           "(start it with --review to approve and reject).</p>", mode="read-only"))

    do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815

    def do_POST(self) -> None:  # noqa: N802 -- http.server's naming
        if not self.review:
            return self._refuse()
        origin = self.headers.get("Origin")
        if self._foreign() or (origin is not None and urlsplit(origin).netloc not in self._hosts()) \
                or self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return self._send(403, _page("Forbidden", "<h1>Forbidden</h1><p>Not from this inspector.</p>"))
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_FORM:
            return self._send(413, _page("Too large", "<h1>Too large</h1>"))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8", "replace")).items()}
        if not hmac.compare_digest(form.get("token", ""), self.token):
            return self._send(403, _page("Forbidden", "<h1>Forbidden</h1><p>Missing or stale form: reload the page.</p>"))
        parts = [unquote(p) for p in urlsplit(self.path).path.strip("/").split("/") if p]
        if len(parts) != 4 or parts[0] != "review" or parts[1] not in ("world", "procedure"):
            return self._send(404, _page("Not found", "<h1>Not found</h1>"))
        _, what, item, action = parts
        mem = self.memory
        try:
            with self.lock, mem.run(f"review:{self.reviewer}"):
                mem.knowledge.catch_up(mem.store)
                if (what, action) == ("world", "confirm"):
                    mem.confirm_fact(item, value=form.get("value") or None)
                    done = "Approved the world fact."
                elif (what, action) == ("world", "reject"):
                    mem.reject_fact(item, reason=form.get("reason") or None)
                    done = "Rejected the world fact."
                elif (what, action) == ("procedure", "confirm"):
                    mem.confirm_procedure(item)
                    done = "Approved the procedure entry."
                elif (what, action) == ("procedure", "share"):
                    mem.confirm_procedure(item, scope=mem.project_scope, scrubbed=form.get("scrubbed") == "1")
                    done = "Shared the exemplar with everyone (known personal values redacted)."
                elif (what, action) == ("procedure", "reject"):
                    mem.reject_procedure(item, reason=form.get("reason") or None)
                    done = "Rejected the procedure entry."
                else:
                    return self._send(404, _page("Not found", "<h1>Not found</h1>"))
        except ValueError as e:
            return self._send(400, _page("Not done", f"<h1>Not done</h1><p>{_e(e)}</p><p><a href='/review'>Back</a></p>",
                                         mode="review"))
        self.send_response(303)
        self.send_header("Location", f"/review?done={_q(done)}")
        self.send_header("Content-Length", "0")
        self.end_headers()


def serve(memory: Memory, *, host: str = "127.0.0.1", port: int = 8765, review: bool = False,
          reviewer: str = "local") -> ThreadingHTTPServer:
    """A server for `memory` (not started: call `serve_forever()`). `review`
    turns on the review actions, recorded under `run_id="review:<reviewer>"`."""
    handler = type("Handler", (InspectorHandler,), {
        "memory": memory, "lock": threading.Lock(), "review": review, "reviewer": reviewer,
        "token": secrets.token_urlsafe(32)})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="commonmemo inspect", description="A local web inspector, read-only by default.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--review", action="store_true", help="also approve and reject what waits for review")
    p.add_argument("--reviewer", default="local", help="who is reviewing; review actions carry run_id review:<name>")
    p.add_argument("--open", action="store_true", help="open it in the browser")
    args = memory_options(p, agent=None).parse_args(argv)
    memory = Memory(args.store, args.engine, project=args.project)
    server = serve(memory, port=args.port, review=args.review, reviewer=args.reviewer)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    mode = f"review, as {args.reviewer}" if args.review else "read-only"
    print(f"CommonMemo inspector ({mode}) at {url}  (Ctrl-C to stop)", file=sys.stderr)
    if args.open:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        run_sync(memory.aclose())
    return 0


if __name__ == "__main__":
    sys.exit(main())
