"""`commonmemo inspect`: a local, read-only web inspector over a memory file (#25).

    commonmemo inspect --store sqlite:///.commonmemo/memory.db     # http://127.0.0.1:8765/

One command, no server or account: the standard library's HTTP server,
bound to 127.0.0.1, answering GET only. It shows

- **Overview:** recent sessions, users, world facts and proposals waiting for review.
- **User page:** the profile with where each fact came from, constraints,
  what was forgotten, past sessions, feedback.
- **Session timeline:** events in order, where a resume would start, and
  each turn trace: what memory pushed, what it left out, and for any item,
  why (the events it came from).

The same views are JSON under `/api/` (`/api/overview`, `/api/user/<id>`,
`/api/session/<id>`, `/api/why?session=&key=`).

Read-only by decision (#25): approving facts or forgetting comes later with
the world-facts review screen. Stored memory is untrusted text, so every
value is HTML-escaped and the pages run no scripts at all (a strict
Content-Security-Policy says so to the browser).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .agent_tools import run_sync
from .cli import memory_options
from .memory import Memory
from .traces import TurnTrace
from .views import overview, session_view, user_view

CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; form-action 'none'; frame-ancestors 'none'"

STYLE = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dd; --card:#fff; --accent:#2f5d8a;
        --warn:#9a5b00; --chip:#eef2f6; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#171716; --fg:#ecebe6; --muted:#a3a29b; --line:#33332f; --card:#1f1f1d; --accent:#8db8e3;
          --warn:#e0a84a; --chip:#262c33; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; gap:16px; align-items:baseline; }
header a { color:var(--fg); font-weight:600; text-decoration:none; }
main { max-width:1000px; margin:0 auto; padding:20px 16px 60px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.muted { color:var(--muted); } a { color:var(--accent); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:10px 0; }
table { border-collapse:collapse; width:100%; } td, th { text-align:left; padding:6px 8px; border-top:1px solid var(--line);
  vertical-align:top; overflow-wrap:break-word; word-break:break-word; }
td.nowrap { white-space:nowrap; word-break:normal; } th { color:var(--muted); font-weight:500; font-size:13px; }
.chip { display:inline-block; background:var(--chip); border-radius:999px; padding:0 8px; font-size:12px; margin-right:4px; }
.warn { color:var(--warn); } code { font-size:13px; }
.wrap { overflow-x:auto; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _q(value: Any) -> str:
    return quote(str(value), safe="")


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width, initial-scale=1'><title>{_e(title)} · CommonMemo</title>"
            f"<style>{STYLE}</style></head><body><header><a href='/'>CommonMemo inspector</a>"
            f"<span class=muted>read-only</span></header><main>{body}</main></body></html>")


def _table(headers: list[str], rows: list[list[str]], empty: str = "nothing yet", nowrap: tuple[int, ...] = ()) -> str:
    """`nowrap`: columns of short words that shouldn't break."""
    if not rows:
        return f"<p class=muted>{_e(empty)}</p>"
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td class=nowrap>{c}</td>" if i in nowrap else f"<td>{c}</td>"
                                    for i, c in enumerate(row)) + "</tr>" for row in rows)
    return f"<div class=wrap><table><tr>{head}</tr>{body}</table></div>"


def overview_page(mem: Memory) -> str:
    v = overview(mem)
    sessions = [[f"<a href='/session/{_q(s['session_id'])}'>{_e(s['session_id'])}</a>",
                 f"<a href='/user/{_q(s['user_id'])}'>{_e(s['user_id'])}</a>" if s["user_id"] else "-",
                 _e(s["agent"] or "-"), _e(s["events"]), _e(s["last"])] for s in v["sessions"]]
    users = ", ".join(f"<a href='/user/{_q(u)}'>{_e(u)}</a>" for u in v["users"]) or "<span class=muted>none</span>"
    world = [[_e(w["key"]), _e(w["value"]), _e(w["scope"]), _e(w["source"])] for w in v["world"]]
    pending = [[_e(w["key"]), _e(w["value"]), _e(w["scope"]), _e(w["user_id"] or "-")] for w in v["pending"]]
    return _page("Overview", f"<h1>Overview</h1><h2>Users</h2><p>{users}</p>"
                 f"<h2>Recent sessions</h2>{_table(['session', 'user', 'agent', 'events', 'last event'], sessions)}"
                 f"<h2>World facts</h2>{_table(['key', 'value', 'scope', 'source'], world)}"
                 f"<h2>Waiting for review</h2>{_table(['key', 'value', 'scope', 'proposed by'], pending, 'no proposals')}")


def user_page(mem: Memory, user_id: str) -> str:
    v = user_view(mem, user_id)
    facts = [[_e(f["key"]), _e(f["value"]), _e(f["since"]), _e(len(f["events"]))] for f in v["facts"]]
    negatives = [[_e(n["statement"]), _e(n["reason"] or ""), _e(n["source"])] for n in v["negatives"]]
    forgotten = [[_e(f["key"]), _e(f["when"])] for f in v["forgotten"]]
    recaps = [[_e(r["ended"]), _e(r["agent"] or "-"),
               f"<a href='/session/{_q(r['session'])}'>{_e(r['session'])}</a>", _e(r["summary"])] for r in v["recaps"]]
    sessions = ", ".join(f"<a href='/session/{_q(s)}'>{_e(s)}</a>" for s in v["sessions"]) or "none"
    feedback = "".join(f"<li>{_e(line)}</li>" for line in v["feedback"]["recent"])
    return _page(f"User {user_id}", f"<h1>User {_e(user_id)}</h1><p class=muted>sessions: {sessions}</p>"
                 f"<h2>Profile</h2>{_table(['key', 'value', 'since', 'source events'], facts, 'no facts')}"
                 f"<h2>Never</h2>{_table(['statement', 'reason', 'source'], negatives, 'no standing constraints')}"
                 f"<h2>Forgotten</h2>{_table(['key', 'when'], forgotten, 'nothing forgotten')}"
                 f"<h2>Past sessions</h2>{_table(['ended', 'agent', 'session', 'recap'], recaps, 'no recaps')}"
                 f"<h2>Feedback ({_e(v['feedback']['count'])})</h2><ul>{feedback}</ul>")


def _trace_card(trace: TurnTrace, labels: dict[str, str], session_id: str) -> str:
    rows = []
    for i in trace.items:
        key = i.get("key") or i.get("id")
        shown = i.get("key") or labels.get(i.get("id") or "") or i.get("id")
        detail = " ".join(f"<span class=chip>{_e(k)} {_e(i[k])}</span>"
                          for k in ("scope", "source", "status", "score", "session") if i.get(k) is not None)
        why = f"<a href='{_e(f'/why?session={_q(session_id)}&key={_q(key)}')}'>why</a>"
        rows.append([_e(i["kind"]), _e(shown), detail, why])
    cut = ", ".join(f"{n} {k}" for k, n in trace.more.items() if n > 0)
    text = f"<details><summary>pushed text</summary><pre>{_e(trace.text)}</pre></details>" if trace.text else ""
    return (f"<div class=card><b>{_e(trace.reason)} push</b> <span class=muted>{_e(trace.chars)} chars"
            f"{', by ' + _e(trace.agent) if trace.agent else ''}</span>"
            f"{_table(['kind', 'item', 'from', ''], rows, 'nothing pushed', nowrap=(0, 3))}"
            f"{f'<p class=warn>left out: {_e(cut)}</p>' if cut else ''}{text}</div>")


def session_page(mem: Memory, session_id: str) -> Optional[str]:
    v = session_view(mem, session_id)
    if v is None:
        return None
    timeline = [[_e(t["when"]), _e(t["agent"] or "-"), _e(t["line"])] for t in v["timeline"]]
    resume = "".join(f"<li>{_e(line)}</li>" for line in v["resume"]) or "<li class=muted>nothing to resume</li>"
    traces = "".join(_trace_card(t, v["labels"], session_id) for t in v["traces"]) or "<p class=muted>no pushes</p>"
    user = f"<a href='/user/{_q(v['user_id'])}'>{_e(v['user_id'])}</a>" if v["user_id"] else "-"
    status = f"closed {_e(v['closed'])}" if v["closed"] else "open"
    return _page(f"Session {session_id}", f"<h1>Session {_e(session_id)}</h1>"
                 f"<p class=muted>user {user} · agent {_e(v['agent'] or '-')} · opened {_e(v['opened'])} · {status}</p>"
                 f"<h2>Turn traces</h2>{traces}<h2>A resume would start from</h2><ul>{resume}</ul>"
                 f"<h2>Timeline</h2>{_table(['when', 'agent', 'event'], timeline, 'no events', nowrap=(1,))}")


def why_page(mem: Memory, session_id: str, key: str) -> Optional[str]:
    found = mem.why(key, session_id=session_id)
    if found is None:
        return None
    rows = [[_e(e.kind), f"<a href='/session/{_q(e.session_id)}'>{_e(e.session_id)}</a>" if e.session_id else "-",
             _e(getattr(e, "source", "") or ""), _e(e.id)] for e in found.sources]
    return _page(f"Why {key}", f"<h1>Why the agent got {_e(key)}</h1>"
                 f"<p>{_e(found.item['kind'])}, pushed in a {_e(found.trace.reason)} push to session "
                 f"<a href='/session/{_q(session_id)}'>{_e(session_id)}</a>.</p><h2>It came from</h2>"
                 f"{_table(['event', 'session', 'source', 'id'], rows, 'its source events are no longer in the Log')}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, TurnTrace):
        return {k: getattr(value, k) for k in value.__dataclass_fields__}
    raise TypeError(type(value).__name__)


class InspectorHandler(BaseHTTPRequestHandler):
    """GET only; every other method is refused."""

    memory: Memory
    lock: threading.Lock

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass

    def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 -- http.server's naming
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
                if value is None:
                    return self._send(404, json.dumps({"error": "not found"}), "application/json")
                return self._send(200, json.dumps(value, default=_jsonable), "application/json")
            routes: dict[str, Callable[[], Optional[str]]] = {
                "": lambda: overview_page(self.memory),
                "user": lambda: user_page(self.memory, parts[1]) if len(parts) == 2 else None,
                "session": lambda: session_page(self.memory, parts[1]) if len(parts) == 2 else None,
                "why": lambda: why_page(self.memory, query.get("session", ""), query.get("key", "")),
            }
            page = routes.get(parts[0] if parts else "", lambda: None)()
        if page is None:
            return self._send(404, _page("Not found", "<h1>Not found</h1><p><a href='/'>Overview</a></p>"))
        self._send(200, page)

    def _refuse(self) -> None:
        self._send(405, _page("Read-only", "<h1>Read-only</h1><p>The inspector only shows memory.</p>"))

    do_POST = do_PUT = do_DELETE = do_PATCH = _refuse  # noqa: N815


def serve(memory: Memory, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """A server for `memory` (not started: call `serve_forever()`)."""
    handler = type("Handler", (InspectorHandler,), {"memory": memory, "lock": threading.Lock()})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="commonmemo inspect", description="A local, read-only web inspector.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="open it in the browser")
    args = memory_options(p, agent=None).parse_args(argv)
    memory = Memory(args.store, args.engine, project=args.project)
    server = serve(memory, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"CommonMemo inspector (read-only) at {url}  (Ctrl-C to stop)", file=sys.stderr)
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
