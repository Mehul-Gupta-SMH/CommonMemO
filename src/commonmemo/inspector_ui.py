"""The inspector's pages (#25, #45, #87): HTML from the views, with no script anywhere.

The server (`inspector.py`) answers requests; this module only turns view data into pages. Stored
memory is untrusted text, so every value goes through `_e`, and the pages run no script (the
Content-Security-Policy says so): filters and disclosures are links, GET forms and `<details>`.

The look is a handful of tokens at the top of `STYLE`, light and dark; a reference style would
replace only those.
"""

from __future__ import annotations

import html
import time
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote

from .traces import TurnTrace
from .views import overview, review_queue, session_view, user_view

if TYPE_CHECKING:
    from .memory import Memory

STYLE = """
:root { color-scheme: light dark;
  --bg:#f6f6f3; --surface:#fff; --sunken:#f0f0ec; --ink:#16181c; --ink-2:#464a52; --muted:#757981; --line:#e4e3dd;
  --accent:#2f54c4; --accent-ink:#fff; --accent-soft:#e8edfb; --ok:#1d7a4b; --ok-soft:#e3f4ea;
  --bad:#b3261e; --bad-soft:#fbe9e7; --warn:#935800; --warn-soft:#fdf0d9;
  --k-message:#2f54c4; --k-tool:#5a6270; --k-memory:#7a45c9; --k-learn:#1d7a4b; --k-other:#8a8d94;
  --radius:10px; --shadow:0 1px 2px rgba(20,22,28,.05); }
@media (prefers-color-scheme: dark) { :root {
  --bg:#121315; --surface:#1a1b1e; --sunken:#202226; --ink:#ecedef; --ink-2:#b9bcc3; --muted:#8e929b; --line:#2c2e33;
  --accent:#8aa4f2; --accent-ink:#0f1320; --accent-soft:#222a45; --ok:#5fcf93; --ok-soft:#18281f;
  --bad:#f2827a; --bad-soft:#331d1b; --warn:#e6b05a; --warn-soft:#30260f;
  --k-message:#8aa4f2; --k-tool:#9aa3b2; --k-memory:#b793f0; --k-learn:#5fcf93; --k-other:#8e929b;
  --shadow:none; } }
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color:var(--accent); text-underline-offset:2px; } a:hover { text-decoration-thickness:2px; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:4px; }
.skip { position:absolute; left:-999px; } .skip:focus { left:12px; top:8px; background:var(--surface); padding:6px 10px; z-index:9; }
header.top { background:var(--surface); border-bottom:1px solid var(--line); }
header.top > div { max-width:1080px; margin:0 auto; padding:0 16px; display:flex; align-items:center; gap:4px; flex-wrap:wrap; min-height:52px; }
.brand { font-weight:650; color:var(--ink); text-decoration:none; margin-right:14px; letter-spacing:-.01em; }
.brand::before { content:""; display:inline-block; width:10px; height:10px; border-radius:3px; background:var(--accent); margin-right:8px; }
nav.main a { color:var(--ink-2); text-decoration:none; padding:15px 10px 13px; border-bottom:2px solid transparent; display:inline-block; }
nav.main a:hover { color:var(--ink); } nav.main a[aria-current=page] { color:var(--ink); border-bottom-color:var(--accent); font-weight:550; }
.badge { display:inline-block; min-width:20px; padding:0 6px; margin-left:6px; border-radius:999px; background:var(--accent); color:var(--accent-ink);
  font-size:12px; line-height:20px; text-align:center; font-weight:600; }
.mode { margin-left:auto; font-size:12px; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:1px 10px; }
.mode.review { color:var(--ok); border-color:var(--ok); }
main { max-width:1080px; margin:0 auto; padding:22px 16px 72px; }
.crumbs { font-size:13px; color:var(--muted); margin:0 0 6px; } .crumbs a { color:var(--muted); }
h1 { font-size:24px; line-height:1.25; margin:0 0 6px; letter-spacing:-.015em; }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:600; margin:34px 0 10px; display:flex; align-items:baseline; gap:8px; }
h2 .count { font-weight:500; letter-spacing:0; text-transform:none; }
.lede { color:var(--ink-2); margin:0 0 4px; } .muted { color:var(--muted); } .small { font-size:13px; }
.meta { display:flex; flex-wrap:wrap; gap:6px 14px; color:var(--ink-2); font-size:14px; margin:6px 0 0; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin:18px 0 0; }
.tile { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:12px 14px; box-shadow:var(--shadow); color:inherit; text-decoration:none; display:block; }
a.tile:hover { border-color:var(--accent); } .tile b { display:block; font-size:26px; line-height:1.2; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
.tile span { color:var(--muted); font-size:13px; } .tile.hot { background:var(--accent-soft); border-color:var(--accent); }
.banner { display:flex; gap:12px; align-items:center; flex-wrap:wrap; background:var(--warn-soft); border:1px solid var(--warn); color:var(--ink);
  border-radius:var(--radius); padding:10px 14px; margin:16px 0 0; }
.banner.ok { background:var(--ok-soft); border-color:var(--ok); } .banner .grow { flex:1; min-width:200px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:14px 16px; margin:10px 0; box-shadow:var(--shadow); }
.cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); gap:12px; } .cards .card { margin:0; }
.card h3 { margin:0 0 2px; font-size:16px; } .card .sub { color:var(--muted); font-size:13px; }
.empty { border:1px dashed var(--line); border-radius:var(--radius); padding:18px 16px; color:var(--muted); background:transparent; }
.empty b { color:var(--ink-2); display:block; margin-bottom:2px; }
.wrap { overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); }
table { border-collapse:collapse; width:100%; } th, td { text-align:left; padding:9px 12px; vertical-align:top; overflow-wrap:anywhere; }
th { color:var(--muted); font-weight:550; font-size:12px; text-transform:uppercase; letter-spacing:.05em; background:var(--sunken); }
td { border-top:1px solid var(--line); } tbody tr:hover td { background:var(--sunken); } td.nowrap, th.nowrap { white-space:nowrap; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.chip { display:inline-block; background:var(--sunken); color:var(--ink-2); border-radius:999px; padding:0 9px; font-size:12px; line-height:20px; margin:0 4px 2px 0; white-space:nowrap; }
.chip.ok { background:var(--ok-soft); color:var(--ok); } .chip.bad { background:var(--bad-soft); color:var(--bad); }
.chip.warn { background:var(--warn-soft); color:var(--warn); } .chip.accent { background:var(--accent-soft); color:var(--accent); }
code, pre { font:13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; } pre { white-space:pre-wrap; overflow-wrap:anywhere; margin:8px 0; background:var(--sunken); border-radius:8px; padding:10px 12px; }
.search { display:flex; gap:8px; margin:22px 0 0; flex-wrap:wrap; } .search input { flex:1; min-width:200px; }
input[type=text], input[type=search] { font:inherit; padding:7px 10px; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--ink); max-width:100%; }
input::placeholder { color:var(--muted); }
button, .btn { font:inherit; padding:7px 14px; border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--ink); cursor:pointer; text-decoration:none; display:inline-block; line-height:1.3; }
button:hover, .btn:hover { border-color:var(--ink-2); } button.primary, .btn.primary { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); font-weight:550; }
button.primary:hover, .btn.primary:hover { filter:brightness(1.07); } button.danger { color:var(--bad); }
.filters { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 12px; align-items:center; }
.filters a { text-decoration:none; color:var(--ink-2); border:1px solid var(--line); background:var(--surface); border-radius:999px; padding:2px 12px; font-size:13px; }
.filters a[aria-current=true] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
.turn { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); margin:12px 0; box-shadow:var(--shadow); overflow:hidden; }
.turn > header { display:flex; gap:10px; align-items:baseline; padding:10px 14px; background:var(--sunken); border-bottom:1px solid var(--line); flex-wrap:wrap; }
.turn > header b { font-size:14px; } .turn > header .said { color:var(--ink-2); flex:1; min-width:200px; overflow-wrap:anywhere; }
ol.events { list-style:none; margin:0; padding:0; } ol.events li { display:grid; grid-template-columns:62px 100px 1fr; gap:8px; padding:7px 14px; border-top:1px solid var(--line); align-items:baseline; }
ol.events li:first-child { border-top:0; } ol.events .t { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; }
ol.events .k { font-size:12px; color:var(--muted); display:flex; align-items:center; gap:6px; }
ol.events .k::before { content:""; width:8px; height:8px; border-radius:50%; background:var(--k-other); flex:none; }
ol.events .k-message .k::before { background:var(--k-message); } ol.events .k-tool .k::before { background:var(--k-tool); }
ol.events .k-memory .k::before { background:var(--k-memory); } ol.events .k-learn .k::before { background:var(--k-learn); }
ol.events li.failed { background:var(--bad-soft); } ol.events li.failed .k::before { background:var(--bad); } ol.events li.failed .k { color:var(--bad); }
ol.events .what { overflow-wrap:anywhere; } ol.events .what .who { color:var(--muted); }
details.disclose > summary { cursor:pointer; color:var(--accent); font-size:13px; list-style:none; display:inline-block; padding:2px 0; }
details.disclose > summary::-webkit-details-marker { display:none; } details.disclose > summary::before { content:"\\25B8"; display:inline-block; margin-right:6px; transition:transform .1s; }
details.disclose[open] > summary::before { transform:rotate(90deg); }
.trace { margin:6px 0 2px; } .trace .wrap { box-shadow:none; }
.diff { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:10px 0; } .diff > div { border-radius:8px; padding:8px 12px; border:1px solid var(--line); overflow-wrap:anywhere; }
.diff .old { background:var(--bad-soft); } .diff .new { background:var(--ok-soft); } .diff small { display:block; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; font-size:11px; }
.diff .old b { text-decoration:line-through; text-decoration-thickness:1px; }
.actions { display:flex; gap:10px 16px; flex-wrap:wrap; align-items:flex-start; margin-top:12px; }
.actions form { display:flex; gap:8px; flex-wrap:wrap; align-items:center; } .actions details form { margin-top:8px; }
.actions details { background:var(--sunken); border-radius:8px; padding:6px 12px; }
dl.facts { display:grid; grid-template-columns:minmax(120px, 28%) 1fr; margin:0; } dl.facts dt, dl.facts dd { margin:0; padding:9px 14px; border-top:1px solid var(--line); overflow-wrap:anywhere; }
dl.facts dt { color:var(--ink-2); font-weight:550; } dl.facts dt:first-of-type, dl.facts dt:first-of-type + dd { border-top:0; }
dl.facts dd small { display:block; color:var(--muted); } ul.plain { margin:0; padding-left:20px; }
.card.done { border-color:var(--ok); background:var(--ok-soft); }
@media (max-width:640px) { ol.events li { grid-template-columns:1fr; gap:0; } .diff { grid-template-columns:1fr; } dl.facts { grid-template-columns:1fr; }
  dl.facts dd { border-top:0; padding-top:0; } nav.main a { padding:12px 8px 10px; } h1 { font-size:21px; } }
@media print { header.top, .filters, .actions, .search { display:none; } body { background:#fff; } }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _q(value: Any) -> str:
    return quote(str(value), safe="")


def ago(ts: Optional[float], now: Optional[float] = None) -> str:
    """`5 min ago`, `3 h ago`, `yesterday`, or the date."""
    if not ts:
        return "-"
    seconds = (time.time() if now is None else now) - ts
    if seconds < 45:
        return "just now"
    if seconds < 3600:
        return f"{round(seconds / 60)} min ago"
    if seconds < 86400:
        return f"{round(seconds / 3600)} h ago"
    if seconds < 172800:
        return "yesterday"
    if seconds < 30 * 86400:
        return f"{int(seconds // 86400)} days ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _when(ts: Optional[float], text: str) -> str:
    """Relative time, with the exact one on hover."""
    return f"<span title='{_e(text)}'>{_e(ago(ts))}</span>" if ts else _e(text)


def _page(title: str, body: str, *, mode: str = "", active: str = "", crumbs: tuple[tuple[str, str], ...] = (),
          waiting: Optional[int] = None) -> str:
    """`mode` ("review" or "read-only") is shown in the header when given; `waiting` is the review count."""
    def link(path: str, label: str, key: str, extra: str = "") -> str:
        current = " aria-current=page" if active == key else ""
        return f"<a href='{path}'{current}>{label}{extra}</a>"

    badge = f"<span class=badge title='waiting for review'>{waiting}</span>" if waiting else ""
    pill = f"<span class='mode {'review' if mode.startswith('review') else ''}'>{_e(mode)}</span>" if mode else ""
    trail = ""
    if crumbs:
        parts = [f"<a href='{_e(href)}'>{_e(label)}</a>" for label, href in crumbs]
        trail = f"<p class=crumbs>{' / '.join(parts)}</p>"
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width, initial-scale=1'><title>{_e(title)} · CommonMemo</title>"
            f"<style>{STYLE}</style></head><body><a class=skip href='#main'>Skip to content</a>"
            f"<header class=top><div><a class=brand href='/'>CommonMemo</a><nav class=main aria-label=Main>"
            f"{link('/', 'Overview', 'overview')}{link('/review', 'Review', 'review', badge)}</nav>{pill}</div></header>"
            f"<main id=main>{trail}{body}</main></body></html>")


def _table(headers: list[str], rows: list[list[str]], empty: str = "", hint: str = "", nowrap: tuple[int, ...] = (),
           num: tuple[int, ...] = ()) -> str:
    """A table; with no rows, an empty state: what is missing, and (`hint`) how it gets there."""
    if not rows:
        return _empty(empty or "Nothing yet", hint)

    def cls(i: int) -> str:
        names = [n for n, cols in (("nowrap", nowrap), ("num", num)) if i in cols]
        return f" class='{' '.join(names)}'" if names else ""

    head = "".join(f"<th scope=col{cls(i)}>{_e(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(row)) + "</tr>" for row in rows)
    return f"<div class=wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _empty(what: str, hint: str = "") -> str:
    return f"<div class=empty><b>{_e(what)}</b>{_e(hint)}</div>"


def _tile(number: Any, label: str, href: str = "", hot: bool = False) -> str:
    cls = "tile hot" if hot else "tile"
    inner = f"<b>{_e(number)}</b><span>{_e(label)}</span>"
    return f"<a class='{cls}' href='{_e(href)}'>{inner}</a>" if href else f"<div class='{cls}'>{inner}</div>"


def waiting_count(mem: "Memory") -> int:
    return len(mem.pending_world_facts()) + len(mem.pending_procedures())


def _user_link(user_id: Optional[str]) -> str:
    return f"<a href='/user/{_q(user_id)}'>{_e(user_id)}</a>" if user_id else "<span class=muted>-</span>"


# -- overview -----------------------------------------------------------------


def overview_page(mem: Memory, query: str = "", mode: str = "") -> str:
    v = overview(mem)
    waiting = len(v["pending"]) + v["pending_procedures"]
    needle = query.strip().lower()

    def hit(*values: Any) -> bool:
        return not needle or any(needle in str(x).lower() for x in values if x)

    sessions = [s for s in v["sessions"] if hit(s["session_id"], s["user_id"], s["agent"])]
    users = [u for u in v["users"] if hit(u)]
    world = [w for w in v["world"] if hit(w["key"], w["value"], w["scope"], w["source"])]
    per_user: dict[str, list[dict[str, Any]]] = {}
    for s in v["sessions"]:
        if s["user_id"]:
            per_user.setdefault(s["user_id"], []).append(s)
    banner = ""
    if waiting:
        banner = (f"<div class=banner role=status><div class=grow><b>{waiting} item{'s' if waiting != 1 else ''} "
                  f"waiting for review.</b> <span class=muted>Facts and procedures agents proposed; nothing is used until approved.</span></div>"
                  f"<a class='btn primary' href='/review'>Review {waiting}</a></div>")
    tiles = (_tile(len(v["sessions"]), "recent sessions") + _tile(len(v["users"]), "users")
             + _tile(len(v["world"]), "world facts") + _tile(waiting, "waiting for review", "/review", hot=bool(waiting)))
    clear = "<a class=btn href='/'>Clear</a>" if query else ""
    search = (f"<form class=search role=search action='/' method=get><input type=search name=q value='{_e(query)}' "
              f"placeholder='Filter sessions, users and facts' aria-label='Filter sessions, users and facts'>"
              f"<button class=primary>Filter</button>{clear}</form>")
    cards = "".join(
        f"<div class=card><h3>{_user_link(u)}</h3><div class=sub>{len(per_user.get(u, []))} session"
        f"{'' if len(per_user.get(u, [])) == 1 else 's'}"
        f"{' · last ' + ago(per_user[u][0]['ts']) if per_user.get(u) else ''}</div></div>" for u in users)
    srows = [[f"<a href='/session/{_q(s['session_id'])}'>{_e(s['session_id'])}</a>", _user_link(s["user_id"]),
              _e(s["agent"] or "-"), _e(s["events"]), _when(s["ts"], s["last"])] for s in sessions]
    wrows = [[f"<b>{_e(w['key'])}</b>", _e(w["value"]), f"<span class=chip>{_e(w['scope'])}</span>",
              f"<span class=chip>{_e(w['source'])}</span>"] for w in world]
    prows = [[f"<b>{_e(w['key'])}</b>", _e(w["value"]), f"<span class=chip>{_e(w['scope'])}</span>", _user_link(w["user_id"])]
             for w in v["pending"]]
    note = f"<p class='muted small'>Showing what matches “{_e(query)}”.</p>" if query else ""
    return _page("Overview", f"<h1>Overview</h1><p class=lede>What memory holds, and what it needs from you.</p>{banner}"
                 f"<div class=tiles>{tiles}</div>{search}{note}"
                 f"<h2>Users <span class=count>{len(users)}</span></h2>"
                 + (f"<div class=cards>{cards}</div>" if cards else _empty("No users yet", "A user appears once a session is opened with a user id."))
                 + f"<h2>Recent sessions <span class=count>{len(sessions)}</span></h2>"
                 + _table(["Session", "User", "Agent", "Events", "Last event"], srows, "No sessions" + (" match" if query else " yet"),
                          "" if query else "Open one with memory.session(...), or run an agent with the hooks.", nowrap=(4,), num=(3,))
                 + f"<h2>World facts <span class=count>{len(world)}</span></h2>"
                 + _table(["Key", "Value", "Scope", "Source"], wrows, "No world facts" + (" match" if query else " yet"),
                          "" if query else "Facts about the project or the world: mem.learn(key, value).")
                 + f"<h2>Waiting for review <span class=count>{len(prows)}</span></h2>"
                 + _table(["Key", "Proposed value", "Scope", "Proposed by"], prows, "Nothing is waiting", "Agent-written facts show up here first.")
                 + ("<p><a href='/review'>Open the review queue →</a></p>" if waiting else ""),
                 mode=mode, active="overview", waiting=waiting)


# -- review -------------------------------------------------------------------


def _form(token: str, action: str, label: str, fields: str = "", cls: str = "") -> str:
    return (f"<form method=post action='{_e(action)}'><input type=hidden name=token value='{_e(token)}'>"
            f"{fields}<button class='{cls}'>{_e(label)}</button></form>")


def _disclose(summary: str, inner: str) -> str:
    return f"<details class=disclose><summary>{_e(summary)}</summary>{inner}</details>"


def review_page(mem: Memory, token: Optional[str], done: str = "", mode: str = "") -> str:
    """The review queue; with a `token` (review mode), with its actions."""
    q = review_queue(mem)
    cards = []
    for w in q["world"]:
        if w["current"] is not None:
            diff = (f"<div class=diff><div class=old><small>Now</small><b>{_e(w['current'])}</b></div>"
                    f"<div class=new><small>Proposed</small><b>{_e(w['value'])}</b></div></div>"
                    f"<p class='muted small'>replaces <b>{_e(w['current'])}</b></p>")
        else:
            diff = (f"<div class=diff><div class=new><small>Proposed (new key)</small><b>{_e(w['value'])}</b></div></div>"
                    f"<p class='muted small'>new key</p>")
        head = (f"<h3>{_e(w['key'])} <span class='chip warn'>{_e(w['status'])}</span>"
                f"<span class=chip>{_e(w['scope'])}</span><span class=chip>source {_e(w['source'])}</span></h3>"
                f"<div class=sub>proposed by {_e(w['proposed_by'] or '-')} · {_when(w['ts'], w['since'])}"
                f"{' · ' + _e(w['note']) if w['note'] else ''}</div>")
        actions = ""
        if token:
            base = f"/review/world/{_q(w['id'])}"
            actions = ("<div class=actions>" + _form(token, f"{base}/confirm", "Approve", cls="primary")
                       + _disclose("Approve with a different value", _form(
                           token, f"{base}/confirm", "Approve this value",
                           "<input type=text name=value required placeholder='corrected value' aria-label='corrected value'>"))
                       + _disclose("Reject…", _form(
                           token, f"{base}/reject", "Reject",
                           "<input type=text name=reason placeholder='reason (optional)' aria-label=reason>", cls="danger"))
                       + "</div>")
        cards.append(f"<div class=card>{head}{diff}{actions}</div>")
    procs = []
    for p in q["procedures"]:
        now = (f"<details class=disclose><summary>current version v{_e(p['current_version'])}</summary><pre>{_e(p['current'])}</pre></details>"
               if p["current"] is not None else "<p class='muted small'>new entry</p>")
        head = (f"<h3><b>{_e(p['task'])} · {_e(p['kind'])} · {_e(p['name'])}</b> <span class='chip warn'>{_e(p['status'])}</span>"
                f"<span class=chip>{_e(p['scope'])}</span><span class=chip>source {_e(p['source'])}</span></h3>"
                f"<div class=sub>proposed by {_e(p['proposed_by'] or '-')} · {_when(p['ts'], p['since'])}"
                f"{' · ' + _e(p['note']) if p['note'] else ''}"
                f"{' · use when: ' + _e(p['use_when']) if p['use_when'] else ''}</div>"
                f"<pre>{_e(p['body'])}</pre>{now}")
        actions = ""
        if token:
            base = f"/review/procedure/{_q(p['id'])}"
            own = p["scope"].startswith("user:")
            actions = "<div class=actions>" + _form(token, f"{base}/confirm", "Approve for this user" if own else "Approve", cls="primary")
            if own and p["kind"] == "exemplar":
                actions += _disclose("Share with everyone…", _form(
                    token, f"{base}/share", "Share with everyone",
                    "<label><input type=checkbox name=scrubbed value=1 required> personal data checked</label>"))
            actions += _disclose("Reject…", _form(
                token, f"{base}/reject", "Reject",
                "<input type=text name=reason placeholder='reason (optional)' aria-label=reason>", cls="danger")) + "</div>"
        procs.append(f"<div class=card>{head}{actions}</div>")
    note = "" if token else ("<div class=banner><div class=grow>Read-only. Start with <code>commonmemo inspect --review</code> "
                             "to approve or reject.</div></div>")
    flash = f"<div class='card done' role=status>✓ {_e(done)}</div>" if done else ""
    total = len(cards) + len(procs)
    allclear = ("<div class='banner ok'><div class=grow><b>All clear.</b> Nothing is waiting for review.</div></div>" if not total else "")
    return _page("Review", f"<h1>Waiting for review</h1><p class=lede>{total} item{'s' if total != 1 else ''}. "
                 f"Agents propose; nothing here is used until it is approved.</p>{flash}{note}{allclear}"
                 f"<h2>World facts <span class=count>{len(cards)}</span></h2>"
                 f"{''.join(cards) or _empty('No world facts waiting')}"
                 f"<h2>Procedures <span class=count>{len(procs)}</span></h2>{''.join(procs) or _empty('No procedure entries waiting')}",
                 mode=mode, active="review", waiting=total, crumbs=(("Overview", "/"),))


# -- user ---------------------------------------------------------------------


def user_page(mem: Memory, user_id: str, mode: str = "") -> str:
    v = user_view(mem, user_id)
    facts = "".join(f"<dt>{_e(f['key'])}</dt><dd>{_e(f['value'])}<small>since {_e(f['since'])} · "
                    f"{len(f['events'])} source event{'' if len(f['events']) == 1 else 's'}</small></dd>" for f in v["facts"])
    negatives = [[_e(n["statement"]), _e(n["reason"] or ""), f"<span class=chip>{_e(n['source'])}</span>"] for n in v["negatives"]]
    forgotten = [[_e(f["key"]), _e(f["when"])] for f in v["forgotten"]]
    recaps = [[_e(r["ended"]), _e(r["agent"] or "-"),
               f"<a href='/session/{_q(r['session'])}'>{_e(r['session'])}</a>", _e(r["summary"])] for r in v["recaps"]]
    sessions = " ".join(f"<a class=chip href='/session/{_q(s)}'>{_e(s)}</a>" for s in v["sessions"]) or "<span class=muted>none</span>"
    feedback = "".join(f"<li>{_e(line)}</li>" for line in v["feedback"]["recent"])
    tiles = (_tile(len(v["facts"]), "profile facts") + _tile(len(v["sessions"]), "sessions")
             + _tile(len(v["negatives"]), "standing rules") + _tile(len(v["forgotten"]), "forgotten")
             + _tile(v["feedback"]["count"], "feedback"))
    return _page(f"User {user_id}", f"<h1>{_e(user_id)}</h1><p class=lede>Everything memory holds about this user.</p>"
                 f"<div class=tiles>{tiles}</div><h2>Sessions</h2><p>{sessions}</p>"
                 f"<h2>Profile <span class=count>{len(v['facts'])}</span></h2>"
                 + (f"<div class=wrap><dl class=facts>{facts}</dl></div>" if facts else
                    _empty("No profile facts", "They come from s.remember(key, value), or from extraction when it is turned on."))
                 + f"<h2>Never <span class=count>{len(negatives)}</span></h2>"
                 + _table(["Statement", "Reason", "Source"], negatives, "No standing rules", "s.never('…') adds one.")
                 + f"<h2>Forgotten <span class=count>{len(forgotten)}</span></h2>"
                 + _table(["Key", "When"], forgotten, "Nothing forgotten", "s.forget(key) hides a fact and keeps it from being re-learned.")
                 + f"<h2>Past sessions <span class=count>{len(recaps)}</span></h2>"
                 + _table(["Ended", "Agent", "Session", "Recap"], recaps, "No recaps yet", "A recap is written when a session closes.", nowrap=(0,))
                 + f"<h2>Feedback <span class=count>{_e(v['feedback']['count'])}</span></h2>"
                 + (f"<div class=card><ul class=plain>{feedback}</ul></div>" if feedback else _empty("No feedback yet")),
                 mode=mode, waiting=waiting_count(mem), crumbs=(("Overview", "/"),))


# -- session ------------------------------------------------------------------

GROUPS = (("messages", "Messages"), ("tools", "Tool calls"), ("memory", "Memory pushes"), ("learning", "Learning"))


def _group(kind: str) -> str:
    if kind == "message":
        return "messages"
    if kind == "tool_invoked":
        return "tools"
    if kind in ("context_pushed", "step_started", "step_finished", "step_failed", "checkpoint"):
        return "memory"
    if kind in ("user_fact", "negative", "world_fact", "feedback", "lesson_learned", "decision_recorded", "forgotten") \
            or kind.startswith("procedure"):
        return "learning"
    return "other"


_KIND_CLASS = {"messages": "k-message", "tools": "k-tool", "memory": "k-memory", "learning": "k-learn", "other": ""}
_KIND_LABEL = {"message": "message", "tool_invoked": "tool", "context_pushed": "memory", "user_fact": "fact",
               "negative": "rule", "world_fact": "world fact", "feedback": "feedback", "lesson_learned": "lesson",
               "checkpoint": "checkpoint", "forgotten": "forgot", "decision_recorded": "decision",
               "procedure_entry": "procedure", "procedure_confirmed": "procedure", "procedure_followed": "procedure"}


def _trace_card(trace: TurnTrace, labels: dict[str, str], session_id: str) -> str:
    rows = []
    for i in trace.items:
        key = i.get("key") or i.get("id")
        shown = i.get("key") or labels.get(i.get("id") or "") or i.get("id")
        detail = " ".join(f"<span class=chip>{_e(k)} {_e(i[k])}</span>"
                          for k in ("scope", "source", "status", "score", "session") if i.get(k) is not None)
        why = f"<a href='{_e(f'/why?session={_q(session_id)}&key={_q(key)}')}'>why</a>"
        rows.append([f"<span class='chip accent'>{_e(i['kind'])}</span>", _e(shown), detail, why])
    cut = ", ".join(f"{n} {k}" for k, n in trace.more.items() if n > 0)
    text = f"<details class=disclose><summary>pushed text</summary><pre>{_e(trace.text)}</pre></details>" if trace.text else ""
    left = f"<p class=small><span class='chip warn'>left out</span> {_e(cut)}</p>" if cut else ""
    return (f"<div class=trace><b>{_e(trace.reason)} push</b> <span class=muted>{_e(trace.chars)} chars"
            f"{', by ' + _e(trace.agent) if trace.agent else ''}</span>"
            f"{_table(['Kind', 'Item', 'From', ''], rows, 'Nothing was pushed', nowrap=(0, 3))}{left}{text}</div>")


def session_page(mem: Memory, session_id: str, show: str = "", mode: str = "") -> Optional[str]:
    v = session_view(mem, session_id)
    if v is None:
        return None
    wanted = {g for g in show.split(",") if g in dict(GROUPS)}
    traces = {t.event_id: t for t in v["traces"]}
    t0 = v["timeline"][0]["ts"] if v["timeline"] else 0.0
    turns: list[dict[str, Any]] = [{"title": "Start", "said": "", "events": []}]
    for ev in v["timeline"]:
        if ev["kind"] == "message" and ev.get("role") == "user":
            turns.append({"title": f"Turn {len(turns)}", "said": ev["line"].removeprefix("user: "), "events": []})
        turns[-1]["events"].append(ev)
    if not turns[0]["events"]:
        turns.pop(0)
    tools = [e for e in v["timeline"] if e["kind"] == "tool_invoked"]
    failed = [e for e in tools if e["failed"]]
    pushes = [e for e in v["timeline"] if e["kind"] == "context_pushed"]
    user = _user_link(v["user_id"])
    status = f"<span class='chip ok'>closed {_e(v['closed'])}</span>" if v["closed"] else "<span class='chip accent'>open</span>"
    base = f"/session/{_q(session_id)}"
    chips = [f"<a href='{base}'{' aria-current=true' if not wanted else ''}>All</a>"]
    for key, label in GROUPS:
        on = key in wanted
        nxt = sorted(wanted ^ {key})
        href = base + (f"?show={','.join(nxt)}" if nxt else "")
        chips.append(f"<a href='{href}'{' aria-current=true' if on else ''}>{_e(label)}</a>")
    rendered = []
    for turn in turns:
        rows = []
        for ev in turn["events"]:
            group = _group(ev["kind"])
            if wanted and group not in wanted:
                continue
            cls = " ".join(x for x in (_KIND_CLASS[group], "failed" if ev["failed"] else "") if x)
            label = "error" if ev["failed"] else _KIND_LABEL.get(ev["kind"], ev["kind"].replace("_", " "))
            extra = ""
            if ev["id"] in traces:
                extra = f"<details class=disclose><summary>what was pushed</summary>{_trace_card(traces[ev['id']], v['labels'], session_id)}</details>"
            offset = f"+{max(0, ev['ts'] - t0):.0f}s" if ev["ts"] else ""
            who = f" <span class=who>· {_e(ev['agent'])}</span>" if ev["agent"] and ev["agent"] != v["agent"] else ""
            rows.append(f"<li class='{cls}' title='{_e(ev['when'])}'><span class=t>{_e(offset)}</span><span class=k>{_e(label)}</span>"
                        f"<span class=what>{_e(ev['line'])}{who}{extra}</span></li>")
        if not rows:
            continue
        said = f"<span class=said>{_e(turn['said'])}</span>" if turn["said"] else "<span class=said muted>before the first message</span>"
        rendered.append(f"<section class=turn><header><b>{_e(turn['title'])}</b>{said}"
                        f"<span class='muted small'>{len(rows)} event{'' if len(rows) == 1 else 's'}</span></header>"
                        f"<ol class=events>{''.join(rows)}</ol></section>")
    resume = ""
    if v["resume"]:
        resume = (f"<h2>A resume would start from</h2><div class=card><ul class=plain>"
                  f"{''.join(f'<li>{_e(line)}</li>' for line in v['resume'])}</ul></div>")
    tiles = (_tile(len(v["timeline"]), "events") + _tile(sum(1 for t in turns if t["said"]), "turns")
             + _tile(len(pushes), "memory pushes") + _tile(f"{len(tools) - len(failed)}/{len(tools)}", "tool calls ok"))
    timeline = "".join(rendered) or _empty("No events" + (" in this view" if wanted else ""),
                                          "Clear the filter to see them all." if wanted else "")
    return _page(f"Session {session_id}", f"<h1>{_e(session_id)}</h1>"
                 f"<div class=meta><span>user {user}</span><span>agent {_e(v['agent'] or '-')}</span><span>opened {_e(v['opened'])}</span>{status}</div>"
                 f"<div class=tiles>{tiles}</div>{resume}<h2>Timeline <span class=count>by turn</span></h2>"
                 f"<div class=filters aria-label='Filter events'>{''.join(chips)}</div>{timeline}",
                 mode=mode, waiting=waiting_count(mem), crumbs=(("Overview", "/"),))


def why_page(mem: Memory, session_id: str, key: str, mode: str = "") -> Optional[str]:
    found = mem.why(key, session_id=session_id)
    if found is None:
        return None
    rows = [[f"<span class='chip accent'>{_e(e.kind)}</span>",
             f"<a href='/session/{_q(e.session_id)}'>{_e(e.session_id)}</a>" if e.session_id else "-",
             _e(getattr(e, "source", "") or ""), f"<code>{_e(e.id)}</code>"] for e in found.sources]
    return _page(f"Why {key}", f"<h1>Why the agent got {_e(key)}</h1>"
                 f"<p class=lede>A {_e(found.item['kind'])}, pushed in a {_e(found.trace.reason)} push to session "
                 f"<a href='/session/{_q(session_id)}'>{_e(session_id)}</a>.</p>"
                 f"<h2>It came from <span class=count>{len(rows)}</span></h2>"
                 f"{_table(['Event', 'Session', 'Source', 'Id'], rows, 'Its source events are no longer in the Log', 'They were pruned or forgotten.')}",
                 mode=mode, waiting=waiting_count(mem), crumbs=(("Overview", "/"), (f"Session {session_id}", f"/session/{_q(session_id)}")))
