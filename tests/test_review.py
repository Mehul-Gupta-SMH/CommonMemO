"""The inspector's review screen (#45)."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from commonmemo import Memory
from commonmemo.agent_tools import build_agent_tools
from commonmemo.inspector import serve

ATTACK = "<script>alert(1)</script>"


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    mem = Memory(url, project="acme")
    mem.learn("refund_window", "30 days")
    with mem.session("s1", user_id="mallory", agent="support") as s:
        s.remember("manager", "Grace Hopper")
        tools = {t.__name__: t for t in build_agent_tools(s)}
        tools["memory_note_fact"]("refund_window", f"{ATTACK} 90 days")
        tools["memory_save_procedure"]("weekly_report", "rule", "emoji", "No emoji.", for_everyone=True)
    with mem.session("s2", user_id="mallory") as s:
        mem.save_procedure("weekly_report", "exemplar", "candidate", "Shipped v2. Grace Hopper approved.",
                           scope="user:mallory", source="extracted")
    asyncio.run(mem.aclose())
    return url


def start(url, **kw):
    mem = Memory(url, project="acme")
    server = serve(mem, port=0, **kw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return mem, server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def review(db):
    mem, server, base = start(db, review=True, reviewer="ada")
    yield mem, base
    server.shutdown()
    server.server_close()
    asyncio.run(mem.aclose())


def call(url, data=None, headers=None):
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def token(base):
    return re.search(r"name=token value='([^']+)'", call(base + "/review")[1]).group(1)


def pending_id(mem, kind):
    mem.knowledge.catch_up(mem.store)
    if kind == "world":
        return mem.pending_world_facts()[0].event_id
    return {p.kind: p.event_id for p in mem.pending_procedures()}[kind]


def test_read_only_by_default(db):
    mem, server, base = start(db)
    try:
        status, body = call(base + "/review")
        assert status == 200 and "No emoji." in body and "name=token" not in body and "--review" in body
        assert call(base + "/review/world/x/confirm", {"token": "t"})[0] == 405
    finally:
        server.shutdown()
        server.server_close()
        asyncio.run(mem.aclose())


def test_the_queue_shows_what_each_item_would_replace(review):
    mem, base = review
    status, body = call(base + "/review")
    assert status == 200 and ATTACK not in body and "&lt;script&gt;" in body
    assert "replaces <b>30 days</b>" in body and "weekly_report · rule · emoji" in body
    assert "Share with everyone" in body and "personal data checked" in body
    data = json.loads(call(base + "/api/review")[1])
    assert data["world"][0]["current"] == "30 days" and {p["kind"] for p in data["procedures"]} == {"rule", "exemplar"}


def test_approve_with_a_correction_and_reject(review):
    mem, base = review
    fact = pending_id(mem, "world")
    status, body = call(f"{base}/review/world/{fact}/confirm", {"token": token(base), "value": "45 days"})
    assert status == 200 and "Approved the world fact." in body  # after the redirect
    mem.knowledge.catch_up(mem.store)
    assert {w.key: w.value for w in mem.world_facts()}["refund_window"] == "45 days"
    [approval] = mem.store.events(kinds=["world_fact_confirmed"])
    assert approval.run_id == "review:ada"
    rule = pending_id(mem, "rule")
    call(f"{base}/review/procedure/{rule}/reject", {"token": token(base), "reason": "style"})
    mem.knowledge.catch_up(mem.store)
    assert mem.knowledge.procedure_entry(rule).status == "rejected"


def test_sharing_an_exemplar_needs_the_personal_data_check(review):
    mem, base = review
    ex = pending_id(mem, "exemplar")
    status, body = call(f"{base}/review/procedure/{ex}/share", {"token": token(base)})
    assert status == 400 and "scrubbed=True" in body
    call(f"{base}/review/procedure/{ex}/share", {"token": token(base), "scrubbed": "1"})
    mem.knowledge.catch_up(mem.store)
    [shared] = mem.procedures("weekly_report", user_id="bob").of("exemplar")
    assert shared.body == "Shipped v2. [redacted] approved."


def test_forged_and_cross_site_posts_are_refused(review):
    mem, base = review
    fact = pending_id(mem, "world")
    target = f"{base}/review/world/{fact}/confirm"
    good = token(base)
    assert call(target, {"value": "1 day"})[0] == 403                                    # no token
    assert call(target, {"token": "guess"})[0] == 403                                    # wrong token
    assert call(target, {"token": good}, {"Origin": "https://evil.example"})[0] == 403   # another site
    assert call(target, {"token": good}, {"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert call(target, {"token": good}, {"Host": "evil.example"})[0] == 403             # a rebound DNS name
    assert call(base + "/", headers={"Host": "evil.example"})[0] == 403                  # reads too
    mem.knowledge.catch_up(mem.store)
    assert mem.pending_world_facts() and mem.store.events(kinds=["world_fact_confirmed"]) == []


def test_a_review_session_can_be_reverted(review):
    mem, base = review
    call(f"{base}/review/world/{pending_id(mem, 'world')}/confirm", {"token": token(base)})
    call(f"{base}/review/procedure/{pending_id(mem, 'rule')}/confirm", {"token": token(base)})
    mem.knowledge.catch_up(mem.store)
    mem.revert(run_id="review:ada")
    assert {w.key: w.value for w in mem.world_facts()}["refund_window"] == "30 days"
    assert "emoji" not in {e.name for e in mem.procedures("weekly_report").entries}
