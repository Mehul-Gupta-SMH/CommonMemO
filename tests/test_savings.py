"""commonmemo savings: pushes against re-sending the history (#158)."""

from __future__ import annotations

import asyncio

from commonmemo import Memory
from commonmemo.cli import main


def run(coro):
    return asyncio.run(coro)


def _two_harness_session(mem):
    """One session continued across harnesses: Claude Code first, Codex second."""
    mem.learn("refund_window", "30 days")
    with mem.session("earlier", user_id="ada", agent="claude-code") as s:
        s.remember("preferred_contact", "email")
    with mem.session("work", user_id="ada", agent="claude-code") as s:
        s.message("user", "a" * 400)
        s.message("assistant", "b" * 600)
        run(s.push_turn("what is the refund window and how will you contact me?"))
    with mem.session("work", user_id="ada", agent="codex") as s:
        s.message("user", "c" * 200)
        run(s.push_turn("and the refund window and contact again?"))


def test_numbers_match_the_log_and_split_by_harness():
    mem = Memory()
    _two_harness_session(mem)
    pushes = mem.traces(session_id="work")
    report = mem.savings(session_id="work")
    assert report.total.pushes == len(pushes)
    assert report.total.pushed == sum(t.chars for t in pushes)
    agents = {r.name: r for r in report.by_agent}
    assert set(agents) == {"claude-code", "codex"}
    # the turn push after 1,000 chars of messages is measured against exactly those
    turns = [t for t in pushes if t.reason == "turn"]
    assert len(turns) == 2
    assert agents["codex"].pushed > 0
    assert agents["codex"].resend >= 1200  # codex was given the push, not the 1,200 chars said so far
    assert agents["codex"].pushed < agents["codex"].resend
    assert report.total.saved_pct is not None and report.total.saved_pct > 0
    run(mem.aclose())


def test_nothing_pushed_is_zero_not_an_error():
    mem = Memory()
    report = mem.savings()
    assert report.total.pushes == 0 and report.total.ratio is None
    assert report.lines() == ["no pushes recorded: nothing to compare"]
    run(mem.aclose())


def test_other_sessions_are_not_counted():
    mem = Memory()
    _two_harness_session(mem)
    with mem.session("other", user_id="bob", agent="claude-code") as s:
        s.message("user", "z" * 5000)
        run(s.push_turn("hello"))
    work = mem.savings(session_id="work")
    ada = mem.savings(user_id="ada")
    assert work.total.resend == ada.total.resend
    assert mem.savings().total.resend > ada.total.resend
    run(mem.aclose())


def test_price_gives_dollars_and_the_limits_are_printed():
    mem = Memory()
    _two_harness_session(mem)
    text = "\n".join(mem.savings(session_id="work").lines(price_per_mtok=3.0))
    assert "per million tokens" in text and "Estimate" in text and "answer quality" in text
    run(mem.aclose())


def test_cli(tmp_path, capsys):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    mem = Memory(url)
    _two_harness_session(mem)
    run(mem.aclose())
    assert main(["savings", "work", "--store", url]) == 0
    out = capsys.readouterr().out
    assert "by harness:" in out and "claude-code" in out and "codex" in out and "smaller" in out
