"""Dream: batch extraction over closed sessions (#159). All offline, with a fake model."""

from __future__ import annotations

import asyncio
import json

import pytest

from commonmemo import Memory
from commonmemo.events import Checkpoint, UserFact
from commonmemo.dream import PREFIX


def run(coro):
    return asyncio.run(coro)


def reply(*facts):
    return json.dumps({"facts": [{"key": k, "value": v} for k, v in facts]})


class Model:
    """Answers from a script, records prompts, and carries a running `cost` like the CLI's wrapper."""

    def __init__(self, *replies, per_call=0.0):
        self.replies, self.prompts, self.per_call = list(replies), [], per_call
        if per_call:  # only a wrapper that bills reports a running cost
            self.cost = 0.0

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.per_call:
            self.cost += self.per_call
        return self.replies.pop(0) if self.replies else reply()


def mem():
    return Memory()  # no extractor: sessions close without extracting, like a harness that set none up


def talk(m, sid, user="ada", text="My sister Lena visits on Friday.", close=True):
    s = m.session(sid, user_id=user, agent="claude-code")
    s.message("user", text)
    if close:
        run(s.aclose())
    return s


def profile(m, user="ada"):
    return {f.key: f.value for f in m._visible_facts(user, run(m.engine.user_profile(user)))}


def facts_of(m):
    return m.store.events(kinds=[UserFact.kind])


def test_a_dry_run_asks_the_model_nothing_and_writes_nothing():
    m, model = mem(), Model(reply(("sister", "Lena visits Friday.")))
    talk(m, "s1")
    before = m.store.position()
    report = run(m.dream(complete=model))
    assert not report.applied and [p.session_id for p in report.planned] == ["s1"]
    assert model.prompts == [] and m.store.position() == before and profile(m) == {}
    assert report.estimated > 0 and "dry run" in "\n".join(report.lines())


def test_apply_writes_under_one_dream_run_id_and_the_profile_follows():
    m = mem()
    talk(m, "s1")
    talk(m, "s2", user="bob", text="I moved to Oslo.")
    model = Model(reply(("sister", "Lena visits Friday.")), reply(("home_city", "Bob lives in Oslo.")))
    report = run(m.dream(complete=model, apply=True))
    assert report.run_id.startswith(PREFIX) and report.count == 2
    assert {e.run_id for e in facts_of(m)} == {report.run_id}
    assert {e.source for e in facts_of(m)} == {"extracted"}
    assert profile(m) == {"sister": "Lena visits Friday."} and profile(m, "bob") == {"home_city": "Bob lives in Oslo."}


def test_a_rerun_does_only_new_work_and_each_session_is_read_once():
    m, model = mem(), Model(reply(("sister", "Lena visits Friday.")), reply(("trip", "Ada flies out Monday.")))
    talk(m, "s1")
    first = run(m.dream(complete=model, apply=True))
    again = run(m.dream(complete=model, apply=True))
    assert len(first.planned) == 1 and again.planned == [] and len(model.prompts) == 1
    talk(m, "s2", text="I fly out on Monday.")
    third = run(m.dream(complete=model, apply=True))
    assert [p.session_id for p in third.planned] == ["s2"] and len(model.prompts) == 2
    assert any(e.label == "dream" for e in m.store.events(kinds=[Checkpoint.kind]))


def test_a_session_already_extracted_at_close_is_not_read_again():
    from commonmemo import ModelExtractor

    m = Memory(extractor=ModelExtractor(Model(reply(("sister", "Lena visits Friday.")))))
    talk(m, "s1")
    model = Model()
    report = run(m.dream(complete=model, apply=True))
    assert report.planned == [] and model.prompts == [] and report.skipped["already dreamt or extracted"] == 1


def test_open_sessions_and_sessions_without_a_user_are_left_alone():
    m, model = mem(), Model(reply(("sister", "Lena visits Friday.")))
    talk(m, "open", close=False)
    s = m.session("anon", agent="claude-code")
    s.message("user", "hello")
    run(s.aclose())
    report = run(m.dream(complete=model, apply=True))
    assert report.planned == [] and model.prompts == []
    assert report.skipped == {"open": 1, "no user": 1}


def test_the_budget_stops_the_run_and_says_where_it_stopped():
    m = mem()
    for i in range(4):
        talk(m, f"s{i}", text=f"fact number {i} about the trip " * 20)
    model = Model(*[reply((f"k{i}", f"v{i}")) for i in range(4)])
    price = dict(price_in=1000.0, price_out=0.0)  # input only, so the estimate and the metered figure agree
    full = run(m.dream(complete=model, **price))
    one = full.planned[0].cost
    report = run(m.dream(complete=model, apply=True, budget=one * 2.5, **price))
    assert [p.session_id for p in report.planned] == ["s0", "s1"] and len(model.prompts) == 2
    assert report.stopped_before.session_id == "s2" and report.position == report.planned[-1].position
    text = "\n".join(report.lines())
    assert "STOPPED by the budget before s2" in text and f"--since {report.stopped_before.position}" in text
    rest = run(m.dream(complete=model, apply=True, **price))
    assert [p.session_id for p in rest.planned] == ["s2", "s3"], "the next pass picks up where the cap stopped it"


def test_a_cost_the_model_function_reports_is_what_the_cap_counts():
    m = mem()
    for i in range(3):
        talk(m, f"s{i}")
    model = Model(per_call=0.30)  # the wrapper says each call cost 30 cents
    report = run(m.dream(complete=model, apply=True, budget=0.50))
    assert len(model.prompts) == 2 and report.stopped_before is not None and report.reported == pytest.approx(0.60)


def test_since_is_a_log_position():
    m, model = mem(), Model()
    talk(m, "s1")
    talk(m, "s2")
    later = m.store.position()
    talk(m, "s3")
    report = run(m.dream(complete=model, since=later + 1))
    assert [p.session_id for p in report.planned] == ["s3"]


def test_reverting_a_dream_run_removes_what_it_wrote():
    m = mem()
    talk(m, "s1")
    run(m.dream(complete=Model(reply(("sister", "Lena visits Friday."))), apply=True))
    [rid] = {e.run_id for e in facts_of(m)}
    assert profile(m) == {"sister": "Lena visits Friday."}
    m.revert(run_id=rid)
    assert profile(m) == {}


def test_reverting_puts_back_the_value_a_dream_fact_replaced():
    m = mem()
    with m.session("s0", user_id="ada", agent="claude-code") as s:
        s.remember("home_city", "Paris")
    talk(m, "s1", text="I live in Rome now.")
    run(m.dream(complete=Model(reply(("home_city", "Ada lives in Rome."))), apply=True, since=m.store.position() - 5))
    assert profile(m)["home_city"] == "Ada lives in Rome."
    [rid] = {e.run_id for e in facts_of(m) if (e.run_id or "").startswith(PREFIX)}
    m.revert(run_id=rid)
    assert profile(m)["home_city"] == "Paris"


def test_a_later_statement_survives_the_revert_of_the_dream_it_replaced():
    m = mem()
    talk(m, "s1")
    run(m.dream(complete=Model(reply(("sister", "Lena visits Friday."))), apply=True))
    [rid] = {e.run_id for e in facts_of(m)}
    with m.session("s9", user_id="ada", agent="claude-code") as s:
        s.remember("sister", "Lena moved in.")
    m.revert(run_id=rid)
    assert profile(m)["sister"] == "Lena moved in."


def test_forget_user_reaches_everything_dream_wrote():
    m = mem()
    talk(m, "s1")
    talk(m, "s2", user="bob", text="I like tea.")
    run(m.dream(complete=Model(reply(("sister", "Lena visits Friday.")), reply(("drink", "Bob likes tea."))), apply=True))
    run(m.forget_user("ada"))
    assert [e.user_id for e in facts_of(m)] == ["bob"]
    assert all(e.user_id != "ada" for e in m.store.events(kinds=[Checkpoint.kind]))
    assert profile(m) == {} and profile(m, "bob") == {"drink": "Bob likes tea."}


def test_a_value_the_user_forgot_is_not_extracted_again():
    m = mem()
    talk(m, "s1", text="My sister Lena visits on Friday, and I fly to Oslo on Monday.")
    with m.session("s2", user_id="ada", agent="claude-code") as s:
        s.remember("sister_visit", "Lena visits Friday")
        s.forget("sister_visit")
    model = Model(reply(("sister", "Lena visits Friday."), ("trip", "Ada flies to Oslo Monday.")))
    run(m.dream(complete=model, apply=True))
    assert "Lena visits Friday" not in "".join(model.prompts)
    assert [e.key for e in facts_of(m) if e.source == "extracted"] == ["trip"]


def test_a_batch_pass_does_not_overwrite_what_a_newer_session_said():
    m = mem()
    talk(m, "old", text="I live in Paris.")
    with m.session("new", user_id="ada", agent="claude-code") as s:
        s.record(UserFact(key="home_city", value="Rome", source="explicit"))
        s.message("user", "I live in Rome now.")
    run(m.dream(complete=Model(reply(("home_city", "Ada lives in Paris.")), reply()), apply=True))
    assert profile(m)["home_city"] == "Rome"


def test_one_failing_session_does_not_cost_the_others_and_is_retried():
    m = mem()
    talk(m, "s1")
    talk(m, "s2", text="I fly out on Monday.")

    def flaky(prompt):
        if "Lena" in prompt:
            raise RuntimeError("provider down")
        return reply(("trip", "Ada flies Monday."))

    report = run(m.dream(complete=flaky, apply=True))
    assert list(report.failed) == ["s1"] and "s2" in report.facts
    retry = run(m.dream(complete=Model(reply(("sister", "Lena visits Friday."))), apply=True))
    assert [p.session_id for p in retry.planned] == ["s1"]


def test_dream_is_refused_on_an_engine_that_extracts_for_itself():
    m = mem()

    class Other:
        pass

    m.engine = Other()
    with pytest.raises(ValueError, match="local engine"):
        run(m.dream(complete=Model()))


def test_a_dream_with_no_model_is_refused():
    with pytest.raises(ValueError, match="needs a model"):
        run(mem().dream())
