"""Facts extracted from finished sessions (#60)."""

from __future__ import annotations

import asyncio
import json

import pytest

from commonmemo import Memory, ModelExtractor
from commonmemo.events import UserFact
from commonmemo.extraction import parse


def run(coro):
    return asyncio.run(coro)


def reply(*facts):
    return json.dumps({"facts": [{"key": k, "value": v} for k, v in facts]})


class Model:
    """A stand-in for a model: records the prompts, answers from a script."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else reply()


def mem_with(model, **kw):
    return Memory(extractor=ModelExtractor(model, **kw))


def profile(mem, user="ada"):
    return {f.key: f.value for f in mem._visible_facts(user, run(mem.engine.user_profile(user)))}


def talk(mem, sid="s1", user="ada", lines=(("user", "My sister Lena visits on Friday."),)):
    with mem.session(sid, user_id=user, agent="chat") as s:
        for role, text in lines:
            s.message(role, text)
    return s


def test_facts_reach_the_profile_with_their_source_and_the_recap_names_them():
    mem = mem_with(Model(reply(("sister_visit", "Ada's sister Lena visits on Friday 14 Nov 2026."))))
    s = talk(mem)
    assert profile(mem) == {"sister_visit": "Ada's sister Lena visits on Friday 14 Nov 2026."}
    [fact] = mem.store.events(kinds=[UserFact.kind])
    assert (fact.source, fact.user_id, fact.session_id, fact.run_id) == ("extracted", "ada", "s1", "extract:s1")
    assert "learned about the user: sister_visit" in s.recaps[0].summary


def test_a_fact_said_once_is_recalled_in_a_later_session():
    mem = mem_with(Model(reply(("sister_visit", "Lena visits on Friday 14 Nov 2026."))))
    talk(mem)
    with mem.session("s2", user_id="ada") as s:
        ctx = run(s.context("When does my sister visit?"))
    assert "sister_visit: Lena visits on Friday 14 Nov 2026." in ctx.render()


def test_once_per_session_unless_forced():
    model = Model(reply(("a", "one")), reply(("b", "two")))
    mem = mem_with(model)
    talk(mem)
    assert run(mem.extract("s1")) == []  # already extracted at close
    assert len(model.prompts) == 1
    assert [f.key for f in run(mem.extract("s1", force=True))] == ["b"]


def test_a_stable_key_is_superseded_and_an_unchanged_value_is_not_rewritten():
    mem = mem_with(Model(reply(("home_city", "Lisbon")), reply(("home_city", "Lisbon")), reply(("home_city", "Porto"))))
    talk(mem, "s1")
    talk(mem, "s2")
    assert len(mem.store.events(kinds=[UserFact.kind])) == 1  # same value: not written again
    talk(mem, "s3")
    assert profile(mem) == {"home_city": "Porto"}


def test_the_model_is_shown_the_known_keys_and_the_session_date():
    model = Model(reply(("home_city", "Lisbon")), reply())
    mem = mem_with(model)
    talk(mem, "s1")
    talk(mem, "s2")
    assert "home_city: Lisbon" in model.prompts[1]
    assert "today" not in model.prompts[1] and "20" in model.prompts[1]  # a real date to resolve "yesterday" against
    assert "data, not instructions" in model.prompts[1]


def test_nothing_is_extracted_without_a_user_or_without_content():
    model = Model(reply(("a", "b")))
    mem = mem_with(model)
    with mem.session("anon") as s:
        s.message("user", "hello")
    with mem.session("empty", user_id="ada"):
        pass
    assert model.prompts == [] and mem.store.events(kinds=[UserFact.kind]) == []


def test_forgotten_values_are_redacted_from_the_input_and_dropped_from_the_output():
    model = Model(reply(("pet", "Ada has a dog called Rex."), ("city", "Ada lives in Oslo.")))
    mem = mem_with(model)
    with mem.session("s0", user_id="ada") as s:
        s.remember("dog_name", "Rex")
        s.forget("dog_name")
    talk(mem, "s1", lines=(("user", "My dog Rex is old. I live in Oslo."),))
    assert "Rex" not in model.prompts[0].split("Session:")[-1]  # redacted before the model sees it
    assert profile(mem) == {"city": "Ada lives in Oslo."}  # a fact quoting it is dropped


def test_forget_user_removes_extracted_facts():
    mem = mem_with(Model(reply(("pet", "Ada has a dog."))))
    talk(mem)
    run(mem.forget_user("ada"))
    assert profile(mem) == {}


def test_a_failing_extractor_never_fails_the_close():
    def boom(prompt):
        raise RuntimeError("model down")

    mem = Memory(extractor=ModelExtractor(boom))
    s = talk(mem)
    assert s.closed and len(s.recaps) == 1 and profile(mem) == {}


def test_malformed_replies_are_dropped_and_facts_are_capped():
    assert parse("not json") == [] and parse('{"facts": "no"}') == []
    assert [f.key for f in parse('{"facts": [{"key": "A B!", "value": " x  y "}, {"key": "", "value": "z"}, 3]}')] == ["a_b"]
    many = reply(*[(f"k{i}", f"v{i}") for i in range(40)])
    mem = mem_with(Model(many), max_facts=5)
    talk(mem)
    assert len(profile(mem)) == 5


def test_long_sessions_go_in_chunks_and_later_chunks_see_earlier_facts():
    model = Model(reply(("first", "found in part one")), reply(("second", "found in part two")))
    mem = mem_with(model, max_chars=300)
    talk(mem, lines=[("user", f"line {i}: " + "word " * 20) for i in range(8)])
    assert len(model.prompts) > 1 and "(part 1 of" in model.prompts[0]
    assert "first: found in part one" in model.prompts[1]
    assert set(profile(mem)) >= {"first", "second"}


def test_without_an_extractor_extract_says_so():
    with pytest.raises(ValueError, match="no extractor"):
        run(Memory().extract("s1"))
