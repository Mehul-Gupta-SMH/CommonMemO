"""Dates in message windows: stamps and relative dates resolved against the message (#71)."""

from __future__ import annotations

import asyncio
import time

from commonmemo import Memory, MemoryPolicy
from commonmemo.dates import annotate, stamp


def at(y, m, d, hh=13, mm=56):
    return time.mktime((y, m, d, hh, mm, 0, 0, 0, -1))


SUNDAY = at(2033, 5, 8)


def test_a_stamp_has_the_weekday_date_and_minute():
    assert stamp(SUNDAY) == "Sun 2033-05-08 13:56" and stamp(SUNDAY, time_of_day=False) == "Sun 2033-05-08"


def test_yesterday_tomorrow_and_named_days():
    assert annotate("I went yesterday.", SUNDAY) == "I went yesterday [= Sat 2033-05-07]."
    assert annotate("Lena comes tomorrow", SUNDAY) == "Lena comes tomorrow [= Mon 2033-05-09]"
    assert "last Friday [= Fri 2033-05-06]" in annotate("It was last Friday", SUNDAY)
    assert "next Friday [= Fri 2033-05-13]" in annotate("See you next Friday", SUNDAY)
    assert "last Sunday [= Sun 2033-05-01]" in annotate("last Sunday", SUNDAY)  # never today


def test_a_bare_weekday_is_the_past_one_only_with_a_past_cue():
    assert "Friday [= Fri 2033-05-13]" in annotate("See you on Friday!", SUNDAY)
    assert "Friday [= Fri 2033-05-06]" in annotate("We went to the lake on Friday.", SUNDAY)
    assert annotate("Wow. I will go on Friday", SUNDAY).endswith("Friday [= Fri 2033-05-13]")  # the cue is per sentence


def test_habits_and_plurals_are_left_alone():
    for text in ("Every Friday we meet", "I swim on Fridays", "each Monday", "Fridays"):
        assert annotate(text, SUNDAY) == text


def test_weeks_months_years_and_counts():
    assert "last week [= week of Mon 2033-04-25]" in annotate("last week", SUNDAY)
    assert "next week [= week of Mon 2033-05-09]" in annotate("next week", SUNDAY)
    assert "last month [= April 2033]" in annotate("last month", SUNDAY)
    assert "next year [= 2034]" in annotate("next year", SUNDAY)
    assert "two weeks ago [= Sun 2033-04-24]" in annotate("two weeks ago", SUNDAY)
    assert "3 days ago [= Thu 2033-05-05]" in annotate("3 days ago", SUNDAY)
    assert "a year ago [= 2032]" in annotate("a year ago", SUNDAY)
    assert "in 2 weeks [= Sun 2033-05-22]" in annotate("in 2 weeks", SUNDAY)
    assert "in a month [= June 2033]" in annotate("in a month", SUNDAY)
    assert "ago [= " in annotate("7 months ago", at(2033, 1, 31)) and "June 2032" in annotate("7 months ago", at(2033, 1, 31))


def test_weekends_follow_the_week_they_are_said_in():
    assert "last weekend [= Sat 2033-04-30 to Sun 2033-05-01]" in annotate("last weekend", SUNDAY)
    assert "this weekend [= Sat 2033-05-07 to Sun 2033-05-08]" in annotate("this weekend", SUNDAY)
    assert "next weekend [= Sat 2033-05-14 to Sun 2033-05-15]" in annotate("next weekend", SUNDAY)
    assert "last weekend [= Sat 2033-05-07 to Sun 2033-05-08]" in annotate("last weekend", at(2033, 5, 11))  # said on a Wednesday


def test_text_without_dates_is_unchanged():
    text = "The key is under the blue pot. Call me at 5 pm."
    assert annotate(text, SUNDAY) == text


# -- in windows ------------------------------------------------------------------------------


def run(coro):
    return asyncio.run(coro)


def mem_with(dates, **kw):
    mem = Memory(policy=MemoryPolicy(messages=2, message_dates=dates, **kw))
    with mem.session("s1", user_id="ada", agent="chat") as s:
        s.message("user", "My sister Lena arrives Friday and I need the guest room ready.", ts=SUNDAY)
        s.message("assistant", "Noted. Anything else?", ts=SUNDAY)
        s.message("user", "Yes, I saw her yesterday and the key is under the blue pot.", ts=at(2033, 5, 10, 9, 5))
    return mem


def ask(mem, query="when does my sister Lena arrive?"):
    with mem.session("s2", user_id="ada", agent="chat") as s:
        return run(s.context(query)).said


def test_off_shows_only_the_day_and_leaves_the_text_alone():
    [w] = ask(mem_with("off"))
    assert w.line().startswith("[session s1, 2033-05-08]\n") and "[=" not in w.text() and "(Tue" not in w.text()


def test_stamp_gives_the_window_and_any_message_recorded_later_their_time():
    [w] = ask(mem_with("stamp"))
    assert w.line().startswith("[session s1, Sun 2033-05-08 13:56]\n")
    assert w.lines[0] == "user: My sister Lena arrives Friday and I need the guest room ready."
    assert w.lines[2].startswith("user (Tue 2033-05-10 09:05): Yes, I saw her yesterday")
    assert "[=" not in w.text()


def test_resolve_adds_what_each_relative_date_means_against_its_own_message():
    [w] = ask(mem_with("resolve"))
    assert "arrives Friday [= Fri 2033-05-13] and" in w.lines[0]
    assert "yesterday [= Mon 2033-05-09]" in w.lines[2]  # said on Tuesday, not on Sunday
    assert "Friday" not in w.lines[2]


def test_the_ranker_scores_what_was_said_not_the_stamps_and_annotations():
    # on LoCoMo, ranking on the annotated text lost evidence that the plain text found (#71)
    mem = mem_with("resolve")
    assert ask(mem, "Tuesday 2033-05-10 Fri 2033-05-13") == []  # dates in the request match nothing in the text
    [w] = ask(mem, "guest room Lena")
    assert w.said.count("\n") == 2 and "[=" not in w.said and "(Tue" not in w.said and "[=" in w.text()


def test_forgotten_values_are_redacted_from_annotations_too():
    mem = Memory(policy=MemoryPolicy(messages=2, message_dates="resolve"))
    with mem.session("s1", user_id="ada") as s:
        s.remember("room", "Fri 2033-05-13")
        s.message("user", "Lena comes Friday to the guest room.", ts=SUNDAY)
        s.forget("room")
    [w] = ask(mem, "guest room Lena")
    assert "2033-05-13" not in w.text() and "[forgotten]" in w.text()


def test_the_flag_reaches_the_policy_and_the_hook_setup():
    import argparse

    from commonmemo.cli import policy_from
    from commonmemo.hooks import hook_config

    args = argparse.Namespace(store="sqlite:///.commonmemo/memory.db", engine="local", project=None, user="ada",
                              agent=None, session=None, read_only=False, recall_tool=False, no_environment=False,
                              messages=4, recall_messages=6, message_dates="resolve")
    assert policy_from(args).message_dates == "resolve"
    assert "--message-dates resolve" in hook_config("claude-code", args)["hooks"]["SessionStart"][0]["hooks"][0]["command"]
