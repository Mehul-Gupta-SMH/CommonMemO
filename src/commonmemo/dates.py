"""Dates for message windows: when a message was said, and what its relative dates mean (#71).

"See you Friday" is only useful if you know which week it was said in. A window
shows each message's own stamp, and can annotate relative dates, resolved
against that message's timestamp:

    user: Lena arrives Friday [= Fri 2033-05-09]

The annotation is derived from the Log's text and timestamp when a window is
read; it is never stored and never replaces what was said, so it follows
`prune`, `forget_user` and redaction like the text does. It uses the calendar
only (no model, no dependency), so it is exact where the phrase is, and it
leaves alone what it cannot place: "Fridays", "every Friday", "a while ago".

One judgment call: a bare weekday ("on Friday") is read as the past one when
the sentence uses a past-tense cue (was, went, had, saw, ...) and as the next
one otherwise. "last Friday" and "next Friday" are not guesses.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional

_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_NUM = r"\d+|" + "|".join(_NUMBERS)
_DAY = "|".join(_WEEKDAYS)

_PAST = re.compile(r"\b(was|were|had|did|went|saw|got|took|made|ran|came|met|visited|attended|joined|started|"
                   r"finished|played|bought|gave|left|called)\b", re.IGNORECASE)

_PATTERNS = re.compile(
    rf"\b(?P<yesterday>yesterday|last night)\b"
    rf"|\b(?P<tomorrow>tomorrow)\b"
    rf"|\b(?P<rel_day>(?P<rel>last|next|this) (?P<rel_wd>{_DAY}))\b"
    rf"|\b(?P<weekend>(?P<wk_rel>last|next|this) weekend)\b"
    rf"|\b(?P<span>(?P<sp_rel>last|next) (?P<unit>week|month|year))\b"
    rf"|\b(?P<ago>(?P<ago_n>{_NUM}) (?P<ago_unit>day|week|month|year)s? ago)\b"
    rf"|\b(?P<soon>in (?P<in_n>{_NUM}) (?P<in_unit>day|week|month|year)s?)\b"
    rf"|(?<![-\w])(?:(?P<every>every|each|on every|most) )?(?P<bare>{_DAY})(?P<plural>s)?\b",
    re.IGNORECASE,
)


def _day(d: date) -> str:
    return f"{_DAYS[d.weekday()]} {d.isoformat()}"


def _month_shift(d: date, months: int) -> date:
    index = d.year * 12 + d.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _weekday_before(d: date, weekday: int) -> date:
    return d - timedelta(days=(d.weekday() - weekday) % 7 or 7)


def _weekday_after(d: date, weekday: int) -> date:
    return d + timedelta(days=(weekday - d.weekday()) % 7 or 7)


def _count(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBERS[word.lower()]


def _span(unit: str, d: date, sign: int, n: int = 1) -> str:
    if unit == "day":
        return _day(d + timedelta(days=sign * n))
    if unit == "week":
        return _day(d + timedelta(days=sign * 7 * n))
    if unit == "month":
        shifted = _month_shift(d, sign * n)
        return f"{_MONTHS[shifted.month - 1]} {shifted.year}"
    return str(d.year + sign * n)


def resolve(text: str, ts: float) -> list[tuple[int, str]]:
    """`(end of the phrase in text, what it means)` for each relative date in `text`, said at `ts`."""
    when = datetime.fromtimestamp(ts).date()
    out: list[tuple[int, str]] = []
    for m in _PATTERNS.finditer(text):
        meaning: Optional[str] = None
        if m["yesterday"]:
            meaning = _day(when - timedelta(days=1))
        elif m["tomorrow"]:
            meaning = _day(when + timedelta(days=1))
        elif m["rel_day"]:
            wd = _WEEKDAYS.index(m["rel_wd"].lower())
            rel = m["rel"].lower()
            meaning = _day(_weekday_before(when, wd) if rel == "last" else _weekday_after(when, wd))
        elif m["weekend"]:
            # weeks run Monday to Sunday: "this weekend" is the one of this week, even on the Sunday
            sat = when - timedelta(days=when.weekday()) + timedelta(days={"last": -2, "this": 5, "next": 12}[m["wk_rel"].lower()])
            meaning = f"Sat {sat.isoformat()} to Sun {(sat + timedelta(days=1)).isoformat()}"
        elif m["span"]:
            sign = -1 if m["sp_rel"].lower() == "last" else 1
            unit = m["unit"].lower()
            if unit == "week":
                monday = when - timedelta(days=when.weekday()) + timedelta(days=7 * sign)
                meaning = f"week of {_day(monday)}"
            else:
                meaning = _span(unit, when, sign)
        elif m["ago"]:
            meaning = _span(m["ago_unit"].lower(), when, -1, _count(m["ago_n"]))
        elif m["soon"]:
            meaning = _span(m["in_unit"].lower(), when, 1, _count(m["in_n"]))
        elif m["bare"] and not m["every"] and not m["plural"]:
            wd = _WEEKDAYS.index(m["bare"].lower())
            sentence = text[max(text.rfind(".", 0, m.start()), text.rfind("?", 0, m.start()),
                                text.rfind("!", 0, m.start())) + 1:m.end()]
            meaning = _day(_weekday_before(when, wd) if _PAST.search(sentence) else _weekday_after(when, wd))
        if meaning:
            out.append((m.end(), meaning))
    return out


def annotate(text: str, ts: float) -> str:
    """`text` with each relative date followed by `[= what it means]`."""
    found = resolve(text, ts)
    for end, meaning in reversed(found):
        text = f"{text[:end]} [= {meaning}]{text[end:]}"
    return text


def stamp(ts: float, *, time_of_day: bool = True) -> str:
    """`Fri 2033-05-09 13:56`: the weekday, date and minute a message was recorded."""
    d = datetime.fromtimestamp(ts)
    out = _day(d.date())
    return f"{out} {d:%H:%M}" if time_of_day else out


__all__ = ["annotate", "resolve", "stamp"]
