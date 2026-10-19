"""Refreshing pushed memory mid-run (#22)."""

from __future__ import annotations

import asyncio

from commonmemo import Memory, MemoryPolicy

OLD = "the charger firmware rollback bricked unit 7; flash it over serial instead"


def earlier(mem: Memory, text: str, *, user: str = "ada") -> None:
    with mem.session(user_id=user, agent="claude-code") as s:
        s.message("user", text)


def run(coro):
    return asyncio.run(coro)


def test_a_tool_result_brings_up_an_older_session():
    mem = Memory()
    earlier(mem, OLD)
    with mem.session(user_id="ada", agent="claude-code") as s:
        turn = run(s.push_turn("Ship release 4.2"))
        assert turn.recaps == []  # nothing in the request points at the old session
        assert run(s.refresh("ran the build: 214 files compiled")) is None  # step 1
        assert run(s.refresh("tests: 88 passed")) is None  # step 2
        step3 = run(s.refresh("deploy: charger firmware rollback started on unit 7"))  # step 3's result
        assert step3 is not None and "flash it over serial" in step3.turn_text()  # pushed for step 4
        assert run(s.refresh("charger firmware rollback still running on unit 7")) is None  # already pushed
        [pushed] = [e for e in s.events(kinds=["context_pushed"]) if e.reason == "refresh"]
        assert pushed.chars == len(step3.turn_text()) and list(pushed.recap_ids) == [step3.recaps[0].id]


def test_too_few_shared_words_is_not_enough():
    mem = Memory()
    earlier(mem, OLD)
    with mem.session(user_id="ada", agent="claude-code") as s:
        run(s.push_turn("Ship release 4.2"))
        assert run(s.refresh("the charger is fine")) is None  # 1 shared word; the default needs 3


def test_refreshes_per_turn_are_bounded_and_reset_on_the_next_turn():
    mem = Memory(policy=MemoryPolicy(max_refreshes=2))
    topics = ["alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa lambda omicron"]
    for words in topics:
        earlier(mem, f"we discussed {words} at length")
    with mem.session(user_id="ada", agent="claude-code") as s:
        run(s.push_turn("Ship release 4.2"))
        got = [run(s.refresh(f"output mentions {words}")) for words in topics]
        assert [g is not None for g in got] == [True, True, False]  # the third is over budget
        run(s.push_turn("next request"))  # a new turn has its own budget
        assert run(s.refresh(f"output mentions {topics[2]}")) is not None


def test_refresh_can_be_turned_off():
    mem = Memory(policy=MemoryPolicy(refresh=False))
    earlier(mem, OLD)
    with mem.session(user_id="ada", agent="claude-code") as s:
        run(s.push_turn("Ship release 4.2"))
        assert run(s.refresh("deploy: charger firmware rollback started on unit 7")) is None
        assert [e.reason for e in s.events(kinds=["context_pushed"])] == ["turn"]


def test_a_recap_pushed_at_the_start_of_the_turn_is_not_pushed_again():
    mem = Memory()
    earlier(mem, OLD)
    with mem.session(user_id="ada", agent="claude-code") as s:
        assert run(s.push_turn("charger firmware rollback on unit 7")).recaps  # already pushed
        assert run(s.refresh("charger firmware rollback started on unit 7")) is None


def test_forgotten_values_stay_out_and_other_users_never_appear():
    mem = Memory()
    with mem.session(user_id="ada", agent="claude-code") as s:
        s.remember("serial_port", "/dev/ttyUSB7")
        s.message("user", "charger firmware rollback needs serial_port /dev/ttyUSB7")
    with mem.session(user_id="ada") as s:
        s.forget("serial_port")
    earlier(mem, "bob's charger firmware rollback secret plan", user="bob")
    with mem.session(user_id="ada", agent="claude-code") as s:
        run(s.push_turn("Ship release 4.2"))
        out = run(s.refresh("charger firmware rollback started"))
        assert out is not None
        text = out.turn_text()
        assert "/dev/ttyUSB7" not in text and "secret" not in text


def test_refresh_never_calls_the_wrapped_engine():
    from commonmemo.engines.local import LocalEngine

    class Remote:
        """A wrapped engine (like Graphiti or Mem0) whose search costs a round trip."""

        def __init__(self) -> None:
            self.index = LocalEngine(":memory:")
            self.searches = 0

        def __getattr__(self, name):
            return getattr(self.index, name)

        async def search_recaps(self, *args, **kwargs):
            self.searches += 1
            return await self.index.search_recaps(*args, **kwargs)

    engine = Remote()
    mem = Memory(engine=engine)
    earlier(mem, OLD)
    with mem.session(user_id="ada", agent="claude-code") as s:
        before = engine.searches
        assert run(s.refresh("deploy: charger firmware rollback started on unit 7")) is not None
        assert engine.searches == before  # found in the local index, no remote search
