"""Checkpoints and resumable steps: restart a stopped process from where it was."""

import asyncio

import pytest

from commonmemo import Memory, StepInDoubt
from commonmemo.events import StepStarted


class Crash(Exception):
    pass


def test_rerun_skips_finished_steps_and_continues(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    calls = []

    def pipeline(mem, crash_at=None):
        @mem.step
        def fetch(n: int) -> list:
            calls.append(("fetch", n))
            return list(range(n))

        @mem.step
        def total(xs: list) -> int:
            calls.append(("total", len(xs)))
            if crash_at == "total":
                raise Crash()
            return sum(xs)

        with mem.session("job-1", user_id="u1") as s:
            xs = fetch(4)
            s.checkpoint("fetched", {"count": len(xs)})
            return total(xs)

    with pytest.raises(Crash):
        pipeline(Memory(url), crash_at="total")
    assert calls == [("fetch", 4), ("total", 4)]

    calls.clear()
    assert pipeline(Memory(url)) == 6  # a new process, same session id
    assert calls == [("total", 4)]  # fetch replayed from memory, not re-run


def test_resume_point_reports_checkpoint_steps_and_work_since(tmp_path):
    mem = Memory(f"sqlite:///{tmp_path}/m.db")

    @mem.step
    def a() -> str:
        return "A"

    with mem.session("job", user_id="u1") as s:
        a()
        s.checkpoint("after a", {"next": "b"})
        s.message("assistant", "working on b")
        s.record(StepStarted(step_id="b"))  # the process dies inside b
    with mem.session("job", user_id="u1") as s:
        rp = s.resume_point()
        ctx = asyncio.run(s.context("continue"))
    assert rp.checkpoint.label == "after a" and rp.checkpoint.data == {"next": "b"}
    assert [x.step_id for x in rp.completed] == ["test_resume_point_reports_checkpoint_steps_and_work_since.<locals>.a"]
    assert [x.step_id for x in rp.in_doubt] == ["b"]
    assert any(getattr(e, "content", "") == "working on b" for e in rp.since_checkpoint)
    turn = ctx.turn_text()
    assert "Last checkpoint 'after a'" in turn and "Interrupted steps" in turn


def test_side_effecting_step_in_doubt_is_not_silently_repeated(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    charges = []

    def make(mem):
        @mem.step(name="charge", side_effects=True)
        def charge(order: str) -> str:
            charges.append(order)
            return f"charged {order}"
        return charge

    mem = Memory(url)
    charge = make(mem)
    with mem.session("job", user_id="u1") as s:
        # Simulate dying after starting the charge but before recording it.
        s.record(StepStarted(step_id="charge:" + __import__("commonmemo.resume", fromlist=["x"])._step_id(
            "charge", None, ("o-1",), {}).split(":", 1)[1], side_effects=True))
    mem2 = Memory(url)
    charge2 = make(mem2)
    with mem2.session("job", user_id="u1") as s:
        with pytest.raises(StepInDoubt) as info:
            charge2("o-1")
        assert charges == []  # not charged a second time
        s.resolve_step(info.value.step_id, result="charged o-1")
        assert charge2("o-1") == "charged o-1"  # settled by hand, replayed
    assert charges == []


def test_failed_step_runs_again_and_args_give_separate_steps():
    mem = Memory()
    attempts = []

    @mem.step
    def flaky(x: int) -> int:
        attempts.append(x)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return x * 2

    with mem.session("s") as s:
        with pytest.raises(RuntimeError):
            flaky(1)
        assert flaky(1) == 2  # the failed attempt is retried
        assert flaky(2) == 4  # different arguments, different step
        assert flaky(1) == 2  # finished: replayed
    assert attempts == [1, 1, 2]
    assert [x.status for x in s.resume_point().completed] == ["finished", "finished"]


def test_non_json_results_run_again_instead_of_replaying_wrongly():
    mem = Memory()
    runs = []

    @mem.step
    def make_obj() -> object:
        runs.append(1)
        return object()

    with mem.session("s"):
        make_obj()
        make_obj()
    assert len(runs) == 2


def test_async_steps_and_plain_calls_outside_a_session():
    mem = Memory()
    runs = []

    @mem.step(key=lambda user: user)
    async def greet(user: str) -> str:
        runs.append(user)
        return f"hi {user}"

    assert asyncio.run(greet("ada")) == "hi ada"  # no session: a plain call

    async def main():
        async with mem.session("s"):
            await greet("ada")
            return await greet("ada")

    assert asyncio.run(main()) == "hi ada"
    assert runs == ["ada", "ada"]  # once outside, once inside; the replay did not run
