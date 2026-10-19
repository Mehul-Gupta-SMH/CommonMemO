"""Durability: projections never drift from the log (#15), and a
side-effecting step runs once across concurrent workers (#16)."""

import threading
import time

import pytest

import commonmemo.knowledge as knowledge
from commonmemo import Memory, StepInDoubt, StepInProgress
from commonmemo.events import Feedback, MemoryEvent, ToolInvoked, event_from_dict
from commonmemo.resume import _claim_key, _step_id


class Crash(Exception):
    pass


def _tool_call(event_id: str, error=None) -> ToolInvoked:
    return ToolInvoked(id=event_id, name="search", arguments={"q": "x"}, error=error)


# -- #15: the log and its projections ----------------------------------------


def test_a_crash_between_append_and_apply_is_caught_up_on_next_open(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/m.db"
    mem = Memory(url)
    real = knowledge._HANDLERS[ToolInvoked]

    def die(self, event):
        raise Crash("process died after the append, before the apply")

    monkeypatch.setitem(knowledge._HANDLERS, ToolInvoked, die)
    with mem.session("s", user_id="u1") as s:
        with pytest.raises(Crash):
            s.record(_tool_call("e1", error="timeout"))
    assert [e.id for e in mem.store.events(kinds=["tool_invoked"])] == ["e1"]  # logged
    assert mem.tool_profile("search").calls == 0  # but not applied

    monkeypatch.setitem(knowledge._HANDLERS, ToolInvoked, real)
    reopened = Memory(url)  # the next process
    profile = reopened.tool_profile("search")
    assert (profile.calls, profile.failures) == (1, 1)


def test_a_retried_duplicate_repairs_the_gap_without_double_counting(monkeypatch):
    mem = Memory()
    real = knowledge._HANDLERS[ToolInvoked]
    monkeypatch.setitem(knowledge._HANDLERS, ToolInvoked, lambda self, e: (_ for _ in ()).throw(Crash()))
    with mem.session("s") as s:
        with pytest.raises(Crash):
            s.record(_tool_call("e1"))
        monkeypatch.setitem(knowledge._HANDLERS, ToolInvoked, real)
        s.record(_tool_call("e1"))  # the caller retries: the log sees a duplicate
        s.record(_tool_call("e1"))
    assert mem.tool_profile("search").calls == 1


def test_replaying_the_log_twice_counts_nothing_twice(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    mem = Memory(url)
    with mem.session("s", user_id="u1") as s:
        for i in range(3):
            s.record(_tool_call(f"e{i}", error="boom" if i == 0 else None))
        s.feedback(-1, "too long")
        s.correct("Pune", "Bangalore")
    before = (mem.tool_profile("search").calls, mem.feedback_summary().count, mem.feedback_summary().corrections)
    assert before == (3, 2, 1)

    assert mem.knowledge.catch_up(mem.store) == 0  # nothing new: nothing applied
    Memory(url)  # opening another handle catches up too
    Memory(url)
    mem.rebuild_knowledge()
    mem.rebuild_knowledge()
    after = (mem.tool_profile("search").calls, mem.feedback_summary().count, mem.feedback_summary().corrections)
    assert after == before


def test_two_handles_on_one_file_share_one_position(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    a, b = Memory(url), Memory(url)
    with a.session("s1") as s:
        s.record(_tool_call("a1"))
    with b.session("s2") as s:
        s.record(_tool_call("b1"))
    a.knowledge.catch_up(a.store)
    b.knowledge.catch_up(b.store)
    assert a.tool_profile("search").calls == b.tool_profile("search").calls == 2
    assert a.knowledge.applied_seq == b.knowledge.applied_seq


def test_feedback_round_trips_through_the_log():
    corrected = Feedback(feedback_kind="correction", wrong="Pune", right="Bangalore")
    back = event_from_dict(corrected.to_dict())
    assert type(back) is Feedback and back.feedback_kind == "correction"


def test_feedback_logged_under_the_old_field_name_still_reads_back():
    old = {"kind": "correction", "id": "f1", "ts": 1.0, "session_id": None, "user_id": "u1", "agent": None,
           "rating": -1.0, "comment": None, "wrong": "Pune", "right": "Bangalore", "target_event_id": None}
    back = event_from_dict(old)
    assert type(back) is Feedback and back.feedback_kind == "correction" and back.right == "Bangalore"


def test_a_field_named_kind_is_refused():
    with pytest.raises(TypeError, match="reserved"):
        class Bad(MemoryEvent):
            kind: str = "x"


# -- #16: claiming side-effecting steps ---------------------------------------


def test_concurrent_workers_run_a_side_effecting_step_once(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/m.db"
    effects, outcomes = [], []
    gate = threading.Barrier(4)
    read_step = knowledge.Knowledge.step

    def slow_read(self, *args):
        # Widen the gap between reading a step's status and acting on it, so
        # every worker reads "not started" -- a check-then-act race always loses here.
        rec = read_step(self, *args)
        time.sleep(0.05)
        return rec

    monkeypatch.setattr(knowledge.Knowledge, "step", slow_read)

    def worker():
        mem = Memory(url)  # each worker is its own handle, as in its own process

        @mem.step(name="charge", side_effects=True)
        def charge(order: str) -> str:
            effects.append(order)
            time.sleep(0.2)  # long enough for the others to arrive mid-charge
            return f"charged {order}"

        gate.wait()
        try:
            with mem.session("job", user_id="u1"):
                outcomes.append(charge("o-1"))
        except StepInDoubt as exc:
            outcomes.append(type(exc).__name__)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert effects == ["o-1"]  # charged exactly once
    assert outcomes.count("charged o-1") >= 1
    assert set(outcomes) <= {"charged o-1", "StepInProgress"}

    mem = Memory(url)

    @mem.step(name="charge", side_effects=True)
    def charge(order: str) -> str:
        effects.append(order)
        return "again"

    with mem.session("job", user_id="u1"):
        assert charge("o-1") == "charged o-1"  # later runs replay
    assert effects == ["o-1"]


def _hold(mem: Memory, sid: str, step_id: str, lease_until: float) -> None:
    """A worker that claimed the step and then died, holding the claim."""
    from commonmemo.events import StepStarted

    started = StepStarted(step_id=step_id, side_effects=True, session_id=sid, user_id="u1")
    assert mem.store.claim(_claim_key(sid, step_id), session_id=sid, owner="dead-worker",
                           lease_until=lease_until, event=started) is None
    mem.knowledge.catch_up(mem.store)


def test_a_live_lease_means_in_progress_and_an_expired_one_in_doubt(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    mem = Memory(url)
    runs = []

    @mem.step(name="email", side_effects=True)
    def email(to: str) -> str:
        runs.append(to)
        return "sent"

    sid_live, sid_dead = "job-live", "job-dead"
    step = _step_id("email", None, ("ada",), {})
    with mem.session(sid_live, user_id="u1"), mem.session(sid_dead, user_id="u1"):
        pass
    _hold(mem, sid_live, step, time.time() + 60)
    _hold(mem, sid_dead, step, time.time() - 1)

    with mem.session(sid_live, user_id="u1"):
        with pytest.raises(StepInProgress) as live:
            email("ada")
    assert live.value.owner == "dead-worker"

    with mem.session(sid_dead, user_id="u1") as s:
        with pytest.raises(StepInDoubt) as dead:
            email("ada")
        assert not isinstance(dead.value, StepInProgress)  # expired: in doubt, never "run again"
        s.resolve_step(step, failed=True)  # it did not take effect
        assert email("ada") == "sent"
        assert email("ada") == "sent"  # replayed
    assert runs == ["ada"]


def test_a_crash_inside_a_claimed_step_leaves_it_in_doubt(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"
    mem = Memory(url)

    @mem.step(name="pay", side_effects=True, lease=0.05)
    def pay() -> str:
        return "paid"

    with mem.session("job", user_id="u1") as s:
        # Claimed by a worker that then died. The lease leaves a slow CI runner
        # plenty of time to see it live (it was 0.05s and flaked).
        _hold(mem, "job", "pay", time.time() + 0.5)
        with pytest.raises(StepInProgress):
            pay()
        time.sleep(0.6)
        with pytest.raises(StepInDoubt) as info:
            pay()
        assert not isinstance(info.value, StepInProgress)
        s.resolve_step("pay", result="paid")
        assert pay() == "paid"


def test_a_failed_claimed_step_is_released_and_runs_again():
    mem = Memory()
    attempts = []

    @mem.step(side_effects=True)
    def send() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise Crash("smtp down")
        return "sent"

    with mem.session("s"):
        with pytest.raises(Crash):
            send()
        assert send() == "sent"
        assert send() == "sent"
    assert len(attempts) == 2


def test_forget_user_removes_their_claims(tmp_path):
    import asyncio

    mem = Memory(f"sqlite:///{tmp_path}/m.db")

    @mem.step(side_effects=True)
    def act() -> str:
        return "ok"

    with mem.session("job", user_id="u1"):
        act()
    key = _claim_key("job", _step_id(act.__qualname__, None, (), {}))
    assert mem.store.get_claim(key) is not None
    asyncio.run(mem.forget_user("u1"))
    assert mem.store.get_claim(key) is None


def test_a_step_that_settles_after_a_stale_read_is_replayed_not_in_doubt(tmp_path, monkeypatch):
    # #40: worker B reads "started" while the holder is still running; the
    # holder then finishes and settles before B looks at the claim. B must
    # replay the result, not report the step in doubt.
    import dataclasses

    url = f"sqlite:///{tmp_path}/m.db"
    runs = []

    def make(mem):
        @mem.step(name="charge", side_effects=True)
        def charge(order: str) -> str:
            runs.append(order)
            return f"charged {order}"

        return charge

    a = Memory(url)
    charge_a = make(a)
    with a.session("job", user_id="u1"):
        assert charge_a("o-1") == "charged o-1"  # A finished; its claim is settled

    b = Memory(url)
    charge_b = make(b)
    real = knowledge.Knowledge.step
    stale = {"left": 1}

    def first_read_is_stale(self, *args):
        rec = real(self, *args)
        if stale["left"] and rec is not None:
            stale["left"] -= 1
            return dataclasses.replace(rec, status="started")  # what B saw before A finished
        return rec

    monkeypatch.setattr(knowledge.Knowledge, "step", first_read_is_stale)
    with b.session("job", user_id="u1"):
        assert charge_b("o-1") == "charged o-1"  # replayed
    assert runs == ["o-1"]


def test_a_claim_that_settles_while_we_look_is_replayed_not_in_doubt(tmp_path, monkeypatch):
    # #40, second window: B's read is stale (step not started yet), B's claim
    # attempt finds A holding it, and A settles before B checks the claim
    # again. B must replay, not report the step in doubt.
    import dataclasses

    from commonmemo.stores.sqlite import SQLiteStore

    url = f"sqlite:///{tmp_path}/m.db"
    runs = []

    def make(mem):
        @mem.step(name="charge", side_effects=True)
        def charge(order: str) -> str:
            runs.append(order)
            return f"charged {order}"

        return charge

    a = Memory(url)
    with a.session("job", user_id="u1"):
        make(a)("o-1")  # A finished

    b = Memory(url)
    charge_b = make(b)
    real_step, real_claim = knowledge.Knowledge.step, SQLiteStore.claim
    stale = {"step": 1, "claim": 1}

    def stale_step(self, *args):
        if stale["step"]:
            stale["step"] -= 1
            return None  # B read before A started
        return real_step(self, *args)

    def stale_claim(self, *args, **kwargs):
        found = real_claim(self, *args, **kwargs)
        if stale["claim"] and found is not None:
            stale["claim"] -= 1
            return dataclasses.replace(found, status="held")  # A still held it when B tried
        return found

    monkeypatch.setattr(knowledge.Knowledge, "step", stale_step)
    monkeypatch.setattr(SQLiteStore, "claim", stale_claim)
    with b.session("job", user_id="u1"):
        assert charge_b("o-1") == "charged o-1"  # replayed
    assert runs == ["o-1"]
