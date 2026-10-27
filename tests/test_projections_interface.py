"""The `Projections` interface (#52): what memory needs from them, by operation."""

import asyncio
import inspect

import pytest

from commonmemo import Memory
from commonmemo.knowledge import Knowledge
from commonmemo.projections import InterfaceOnly, Projections


def test_every_public_operation_of_the_sql_projections_is_in_the_interface():
    public = {n for n, m in inspect.getmembers(Knowledge) if not n.startswith("_") and (callable(m) or isinstance(m, property))}
    declared = set(Projections.__abstractmethods__)
    assert public - {"path"} == declared, (public ^ declared)  # add an operation to both, or to neither


def test_the_records_are_importable_from_where_they_always_were():
    from commonmemo.knowledge import (  # noqa: F401
        FeedbackSummary, NegativeRecord, ProcedureRecord, StepRecord, WorldRecord, source_rank)
    from commonmemo.projections import WorldRecord as moved
    assert WorldRecord is moved and source_rank("agent") == 1


def test_interface_only_hides_everything_else():
    wrapped = InterfaceOnly(Knowledge(":memory:"))
    assert wrapped.path == ":memory:" and wrapped.negatives(["global"]) == []
    for hidden in ("_conn", "_lock", "_world", "window_vectors_table"):
        with pytest.raises(AttributeError):
            getattr(wrapped, hidden)
    with pytest.raises(AttributeError):
        wrapped.path = "elsewhere"
    assert isinstance(wrapped.scratch(), InterfaceOnly)  # a scratch copy stays behind the interface


def test_memory_takes_projections_and_runs_through_the_interface_alone():
    projections = InterfaceOnly(Knowledge(":memory:"))
    mem = Memory("memory://", projections=projections)
    assert mem.knowledge is projections
    mem.learn("refund_window", "30 days")
    with mem.session("s1", user_id="ada") as s:
        s.message("user", "the guest room key is under the blue pot")
        s.remember("preferred_contact", "email")
        s.never("call me", reason="hates calls")
    mem.snapshot("v1")
    with mem.session("s2", user_id="ada") as s:
        ctx = asyncio.run(s.push_turn("where is the guest room key"))
    assert "blue pot" in ctx.turn_text() and "30 days" in ctx.stable_text()
    assert [n.statement for n in ctx.negatives] == ["call me"]
    assert mem.as_of("v1").world_facts()  # replays read a scratch() projection, behind the interface too
    with mem.run("r1"):
        mem.learn("k", "v")
    assert mem.revert(run_id="r1").events  # a revert replays the log into scratch() projections
    assert "k" not in {w.key for w in mem.world_facts()}
    assert asyncio.run(mem.forget_user("ada")) > 0
