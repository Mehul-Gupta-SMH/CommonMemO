import dataclasses

import pytest

from commonmemo import ExternalEvent, MemoryEvent, Message, ToolInvoked, UserFact, event_from_dict


class OrderPlaced(MemoryEvent):
    order_id: str
    amount: float


class Custom(MemoryEvent):
    kind = "my.custom"
    note: str = ""


def test_subclass_is_a_frozen_kw_only_dataclass_without_a_decorator():
    e = OrderPlaced(order_id="A1", amount=40.0)
    assert dataclasses.is_dataclass(e)
    assert e.kind == "order_placed"
    assert e.payload == {"order_id": "A1", "amount": 40.0}
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.order_id = "B2"  # type: ignore[misc]
    with pytest.raises(TypeError):
        OrderPlaced("A1", 40.0)  # type: ignore[misc]  -- keyword-only


def test_explicit_kind_is_kept():
    assert Custom(note="x").kind == "my.custom"


def test_round_trip_through_dict():
    for e in [
        OrderPlaced(order_id="A1", amount=1.5, user_id="u1"),
        Message(role="user", content="hi"),
        ToolInvoked(name="search", arguments={"q": "x"}, result="ok", duration_ms=1.0),
        UserFact(key="units", value="metric"),
        Custom(note="n"),
    ]:
        assert event_from_dict(e.to_dict()) == e


def test_unknown_kind_comes_back_as_external_event_not_an_error():
    e = event_from_dict({"kind": "never.defined", "id": "x1", "ts": 1.0, "foo": 1})
    assert isinstance(e, ExternalEvent)
    assert e.source == "never.defined"
    assert e.data == {"foo": 1}
    assert e.id == "x1"


def test_duplicate_kind_from_a_different_class_is_rejected():
    with pytest.raises(TypeError, match="already registered"):

        class Clash(MemoryEvent):  # noqa: F841
            kind = "order_placed"


def test_bind_fills_only_unset_envelope_fields():
    e = Message(role="user", content="hi", agent="explicit")
    bound = e.bind(session_id="s1", user_id="u1", agent="session-agent")
    assert (bound.session_id, bound.user_id, bound.agent) == ("s1", "u1", "explicit")
    assert bound.id == e.id
