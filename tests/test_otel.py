"""Turn traces as OpenTelemetry spans (#25)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from commonmemo import Memory, MemoryPolicy  # noqa: E402
from commonmemo.otel import SPAN_NAME, export, instrument  # noqa: E402


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


def seeded(**kwargs) -> Memory:
    mem = Memory(**kwargs)
    mem.learn("refund_window", "30 days")
    with mem.session("s1", user_id="ada") as s:
        s.remember("preferred_contact", "email")
    return mem


def test_every_push_becomes_a_span_under_the_current_one(spans):
    exporter, tracer = spans
    mem = seeded()
    instrument(mem, tracer=tracer)
    with tracer.start_as_current_span("llm.call"):
        with mem.session("s2", user_id="ada", agent="support") as s:
            asyncio.run(s.push_turn("hello"))
    llm = next(sp for sp in exporter.get_finished_spans() if sp.name == "llm.call")
    [push] = [sp for sp in exporter.get_finished_spans() if sp.name == SPAN_NAME]
    assert push.parent.span_id == llm.context.span_id  # next to the LLM call
    a = push.attributes
    assert a["commonmemo.reason"] == "turn" and a["session.id"] == "s2" and a["commonmemo.agent"] == "support"
    assert list(a["commonmemo.facts"]) == ["preferred_contact"] and list(a["commonmemo.world_facts"]) == ["refund_window"]
    assert a["commonmemo.items"] == len(push.events) and "enduser.id" not in a and "commonmemo.text" not in a


def test_user_and_text_only_when_asked(spans):
    exporter, tracer = spans
    mem = seeded(policy=MemoryPolicy(trace_text=True))
    with mem.session("s2", user_id="ada") as s:
        asyncio.run(s.push("hello", reason="start", part="all"))  # stable and turn: the facts are in it
    assert export(mem, session_id="s2", tracer=tracer) == 1
    assert "commonmemo.text" not in exporter.get_finished_spans()[0].attributes
    export(mem, session_id="s2", tracer=tracer, include_user=True, include_text=True)
    a = exporter.get_finished_spans()[-1].attributes
    assert a["enduser.id"] == "ada" and "email" in a["commonmemo.text"]


def test_a_broken_exporter_never_breaks_a_write():
    mem = seeded()

    class Broken:
        def start_span(self, *a, **k):
            raise RuntimeError("collector down")

    instrument(mem, tracer=Broken())
    with mem.session("s2", user_id="ada") as s:
        asyncio.run(s.push_turn("hello"))
    assert len(mem.traces(session_id="s2")) == 1
