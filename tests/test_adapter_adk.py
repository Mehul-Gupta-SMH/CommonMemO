"""The ADK adapter (#109), against the installed google-adk's own types and base class."""

import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.events.event import Event  # noqa: E402
from google.adk.memory.base_memory_service import BaseMemoryService  # noqa: E402
from google.adk.memory.memory_entry import MemoryEntry  # noqa: E402
from google.adk.sessions.session import Session as AdkSession  # noqa: E402
from google.genai import types  # noqa: E402

from commonmemo import Memory  # noqa: E402
from commonmemo.adapters.adk import CommonMemoMemoryService  # noqa: E402


def say(author, text, event_id, ts=1000.0):
    return Event(id=event_id, author=author, invocation_id="inv", timestamp=ts,
                 content=types.Content(role="user" if author == "user" else "model", parts=[types.Part(text=text)]))


def adk_session(session_id="s1", user_id="ada"):
    return AdkSession(id=session_id, app_name="shop", user_id=user_id, events=[
        say("user", "My daughter Maya is allergic to peanuts, so no peanut snacks.", "e1"),
        say("support", "Noted, I will keep peanuts out of every order.", "e2", 1001.0),
        Event(id="e3", author="support", invocation_id="inv", timestamp=1002.0),  # no text: not stored
    ])


@pytest.fixture
def service(tmp_path):
    return CommonMemoMemoryService(Memory(f"sqlite:///{tmp_path}/m.db", project="shop"))


def search(service, query, user_id="ada"):
    return asyncio.run(service.search_memory(app_name="shop", user_id=user_id, query=query)).memories


def texts(memories):
    return " ".join(m.content.parts[0].text for m in memories)


def test_is_an_adk_memory_service(service):
    assert isinstance(service, BaseMemoryService)


def test_a_session_added_is_found_by_what_was_said(service):
    asyncio.run(service.add_session_to_memory(adk_session()))
    found = search(service, "which snacks is Maya allergic to?")
    assert found and "peanut" in texts(found)
    assert all(isinstance(m, MemoryEntry) and m.custom_metadata["kind"] for m in found)


def test_adding_the_same_session_twice_records_nothing_new(service):
    asyncio.run(service.add_session_to_memory(adk_session()))
    before = len(service.memory.store.events())
    asyncio.run(service.add_session_to_memory(adk_session()))
    assert len(service.memory.store.events()) == before


def test_events_without_text_are_not_stored(service):
    asyncio.run(service.add_session_to_memory(adk_session()))
    messages = [e for e in service.memory.store.events() if e.kind == "message"]
    assert [m.role for m in messages] == ["user", "assistant"]


def test_add_events_to_memory_is_a_delta(service):
    delta = [say("user", "I prefer invoices as PDF attachments.", "d1")]
    asyncio.run(service.add_events_to_memory(app_name="shop", user_id="ada", events=delta, session_id="s2"))
    assert "invoices" in texts(search(service, "how do I like invoices"))


def test_add_memory_becomes_a_user_fact(service):
    entry = MemoryEntry(id="home_city", content=types.Content(parts=[types.Part(text="Ada lives in Lisbon")]))
    asyncio.run(service.add_memory(app_name="shop", user_id="ada", memories=[entry]))
    found = search(service, "where does Ada live")
    assert "Lisbon" in texts(found) and found[0].custom_metadata["kind"] == "user"
    asyncio.run(service.add_memory(app_name="shop", user_id="ada", memories=[entry]))
    assert len([e for e in service.memory.store.events() if e.kind == "user_fact"]) == 1


def test_one_user_never_finds_anothers_memory(service):
    asyncio.run(service.add_session_to_memory(adk_session()))
    assert search(service, "which snacks is Maya allergic to?", user_id="grace") == []


def test_a_search_does_not_close_or_consolidate_anything(service):
    asyncio.run(service.add_session_to_memory(adk_session()))
    search(service, "peanuts")
    search(service, "peanuts")
    assert not [e for e in service.memory.store.events() if e.kind == "session_closed"]


def test_max_results_caps_the_answer(tmp_path):
    service = CommonMemoMemoryService(Memory(f"sqlite:///{tmp_path}/m.db", project="shop"), max_results=1)
    entries = [MemoryEntry(id=f"k{i}", content=types.Content(parts=[types.Part(text=f"Ada likes tea number {i}")]))
               for i in range(3)]
    asyncio.run(service.add_memory(app_name="shop", user_id="ada", memories=entries))
    assert len(search(service, "what tea does Ada like")) == 1
