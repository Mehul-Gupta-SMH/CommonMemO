import asyncio

import pytest

from commonmemo import Memory, ToolInvoked


@pytest.fixture
def mem():
    m = Memory()
    yield m
    asyncio.run(m.aclose())


def test_sync_call_is_recorded_and_unchanged(mem):
    @mem.track
    def add(a: int, b: int = 1) -> int:
        """Adds."""
        return a + b

    assert add(2, b=3) == 5
    assert add.__doc__ == "Adds." and add.__name__ == "add"
    (e,) = mem.store.events()
    assert isinstance(e, ToolInvoked)
    assert (e.name, e.arguments, e.result, e.error) == ("add", {"a": 2, "b": 3}, "5", None)
    assert e.duration_ms is not None and e.duration_ms >= 0


def test_async_call_and_configured_decorator(mem):
    @mem.track(name="search", capture_result=False)
    async def fetch(query: str) -> str:
        return "secret result"

    assert asyncio.run(fetch("x")) == "secret result"
    (e,) = mem.store.events()
    assert e.name == "search" and e.result is None and e.arguments == {"query": "x"}


def test_exception_passes_through_and_is_recorded(mem):
    @mem.track
    def boom() -> None:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        boom()
    (e,) = mem.store.events()
    assert e.error == "ValueError: bad input" and e.result is None


def test_records_into_the_current_session(mem):
    @mem.track
    def ping() -> str:
        return "pong"

    async def main():
        async with mem.session(user_id="u1", agent="bot") as s:
            ping()
        return s

    s = asyncio.run(main())
    (e,) = mem.store.events()
    assert (e.session_id, e.user_id, e.agent) == (s.session_id, "u1", "bot")


def test_a_broken_store_never_breaks_the_call(mem):
    @mem.track
    def ok() -> int:
        return 1

    def explode(event):
        raise RuntimeError("disk full")

    mem.store.append = explode  # type: ignore[method-assign]
    assert ok() == 1


def test_long_results_are_clipped(mem):
    @mem.track(max_chars=10)
    def big() -> str:
        return "x" * 100

    big()
    (e,) = mem.store.events()
    assert len(e.result) == 10
