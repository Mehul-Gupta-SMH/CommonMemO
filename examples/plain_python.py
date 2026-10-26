"""Memory in plain Python: no framework, no keys, no network.

Session one: a user says something, a tool fails and is fixed.
Session two: a different run starts, and memory is pushed in front of the prompt.

    python examples/plain_python.py

Then look at what happened: `commonmemo inspect --store <the path printed below>`.
"""

import asyncio
import tempfile
from pathlib import Path

from commonmemo import Memory, ToolInvoked


async def main() -> None:
    store = Path(tempfile.mkdtemp(prefix="commonmemo-example-")) / "memory.db"
    mem = Memory(f"sqlite:///{store}")

    # Session one: everything recorded here goes into one append-only log.
    async with mem.session(user_id="ada", agent="planner") as s:
        s.message("user", "I'm vegetarian, and I always book aisle seats.")
        s.record(ToolInvoked(name="search_flights", arguments={"date": "17/10/2026"}, error="ValueError: bad date '17/10/2026'"))
        s.record(ToolInvoked(name="search_flights", arguments={"date": "2026-10-17"}, result="3 flights"))
        s.message("assistant", "Booked an aisle seat on the 17th.")

    # Session two, later: ask what memory would push for this request.
    async with mem.session(user_id="ada", agent="planner") as s:
        context = await s.push_turn("Plan dinner for the trip. Any seating preferences?", recent=0)
        print(context.stable_text() + "\n\n" + context.turn_text())

    print(f"\nstore: {store}")


if __name__ == "__main__":
    asyncio.run(main())
