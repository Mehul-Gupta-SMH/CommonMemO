"""Memory tools for agents -- the *pull* half of hybrid memory.

`Memory.agent_tools(session)` returns plain, typed, documented functions
bound to one session, so any framework that turns Python functions into
tools can use them as they are (CommonADK requires exactly that shape).

Reads go further than the push: `memory_recall` searches every kind with no
relevance threshold. Writes are recorded with `source="agent"`, so what an
agent wrote can always be told apart from what the user or the application
stated -- and reviewed, or revoked.

Every name starts with `memory_`, so the tools cannot collide with an
agent's own.

The tools are **synchronous**: not every framework awaits an async tool
(CommonADK's Claude adapter calls tool functions directly). Engine search
is async, so `memory_recall` runs it on a worker thread with its own event
loop; an engine whose client is bound to the caller's loop (Graphiti's
async Neo4j driver can be) falls back to the exact local index rather than
failing the tool call.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .context import world_line
from .events import Checkpoint, Negative, UserFact
from .ranking import rank

if TYPE_CHECKING:
    from .session import Session

def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run `coro` to completion from synchronous code, inside or outside a
    running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def build_agent_tools(session: "Session", *, writes: bool = True) -> list[Callable[..., str]]:
    """The memory tools for `session`. `writes=False` gives read-only
    tools (recall only). The session then knows its agent can recall, so
    its push says when it left things out."""
    memory = session.memory
    session.recall_tool = True

    def memory_recall(query: str) -> str:
        """Search long-term memory: past sessions, facts about the user,
        known world facts, and past feedback. Use it when the request may
        depend on something from an earlier conversation that is not in
        front of you. Returns matching entries, best first, or a note that
        nothing matched.

        Args:
            query: What to look for, in plain words.
        """
        try:
            ctx = run_sync(session.context(query, policy=memory.policy_for_recall))
        except Exception:  # noqa: BLE001 -- see module docstring: degrade to the local index
            ctx = run_sync(session.context(query, policy=memory.policy_for_recall, local_only=True))
        bar = memory.policy_for_recall.min_recall_score

        def related(items: list, text: Callable[[Any], str]) -> list:
            # Best first, by the memory's ranker: by meaning with an embedder (#21).
            return run_sync(rank(memory.ranker, query, items, text, threshold=bar))[0]

        lines = []
        lines += [f"[user] {f.key}: {f.value}" for f in related(ctx.facts, lambda f: f"{f.key} {f.value}")]
        lines += [f"[fact] {world_line(w)}" for w in related([*ctx.world, *ctx.world_turn], memory._world_text)]
        lines += [f"[feedback] {item.line()}" for item in related(ctx.lessons, lambda item: item.line())]
        lines += [f"[past session] {ctx.recap_line(e)}" for e in ctx.recaps]
        return "\n".join(lines) if lines else "Nothing in memory matches that."

    if not writes:
        return [memory_recall]

    def memory_remember(key: str, value: str) -> str:
        """Save a durable fact about the current user for future sessions,
        e.g. key "preferred_contact", value "email". Use it only for things
        the user stated or clearly confirmed. A new value for the same key
        replaces the old one.

        Args:
            key: Short snake_case name of the fact.
            value: The fact itself.
        """
        if session.user_id is None:
            return "Not saved: this session has no user to remember things about."
        session.record(UserFact(key=key, value=value, source="agent"))
        return f"Saved: {key} = {value}"

    def memory_note_fact(key: str, value: str, valid_for_seconds: int = 0) -> str:
        """Note a fact about the world or this project (not about the user)
        for future sessions, e.g. key "refund_window", value "30 days".
        Facts you note are proposals: only this user sees them until an
        operator approves them, because a world fact is shown to every user.
        Set valid_for_seconds for facts that go stale (prices, statuses);
        0 means it does not expire.

        Args:
            key: Short snake_case name of the fact.
            value: The fact itself.
            valid_for_seconds: How long the fact stays true; 0 for no expiry.
        """
        event = session.learn(key, value, ttl=valid_for_seconds or None, source="agent")
        if event.proposed:
            why = f" Note: {event.note}." if event.note else ""
            return f"Proposed fact (other users see it once an operator approves it): {key} = {value}.{why}"
        return f"Saved fact: {key} = {value}"

    if memory.fact_keys:
        # List the registered keys where the model reads them: the tool's own
        # description, so it reuses `refund_window` instead of inventing
        # `refund_period`.
        head, _, args = (memory_note_fact.__doc__ or "").partition("\n\n        Args:")
        known = "\n".join(f"        - {line}" for line in memory.fact_keys.agent_lines())
        memory_note_fact.__doc__ = f"{head}\n\n        Use one of these keys when it fits:\n{known}\n\n        Args:{args}"

    def memory_avoid(statement: str, reason: str = "") -> str:
        """Record something to never do again for this user (or, without a
        user, for this agent), e.g. "suggest phone calls". Use it when the
        user asks you to stop doing something.

        Args:
            statement: What to avoid, phrased as an action.
            reason: Why, if the user said.
        """
        scope = f"user:{session.user_id}" if session.user_id else f"agent:{session.agent or 'default'}"
        session.record(Negative(statement=statement, scope=scope, reason=reason or None, source="agent"))
        return f"Will avoid: {statement}"

    def memory_save_progress(note: str, data_json: str = "{}") -> str:
        """Record progress on a long task so it can be resumed if this
        session is interrupted: what is done and what comes next.

        Args:
            note: One line on where the work stands.
            data_json: Optional JSON object with details needed to continue.
        """
        try:
            data = json.loads(data_json) if data_json.strip() else {}
        except ValueError:
            data = {"raw": data_json}
        if not isinstance(data, dict):
            data = {"value": data}
        session.record(Checkpoint(label=note, data=data))
        return "Progress saved."

    return [memory_recall, memory_remember, memory_note_fact, memory_avoid, memory_save_progress]


__all__ = ["build_agent_tools"]
