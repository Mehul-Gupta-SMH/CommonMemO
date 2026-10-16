"""The event tier: `MemoryEvent` and the built-in event types.

An event is an immutable record of something that happened. Events are the
source of truth -- session state is derived live from them, and episodic and
user memory are consolidated from them at session close. Every derived memory
keeps the ids of the events it came from (provenance).

Defining your own event is subclassing, nothing more::

    class OrderPlaced(MemoryEvent):
        order_id: str
        amount: float

Every subclass is turned into a frozen, keyword-only dataclass automatically
(`__init_subclass__`), so there is no `@dataclass` to remember. Its `kind`
defaults to the snake_case class name (`order_placed`); set `kind = "..."`
as a plain class attribute to override it.

This module has no third-party import: constructing, serializing and
round-tripping events never needs any engine or framework installed.
"""

from __future__ import annotations

import dataclasses
import re
import time
import uuid
from typing import Any, ClassVar, Optional

_registry: dict[str, type["MemoryEvent"]] = {}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@dataclasses.dataclass(frozen=True, kw_only=True)
class MemoryEvent:
    """Base of every event. Subclass it to define a domain event."""

    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = dataclasses.field(default_factory=time.time)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent: Optional[str] = None

    kind: ClassVar[str] = "event"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "kind" not in cls.__dict__:
            cls.kind = _snake(cls.__name__)
        if cls.kind in _registry and _registry[cls.kind].__qualname__ != cls.__qualname__:
            raise TypeError(
                f"commonmemo: event kind {cls.kind!r} is already registered by "
                f"{_registry[cls.kind].__qualname__}; set a distinct `kind` on "
                f"{cls.__qualname__}"
            )
        # Applying the dataclass decorator here (rather than asking every
        # subclass to) is what makes "just subclass it" the whole contract.
        dataclasses.dataclass(frozen=True, kw_only=True)(cls)
        _registry[cls.kind] = cls

    @property
    def payload(self) -> dict[str, Any]:
        """The subclass's own fields -- everything but the common envelope."""
        envelope = {f.name for f in dataclasses.fields(MemoryEvent)}
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in envelope
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict, with `kind` as the discriminator."""
        d = dataclasses.asdict(self)
        d["kind"] = self.kind
        return d

    def bind(self, **envelope: Any) -> "MemoryEvent":
        """A copy with envelope fields (session_id, user_id, agent) filled in
        where they are still unset -- never overwriting an explicit value."""
        changes = {k: v for k, v in envelope.items() if getattr(self, k) is None and v is not None}
        return dataclasses.replace(self, **changes) if changes else self


def event_from_dict(data: dict[str, Any]) -> MemoryEvent:
    """Inverse of `MemoryEvent.to_dict()`.

    An unknown `kind` (an event type defined in code that is not imported
    in this process) comes back as an `ExternalEvent` carrying the original
    fields, rather than failing -- a stored log must stay readable.
    """
    data = dict(data)
    kind = data.pop("kind", "event")
    cls = _registry.get(kind)
    names = {f.name for f in dataclasses.fields(cls)} if cls else set()
    if cls is None or not set(data) <= names:
        envelope = {f.name for f in dataclasses.fields(MemoryEvent)}
        return ExternalEvent(
            source=kind,
            data={k: v for k, v in data.items() if k not in envelope},
            **{k: v for k, v in data.items() if k in envelope},
        )
    return cls(**data)


class Message(MemoryEvent):
    """A conversational turn: `role` is `user`, `assistant` or `system`."""

    role: str
    content: str


class ToolInvoked(MemoryEvent):
    """One call of a function wrapped by `Memory.track`."""

    name: str
    arguments: dict[str, Any] = dataclasses.field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


class UserFact(MemoryEvent):
    """An explicit, durable fact about a user (`key` -> `value`).

    Recording one is the deterministic path into user memory -- every engine
    must honor it. Engines that extract facts with an LLM add to this; they
    never replace it. A later fact with the same `key` supersedes an earlier
    one rather than accumulating next to it.
    """

    key: str
    value: str
    source: str = "explicit"
    """Who stated it: `explicit` (the application), `agent` (an agent's
    `memory_remember` tool), or an integration's name."""


class WorldFact(MemoryEvent):
    """A fact about the world or a project rather than a user: "the refund
    window is 30 days", "this repo targets Python 3.10".

    `scope` is `global` or `project:<name>`. A later fact with the same
    `(scope, key)` supersedes the earlier one (history is kept).
    `valid_until` (epoch seconds) marks facts that go stale -- a price, an
    order status: past it, the fact is no longer current and is not pushed.
    """

    key: str
    value: str
    scope: str = "global"
    valid_until: Optional[float] = None
    source: str = "explicit"


class Feedback(MemoryEvent):
    """A judgement about the agent's work.

    `kind` is `rating` (thumbs or a score), `correction` (`wrong` -> `right`:
    "no, I meant..."), or `comment`. `rating` is in [-1, 1] when given.
    `target_event_id` points at what was judged, when known.
    """

    kind: str = "rating"
    rating: Optional[float] = None
    comment: Optional[str] = None
    wrong: Optional[str] = None
    right: Optional[str] = None
    target_event_id: Optional[str] = None


class Negative(MemoryEvent):
    """A standing "never / avoid" rule: "never suggest phone calls".

    `scope` is `user:<id>`, `agent:<name>`, `project:<name>` or `global`.
    Negatives are pushed ahead of everything else and are never ranked out.
    """

    statement: str
    scope: str = "global"
    reason: Optional[str] = None
    source: str = "explicit"


class NegativeRevoked(MemoryEvent):
    """Withdraws a `Negative` by its event id."""

    negative_id: str


class Forgotten(MemoryEvent):
    """A tombstone: "forget my old address". Removes the user's fact with this
    `key` and keeps it from coming back -- an engine that re-extracts the
    same thing from old conversations is filtered against tombstones.
    `pattern`, when given, also hides extracted facts containing it."""

    key: str
    pattern: Optional[str] = None


class Checkpoint(MemoryEvent):
    """An explicit restart point: what a process needs to continue from here.
    `data` must be JSON-serializable."""

    label: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


class StepStarted(MemoryEvent):
    """`@mem.step` began running `step_id`."""

    step_id: str
    side_effects: bool = False


class StepFinished(MemoryEvent):
    """`step_id` completed; `result` is its JSON-serialized return value
    (`replayable` is False when the value was not JSON-serializable)."""

    step_id: str
    result: Optional[str] = None
    replayable: bool = True


class StepFailed(MemoryEvent):
    """`step_id` raised; a resumed run tries it again."""

    step_id: str
    error: str


class ExternalEvent(MemoryEvent):
    """An event from another system, kept verbatim (`source` names it,
    e.g. `commonadk.tool_call`). Integrations translate into this."""

    source: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


BOOKKEEPING: tuple[type[MemoryEvent], ...] = (
    Checkpoint, StepStarted, StepFinished, StepFailed, NegativeRevoked, Forgotten, ExternalEvent,
)
"""Events that record mechanics, not content. They stay in the event log
but are left out of episode text and of what engines extract facts from --
a forgotten fact must not be re-learned from its own tombstone."""


def describe(event: MemoryEvent) -> Optional[str]:
    """One plain-language line for the structured kinds, from the user's
    point of view; None for kinds without one."""
    if isinstance(event, Feedback):
        if event.kind == "correction":
            return f"No, not '{event.wrong}' but '{event.right}'." + (f" {event.comment}" if event.comment else "")
        mood = {1: "That was helpful.", -1: "That was not helpful."}.get(
            None if event.rating is None else (1 if event.rating > 0 else -1 if event.rating < 0 else 0), ""
        )
        return " ".join(p for p in (mood, event.comment or "") if p) or None
    if isinstance(event, Negative):
        return f"Never {event.statement}." + (f" ({event.reason})" if event.reason else "")
    if isinstance(event, WorldFact):
        return f"Note: {event.key} is {event.value}."
    return None


__all__ = [
    "BOOKKEEPING",
    "describe",
    "MemoryEvent",
    "Message",
    "ToolInvoked",
    "UserFact",
    "WorldFact",
    "Feedback",
    "Negative",
    "NegativeRevoked",
    "Forgotten",
    "Checkpoint",
    "StepStarted",
    "StepFinished",
    "StepFailed",
    "ExternalEvent",
    "event_from_dict",
]
