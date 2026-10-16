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


class ExternalEvent(MemoryEvent):
    """An event from another system, kept verbatim (`source` names it,
    e.g. `commonadk.tool_call`). Integrations translate into this."""

    source: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


__all__ = [
    "MemoryEvent",
    "Message",
    "ToolInvoked",
    "UserFact",
    "ExternalEvent",
    "event_from_dict",
]
