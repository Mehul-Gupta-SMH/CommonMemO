"""What goes into the Log: `MemoryEvent` and the built-in event types.

An event is an immutable record of something that happened. Events are the
source of truth -- Working memory is derived live from them, and recaps and
UserProfile facts are consolidated from them at session close. Every derived memory
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
    run_id: Optional[str] = None
    """What produced this event: a deploy, an import, a consolidation run, a
    framework run (`mem.run(...)`, `mem.session(run_id=...)`). Lets history
    answer "what did run R change?" and, later, revert by cause."""

    kind: ClassVar[str] = "event"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = cls.__dict__.get("__annotations__", {})
        if "kind" in own and "ClassVar" not in str(own["kind"]):
            # A field named `kind` would replace the discriminator on every
            # instance, so the event would be logged under the field's value
            # and come back from the log as some other kind.
            raise TypeError(
                f"commonmemo: {cls.__qualname__} declares a field named 'kind', which is reserved "
                "for the event type; name the field something else"
            )
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


_OLD_FEEDBACK_KINDS = frozenset({"rating", "correction", "comment"})


def event_from_dict(data: dict[str, Any]) -> MemoryEvent:
    """Inverse of `MemoryEvent.to_dict()`.

    An unknown `kind` (an event type defined in code that is not imported
    in this process) comes back as an `ExternalEvent` carrying the original
    fields, rather than failing -- a stored log must stay readable.
    """
    data = dict(data)
    kind = data.pop("kind", "event")
    if kind in _OLD_FEEDBACK_KINDS and kind not in _registry and "wrong" in data:
        # Logged before `Feedback.kind` was renamed `feedback_kind`: the
        # field's value was written where the event type belongs.
        data["feedback_kind"], kind = kind, Feedback.kind
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

    Recording one is the deterministic path into the UserProfile -- every engine
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
    `(scope, key)` supersedes the earlier one (history is kept) -- unless it
    comes from a less trusted `source` than the current one (see
    `knowledge.source_rank`), in which case it is kept as *disputed*.
    `valid_until` (epoch seconds) marks facts that go stale -- a price, an
    order status: past it, the fact is no longer current and is not pushed.

    `proposed` facts wait for approval (`Memory.confirm_fact`) before anyone
    but the user or session that proposed them sees them. A world fact
    reaches every user in its scope, so one conversation must not be able to
    change what everyone is told. `None` decides by source: agent-written
    and extracted facts are proposals.
    """

    key: str
    value: str
    scope: str = "global"
    valid_until: Optional[float] = None
    source: str = "explicit"
    proposed: Optional[bool] = None
    note: Optional[str] = None
    """Why a proposal waits for review, when the registry says (`facts.py`)."""


class WorldFactConfirmed(MemoryEvent):
    """An operator approved a proposed or disputed world fact (`fact_id`).
    It carries a copy of the fact, so the approval stands on its own even
    after the conversation that proposed it is forgotten."""

    fact_id: str
    key: str
    value: str
    scope: str = "global"
    valid_until: Optional[float] = None


class WorldFactRejected(MemoryEvent):
    """An operator turned down a proposed or disputed world fact."""

    fact_id: str
    reason: Optional[str] = None


class WorldFactRetracted(MemoryEvent):
    """`(scope, key)` is no longer true, and nothing replaces it: "we no
    longer offer refunds". History is kept."""

    key: str
    scope: str = "global"
    reason: Optional[str] = None
    source: str = "explicit"


class Feedback(MemoryEvent):
    """A judgement about the agent's work.

    `feedback_kind` is `rating` (thumbs or a score), `correction` (`wrong` -> `right`:
    "no, I meant..."), or `comment`. `rating` is in [-1, 1] when given.
    `target_event_id` points at what was judged, when known.
    """

    feedback_kind: str = "rating"
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


class ContextPushed(MemoryEvent):
    """A turn trace: what memory pushed into an agent's context, at the start
    of a session (`reason="start"`), of a turn (`"turn"`, `Session.push_turn`)
    or mid-run (`"refresh"`, `Session.refresh`). It bounds refreshes, keeps a
    recap from being pushed twice in one turn, records what each push cost
    (`chars`), and is what `Memory.traces` / `Memory.why` read (#25).

    `items` are references, not text: one dict per pushed item with its
    `kind` ("fact", "world", "negative", "lesson", "procedure", "recap"), its key or id,
    and where it came from (`source`, `events`, `score`). `more` counts what
    was left out (`MemoryContext.more`). `text` is the pushed text itself,
    only with `MemoryPolicy.trace_text`. Traces carry the user's id, so
    `forget_user` removes them."""

    reason: str
    recap_ids: tuple[str, ...] = ()
    chars: int = 0
    items: tuple[dict[str, Any], ...] = ()
    more: dict[str, int] = dataclasses.field(default_factory=dict)
    text: Optional[str] = None


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


class Snapshot(MemoryEvent):
    """A named position in the log (`Memory.snapshot`), for `as_of` and
    `diff`. Nothing is copied: the log up to here is the snapshot."""

    label: str


class Reverted(MemoryEvent):
    """An operator reverted memory (`Memory.revert`). Its own `id` is the
    revert id; the compensating events it appended carry
    `run_id="revert:<id>"`, so a revert can itself be reverted by cause.
    `target` says what was reverted, in words: "to <point>" or "run R"."""

    target: str
    kinds: tuple[str, ...] = ()
    scope: Optional[str] = None
    reason: Optional[str] = None
    changes: int = 0


class Freeze(MemoryEvent):
    """While active, direct world-fact writes in `scope` (any scope when
    None) from `source` (any source when None) are recorded as proposals.
    Ends with `Unfreeze(freeze_id=<this id>)`."""

    scope: Optional[str] = None
    source: Optional[str] = None
    reason: Optional[str] = None


class Unfreeze(MemoryEvent):
    freeze_id: str


class ProcedureEntry(MemoryEvent):
    """How a type of task is done (ADR 0003): one `template` (the shape of a
    good result), `rule` (a constraint it must follow), `procedure` (the
    steps that produce it) or `exemplar` (an accepted result, as a
    reference), for `task`.

    `(scope, task, entry_kind, name)` identifies an entry; a later write with
    a different body is its next version. `scope` is `global`,
    `project:<name>` or `user:<id>` (one user's own way of doing it); the
    more specific wins. `steps` names a procedure's steps (`@mem.step` names,
    when they are backed by steps). Like a world fact, an entry from a less
    trusted `source` never replaces a more trusted one, and `proposed`
    entries wait for `Memory.confirm_procedure`; `None` decides by source."""

    task: str
    entry_kind: str
    name: str
    body: str
    use_when: str = ""
    """One line saying when it applies; requests are matched against it."""
    steps: tuple[str, ...] = ()
    scope: str = "global"
    source: str = "explicit"
    proposed: Optional[bool] = None
    note: Optional[str] = None


class ProcedureConfirmed(MemoryEvent):
    """An operator approved a proposed entry (`entry_id`), into `scope`. It
    carries a copy of the entry (scrubbed, for an exemplar shared beyond its
    user), so the approval stands after the original is forgotten."""

    entry_id: str
    task: str
    entry_kind: str
    name: str
    body: str
    use_when: str = ""
    steps: tuple[str, ...] = ()
    scope: str = "global"


class ProcedureRejected(MemoryEvent):
    entry_id: str
    reason: Optional[str] = None


class ProcedureRetired(MemoryEvent):
    """`(scope, task, entry_kind, name)` no longer applies. History is kept."""

    task: str
    entry_kind: str
    name: str
    scope: str = "global"
    reason: Optional[str] = None
    source: str = "explicit"


class ProcedureFollowed(MemoryEvent):
    """A session used an entry (`entry_id`, at `version`): pushed, pulled, or
    stated by the app. `skipped` lists a procedure's steps the session never
    finished, when its steps are backed by `@mem.step` (`steps_tracked`).
    With feedback in the same session, this gives per-version outcomes."""

    entry_id: str
    task: str
    entry_kind: str
    name: str
    version: int
    skipped: tuple[str, ...] = ()
    steps_tracked: bool = False


class DecisionRecorded(MemoryEvent):
    """A decision a session made, distilled at close by a learner (#26):
    the goal, the plan, the approach chosen and the ones rejected, and how
    it turned out. Never the model's raw reasoning: only what the session's
    events show."""

    goal: str
    chosen: str
    plan: str = ""
    rejected: tuple[str, ...] = ()
    outcome: str = ""
    task: Optional[str] = None


class LessonLearned(MemoryEvent):
    """A lesson distilled from a session ("check the timezone before
    scheduling"), for the user it came from (`scope` is `user:<id>`).
    `same_as` names the lesson it repeats, decided at write time: the two
    are merged, and the merged one counts the sessions it was seen in."""

    text: str
    scope: str
    task: Optional[str] = None
    same_as: Optional[str] = None
    source: str = "extracted"


class ExternalEvent(MemoryEvent):
    """An event from another system, kept verbatim (`source` names it,
    e.g. `commonadk.tool_call`). Integrations translate into this."""

    source: str
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


BOOKKEEPING: tuple[type[MemoryEvent], ...] = (
    Checkpoint, ContextPushed, StepStarted, StepFinished, StepFailed, NegativeRevoked, Forgotten, ExternalEvent,
    WorldFactConfirmed, WorldFactRejected, WorldFactRetracted, Snapshot, Reverted, Freeze, Unfreeze,
    ProcedureEntry, ProcedureConfirmed, ProcedureRejected, ProcedureRetired, ProcedureFollowed,
    DecisionRecorded, LessonLearned,
)
"""Events that record mechanics, not content. They stay in the event log
but are left out of episode text and of what engines extract facts from --
a forgotten fact must not be re-learned from its own tombstone."""


def describe(event: MemoryEvent) -> Optional[str]:
    """One plain-language line for the structured kinds, from the user's
    point of view; None for kinds without one."""
    if isinstance(event, Feedback):
        if event.feedback_kind == "correction":
            return f"No, not '{event.wrong}' but '{event.right}'." + (f" {event.comment}" if event.comment else "")
        mood = {1: "That was helpful.", -1: "That was not helpful."}.get(
            None if event.rating is None else (1 if event.rating > 0 else -1 if event.rating < 0 else 0), ""
        )
        return " ".join(p for p in (mood, event.comment or "") if p) or None
    if isinstance(event, Negative):
        return f"Never {event.statement}." + (f" ({event.reason})" if event.reason else "")
    if isinstance(event, WorldFact):
        if event.proposed:
            return None  # an unapproved claim: engines must not learn it as stated fact
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
    "WorldFactConfirmed",
    "WorldFactRejected",
    "WorldFactRetracted",
    "Snapshot",
    "Reverted",
    "Freeze",
    "Unfreeze",
    "Feedback",
    "Negative",
    "NegativeRevoked",
    "Forgotten",
    "Checkpoint",
    "ContextPushed",
    "ProcedureEntry",
    "ProcedureConfirmed",
    "ProcedureRejected",
    "ProcedureRetired",
    "ProcedureFollowed",
    "DecisionRecorded",
    "LessonLearned",
    "StepStarted",
    "StepFinished",
    "StepFailed",
    "ExternalEvent",
    "event_from_dict",
]
