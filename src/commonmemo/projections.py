"""The projections' contract and the records they return (#52).

Projections are everything memory reads that is derived from the Log: tool memory, world facts,
procedures, decisions, feedback, negatives, tombstones, checkpoints and steps, user links, and the
message-window index. `Projections` names them by operation; `knowledge.py` is the SQL
implementation. `tests/test_projections_interface.py` runs the library through the interface alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from .tool_memory import ToolProfile

if TYPE_CHECKING:
    from .stores.base import Store

Row = Mapping[str, Any]
"""A window row: readable as `row["id"]`, `["wid"]`, `["session_id"]`, `["agent"]`, `["first_ts"]`, `["event_ids"]`
(a JSON list), and for a candidate search `["hits"]`."""


@dataclass(frozen=True)
class WorldRecord:
    scope: str
    key: str
    value: str
    valid_from: float
    """When it became current: written, or confirmed. Shown as its age."""
    valid_until: Optional[float]
    source: str
    event_id: str
    status: str = "current"
    """`current`, or for a pending fact `proposed` / `disputed`."""
    user_id: Optional[str] = None
    """Who proposed it, for a pending fact."""
    note: Optional[str] = None

    @property
    def pending(self) -> bool:
        return self.status in PENDING


PENDING = ("proposed", "disputed")

_RANKS = {"extracted": 0, "agent": 1, "tool": 2}


def source_rank(source: str) -> int:
    """How much a world fact's source is trusted: `extracted` 0, `agent` 1,
    `tool` (checked against a tool result) 2, and anything else -- `explicit`,
    `operator`, an integration's name -- 3, since application code wrote it.
    A fact can only supersede a current fact of equal or lower rank."""
    return _RANKS.get(source, 3)


@dataclass(frozen=True)
class ProcedureRecord:
    """One procedural entry (ADR 0003), as the projection holds it."""

    event_id: str
    scope: str
    task: str
    kind: str
    """`template`, `rule`, `procedure` or `exemplar`."""
    name: str
    use_when: str
    body: str
    steps: tuple[str, ...]
    version: int
    """1 for the first body that became current, then +1 per change (0 while pending)."""
    source: str
    status: str
    valid_from: float
    user_id: Optional[str] = None
    note: Optional[str] = None

    @property
    def pending(self) -> bool:
        return self.status in PENDING

    @property
    def key(self) -> str:
        """How traces name it: `task/kind/name`."""
        return f"{self.task}/{self.kind}/{self.name}"


@dataclass
class ProcedureOutcome:
    """How runs that used one version of an entry went (`Memory.procedure_outcomes`)."""

    task: str
    kind: str
    name: str
    version: int
    runs: int = 0
    rated: int = 0
    positive: int = 0
    negative: int = 0
    corrections: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    """Step name -> runs that never finished it (runs with tracked steps only)."""

    @property
    def approval(self) -> Optional[float]:
        return self.positive / self.rated if self.rated else None


@dataclass(frozen=True)
class DecisionRecord:
    event_id: str
    ts: float
    session_id: Optional[str]
    user_id: Optional[str]
    agent: Optional[str]
    task: Optional[str]
    goal: str
    plan: str
    chosen: str
    rejected: tuple[str, ...]
    outcome: str

    def line(self) -> str:
        out = f"{self.goal}: chose {self.chosen}"
        if self.rejected:
            out += f" over {', '.join(self.rejected)}"
        return out + (f"; {self.outcome}" if self.outcome else "")


@dataclass(frozen=True)
class LessonRecord:
    event_id: str
    scope: str
    text: str
    task: Optional[str]
    sessions: int
    """How many sessions taught it (merged repeats)."""
    first_seen: float
    last_seen: float

    def line(self) -> str:
        return self.text + (f" (seen in {self.sessions} sessions)" if self.sessions > 1 else "")


@dataclass(frozen=True)
class FeedbackRecord:
    event_id: str
    ts: float
    user_id: Optional[str]
    agent: Optional[str]
    session_id: Optional[str]
    kind: str
    rating: Optional[float]
    comment: Optional[str]
    wrong: Optional[str]
    right: Optional[str]
    target_event_id: Optional[str]

    def line(self) -> str:
        if self.kind == "correction" and self.wrong and self.right:
            return f"Corrected: not '{self.wrong}' but '{self.right}'" + (f" ({self.comment})" if self.comment else "")
        mood = "" if self.rating is None else ("liked: " if self.rating > 0 else "disliked: " if self.rating < 0 else "")
        return f"{mood}{self.comment}" if self.comment else f"{mood}rating {self.rating}"


@dataclass(frozen=True)
class NegativeRecord:
    event_id: str
    ts: float
    scope: str
    statement: str
    reason: Optional[str]
    source: str


@dataclass(frozen=True)
class FreezeRecord:
    event_id: str
    ts: float
    scope: Optional[str]
    source: Optional[str]
    reason: Optional[str]


@dataclass(frozen=True)
class CheckpointRecord:
    event_id: str
    session_id: str
    ts: float
    label: str
    data: dict[str, Any]


@dataclass(frozen=True)
class StepRecord:
    session_id: str
    step_id: str
    status: str  # "started" | "finished" | "failed"
    result: Optional[str]
    replayable: bool
    error: Optional[str]
    side_effects: bool
    started_at: Optional[float]
    finished_at: Optional[float]


@dataclass
class FeedbackSummary:
    count: int = 0
    rated: int = 0
    positive: int = 0
    negative: int = 0
    corrections: int = 0
    recent: list[FeedbackRecord] = field(default_factory=list)

    @property
    def approval(self) -> Optional[float]:
        return self.positive / self.rated if self.rated else None




class Projections(ABC):
    """Everything memory reads that is derived from the Log, by operation (#52).

    `Knowledge` implements it on SQL (SQLite and Postgres). A backend that is not SQL
    implements these operations with whatever it has (conditional writes, transactions, a
    stream consumer) and passes it to `Memory(projections=...)`. What it must keep true:

    - **A consumer of the log:** it remembers the last log position it applied, `catch_up`
      applies everything after it, and the new position is saved together with the changes
      (atomically), so applying the same stretch twice, or losing one, is impossible.
    - **Rebuildable:** `rebuild` recomputes everything from the whole log; the result equals
      what catching up event by event gave.
    - **Exact kinds:** counted failures, step results, rules and world-fact supersession are
      not approximated.
    - **Forgetting:** `forget_user` removes everything user-specific, windows and vectors included.

    The records it returns are the dataclasses in this module.
    """

    path: str
    """Names where it lives (a file, a URL); `":memory:"` when it lives nowhere."""

    @abstractmethod
    def scratch(self) -> "Projections":
        """A new, empty projection of the same kind that lives only in memory and indexes no
        message windows. History and reverts replay the Log into one to ask what a past
        position looked like."""

    # -- the projections and the log ---------------------------------

    @abstractmethod
    def catch_up(self, store: Store) -> int:
        """Apply every event the log holds past `applied_seq`; return how
        many were applied. With no recorded position (a new file, or after
        `rebuild`), every projection is recomputed from the whole log."""

    @property
    @abstractmethod
    def applied_seq(self) -> Optional[int]:
        """The last log position applied, or None before the first catch-up."""

    @abstractmethod
    def rebuild(self, store: Store) -> None:
        """Recompute every projection from the whole log."""

    @abstractmethod
    def close(self) -> None:
        """See `Knowledge.close`."""

    # -- tool memory -------------------------------------------------

    @abstractmethod
    def tool_profile(self, tool: str, agent: Optional[str] = None) -> ToolProfile:
        """The tool's profile for `agent`, or across all agents when None."""

    @abstractmethod
    def tool_profiles(self, agent: Optional[str] = None) -> dict[str, ToolProfile]:
        """`{tool: profile}` for every tool, in two queries however many tools there are.
        Each equals `tool_profile(tool, agent)`."""

    @abstractmethod
    def tools(self, agent: Optional[str] = None) -> list[str]:
        """See `Knowledge.tools`."""

    # -- world facts -------------------------------------------------

    @abstractmethod
    def world_facts(self, scopes: Sequence[str], *, now: Optional[float] = None, include_stale: bool = False) -> list[WorldRecord]:
        """Current facts in `scopes` (later scopes win on equal keys).
        Facts past `valid_until` are left out unless `include_stale`.
        Pending facts are never included (see `pending_world_facts`)."""

    @abstractmethod
    def pending_world_facts(self, scopes: Optional[Sequence[str]] = None, *, user_id: Optional[str] = None, session_id: Optional[str] = None) -> list[WorldRecord]:
        """Proposed and disputed facts, oldest first: all of them, or only
        those proposed by `user_id` or in `session_id` when either is given."""

    @abstractmethod
    def world_fact(self, event_id: str) -> Optional[WorldRecord]:
        """See `Knowledge.world_fact`."""

    @abstractmethod
    def world_history(self, scope: str, key: str) -> list[tuple[str, float, Optional[float]]]:
        """`(value, valid_from, valid_to)` for every value that was current."""

    @abstractmethod
    def scopes_in_use(self) -> list[str]:
        """Every scope that holds a current world fact or an active negative."""

    @abstractmethod
    def freezes(self) -> list[FreezeRecord]:
        """Active freezes, oldest first."""

    @abstractmethod
    def frozen(self, scope: str, source: str) -> Optional[FreezeRecord]:
        """The active freeze covering a write to `scope` from `source`, if any."""

    # -- procedures, decisions and lessons ---------------------------

    @abstractmethod
    def procedure_entries(self, scopes: Sequence[str], task: Optional[str] = None) -> list[ProcedureRecord]:
        """Current entries in `scopes` (for `task`, or every task), with the
        most specific scope winning per `(task, kind, name)`: later scopes in
        `scopes` are more specific."""

    @abstractmethod
    def procedure_entry(self, event_id: str) -> Optional[ProcedureRecord]:
        """See `Knowledge.procedure_entry`."""

    @abstractmethod
    def pending_procedures(self, task: Optional[str] = None) -> list[ProcedureRecord]:
        """Proposed and disputed entries, oldest first."""

    @abstractmethod
    def procedure_versions(self, scope: str, task: str, kind: str, name: str) -> list[ProcedureRecord]:
        """Every version that was current, oldest first."""

    @abstractmethod
    def procedure_outcomes(self, task: Optional[str] = None) -> list[ProcedureOutcome]:
        """Per entry version: the runs that used it, and the feedback given in
        those runs' sessions."""

    @abstractmethod
    def decisions(self, *, user_id: Optional[str] = None, agent: Optional[str] = None, session_id: Optional[str] = None) -> list[DecisionRecord]:
        """Decision records, newest first."""

    @abstractmethod
    def learned_lessons(self, scopes: Sequence[str]) -> list[LessonRecord]:
        """Lessons in `scopes`, the most often seen first, then the newest."""

    @abstractmethod
    def all_procedures(self) -> list[ProcedureRecord]:
        """Every current entry, in every scope (for reverts)."""

    # -- feedback, negatives, forgetting -----------------------------

    @abstractmethod
    def feedback_summary(self, *, user_id: Optional[str] = None, agent: Optional[str] = None, recent: int = 5) -> FeedbackSummary:
        """See `Knowledge.feedback_summary`."""

    @abstractmethod
    def lessons(self, *, user_id: Optional[str], agent: Optional[str], limit: int = 5) -> list[FeedbackRecord]:
        """Feedback worth acting on next time: corrections, and anything
        negative or commented -- newest first."""

    @abstractmethod
    def negatives(self, scopes: Sequence[str]) -> list[NegativeRecord]:
        """See `Knowledge.negatives`."""

    @abstractmethod
    def all_negatives(self) -> list[NegativeRecord]:
        """Every active negative, in every scope."""

    @abstractmethod
    def tombstones(self, user_id: str) -> list[tuple[str, Optional[str], float]]:
        """`(key, pattern, ts)` for each tombstone of `user_id`."""

    @abstractmethod
    def forget_user(self, user_id: str, session_ids: Sequence[str]) -> int:
        """Remove everything user-specific. Tool memory holds only shapes and
        scrubbed signatures, and current world facts are not about the user,
        so both stay; world facts the user's conversations proposed and nobody
        approved go."""

    # -- checkpoints and steps ---------------------------------------

    @abstractmethod
    def latest_checkpoint(self, session_id: str) -> Optional[CheckpointRecord]:
        """See `Knowledge.latest_checkpoint`."""

    @abstractmethod
    def step(self, session_id: str, step_id: str) -> Optional[StepRecord]:
        """See `Knowledge.step`."""

    @abstractmethod
    def steps(self, session_id: str) -> list[StepRecord]:
        """See `Knowledge.steps`."""

    # -- who is who --------------------------------------------------

    @abstractmethod
    def user_links(self) -> list[tuple[str, str, str, Optional[str]]]:
        """Active links, oldest first: `(link id, user_a, user_b, reason)`."""

    @abstractmethod
    def linked_users(self, user_id: str) -> list[str]:
        """`user_id` and every id reachable from it by active links (one person), `user_id` first, the rest sorted."""

    # -- message windows (said.py) -----------------------------------

    @abstractmethod
    def window_candidates(self, user_id: Union[str, Sequence[str]], query_terms: Sequence[str], *, exclude_session: Optional[str] = None, agent: Optional[str] = None, limit: int = 300) -> list[Row]:
        """The user's windows that share terms with a query, most shared terms first, then newest:
        `id, wid, session_id, agent, first_ts, event_ids, hits`."""

    @abstractmethod
    def recent_windows(self, user_id: Union[str, Sequence[str]], *, exclude_session: Optional[str] = None, agent: Optional[str] = None, limit: int = 400) -> list[Row]:
        """The user's newest windows, for rankers that don't work on shared terms (embeddings)."""

    @abstractmethod
    def window_events(self, window_id: str) -> list[str]:
        """The event ids of the window `window_id` (`<session>#<number>`), or none if it is gone."""

    @abstractmethod
    def window_vectors(self, wids: Sequence[int], redaction: str) -> dict[int, list[float]]:
        """The stored vectors of these windows made from text redacted as `redaction` says."""

    @abstractmethod
    def store_window_vectors(self, rows: Sequence[tuple[int, str, Sequence[float]]]) -> None:
        """See `Knowledge.store_window_vectors`."""

    @abstractmethod
    def vector_probe(self) -> Optional[list[float]]:
        """The embedding of the fixed probe text when the stored vectors were made: if an embedder now
        gives another, they are from another model."""

    @abstractmethod
    def reset_vectors(self, probe: Sequence[float]) -> None:
        """Drop every stored vector and remember `probe` as the embedder's answer for the probe text."""

    @abstractmethod
    def prune_windows(self, store: Store) -> int:
        """Drop windows whose messages `prune` has removed from the Log, so their terms don't outlive
        the text. Returns windows dropped."""


class InterfaceOnly:
    """Shows only what `Projections` declares of the object it wraps, and nothing else.

    A conformance check, not a production class: wrap an implementation, hand it to `Memory(projections=...)`,
    and anything the library reaches for beyond the interface (a private method, a SQL connection) fails with an
    `AttributeError` here instead of working by accident. The test suite can run entirely through it
    (`COMMONMEMO_TEST_INTERFACE=1`). `scratch()` comes back wrapped too."""

    def __init__(self, inner: Projections) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        if name not in _INTERFACE:
            raise AttributeError(f"{name!r} is not part of the Projections interface")
        value = getattr(self._inner, name)  # type: ignore[attr-defined]
        if name == "scratch":
            return lambda: InterfaceOnly(value())
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("projections are read through their operations only")


_INTERFACE = frozenset(Projections.__abstractmethods__) | {"path"}


__all__ = [
    "InterfaceOnly", "Projections", "WorldRecord", "ProcedureRecord", "ProcedureOutcome", "DecisionRecord", "LessonRecord", "FeedbackRecord",
    "NegativeRecord", "FreezeRecord", "CheckpointRecord", "StepRecord", "FeedbackSummary", "PENDING", "source_rank",
]
