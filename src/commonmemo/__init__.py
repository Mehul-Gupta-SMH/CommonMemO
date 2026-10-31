"""CommonMemo -- framework-neutral memory for agents and workflows.

Four tiers behind one `Memory` handle, plus structured kinds (tool memory,
world facts, feedback, negatives, resume points) kept beside them:

- **Log**         -- everything that happened, in order, never edited (owned by CommonMemo)
- **Working**     -- what the current conversation or run has in hand (owned by CommonMemo)
- **Recaps**      -- short summaries of finished sessions (delegated to an engine)
- **UserProfile** -- durable facts about one user (delegated to an engine)

See docs/design.md for the full design.
"""

from __future__ import annotations

TYPE_CHECKING = False  # not `from typing import TYPE_CHECKING`: importing typing costs 7 ms a hook (#92)

if TYPE_CHECKING:
    from .context import MemoryContext, MemoryPolicy
    from .engines import Engine, Fact, LocalEngine, Recap
    from .events import (
        Checkpoint,
        ExternalEvent,
        Feedback,
        Forgotten,
        MemoryEvent,
        Message,
        Negative,
        ProcedureEntry,
        ProcedureFollowed,
        ToolInvoked,
        UserFact,
        WorldFact,
        event_from_dict,
    )
    from .environment import EnvironmentResult
    from .facts import FactKey
    from .ranking import EmbeddingRanker, KeywordRanker
    from .history import Diff, History, MemoryView
    from .extraction import ExtractedFact, Extractor, ModelExtractor
    from .learning import Learner, ModelLearner
    from .memory import Memory
    from .procedures import TaskProcedures
    from .reins import Alert, AlertPolicy, Budget, RevertResult
    from .resume import ResumePoint, StepInDoubt, StepInProgress
    from .runs import Run, current_run
    from .session import Session, current_session
    from .stores import SQLiteStore, Store
    from .tool_memory import ToolProfile


# Importing the package must stay cheap: a hook process imports `commonmemo.hookclient` and nothing
# else (#92), so the public names load on first use (PEP 562).
_EXPORTS = {
    "MemoryContext": ".context",
    "MemoryPolicy": ".context",
    "Engine": ".engines",
    "Fact": ".engines",
    "LocalEngine": ".engines",
    "Recap": ".engines",
    "Checkpoint": ".events",
    "ExternalEvent": ".events",
    "Feedback": ".events",
    "Forgotten": ".events",
    "MemoryEvent": ".events",
    "Message": ".events",
    "Negative": ".events",
    "ProcedureEntry": ".events",
    "ProcedureFollowed": ".events",
    "ToolInvoked": ".events",
    "UserFact": ".events",
    "WorldFact": ".events",
    "event_from_dict": ".events",
    "EnvironmentResult": ".environment",
    "FactKey": ".facts",
    "EmbeddingRanker": ".ranking",
    "KeywordRanker": ".ranking",
    "Diff": ".history",
    "History": ".history",
    "MemoryView": ".history",
    "ExtractedFact": ".extraction",
    "Extractor": ".extraction",
    "ModelExtractor": ".extraction",
    "Learner": ".learning",
    "ModelLearner": ".learning",
    "Memory": ".memory",
    "TaskProcedures": ".procedures",
    "Alert": ".reins",
    "AlertPolicy": ".reins",
    "Budget": ".reins",
    "RevertResult": ".reins",
    "ResumePoint": ".resume",
    "StepInDoubt": ".resume",
    "StepInProgress": ".resume",
    "Run": ".runs",
    "current_run": ".runs",
    "Session": ".session",
    "current_session": ".session",
    "SQLiteStore": ".stores",
    "Store": ".stores",
    "ToolProfile": ".tool_memory",
    "instrument": ".adapters.auto",
    "uninstrument": ".adapters.auto",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'commonmemo' has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__version__ = "0.0.2"

__all__ = [
    "instrument",
    "uninstrument",
    "Memory",
    "Session",
    "current_session",
    "MemoryContext",
    "MemoryPolicy",
    "EnvironmentResult",
    "FactKey",
    "EmbeddingRanker",
    "KeywordRanker",
    "WorldFact",
    "Feedback",
    "Negative",
    "Forgotten",
    "Checkpoint",
    "ProcedureEntry",
    "ProcedureFollowed",
    "TaskProcedures",
    "ExtractedFact",
    "Extractor",
    "Learner",
    "ModelExtractor",
    "ModelLearner",
    "ResumePoint",
    "StepInDoubt",
    "StepInProgress",
    "ToolProfile",
    "History",
    "MemoryView",
    "Diff",
    "Run",
    "Alert",
    "AlertPolicy",
    "Budget",
    "RevertResult",
    "current_run",
    "MemoryEvent",
    "Message",
    "ToolInvoked",
    "UserFact",
    "ExternalEvent",
    "event_from_dict",
    "Engine",
    "Recap",
    "Fact",
    "LocalEngine",
    "Store",
    "SQLiteStore",
    "__version__",
]
