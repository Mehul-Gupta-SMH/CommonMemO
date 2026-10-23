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
from .learning import Learner, ModelLearner
from .memory import Memory
from .procedures import TaskProcedures
from .reins import Alert, AlertPolicy, Budget, RevertResult
from .resume import ResumePoint, StepInDoubt, StepInProgress
from .runs import Run, current_run
from .session import Session, current_session
from .stores import SQLiteStore, Store
from .tool_memory import ToolProfile

__version__ = "0.0.1"

__all__ = [
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
    "Learner",
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
