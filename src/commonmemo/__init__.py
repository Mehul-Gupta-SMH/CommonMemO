"""CommonMemo -- framework-neutral memory for agents and workflows.

Four tiers behind one `Memory` handle, plus structured kinds (tool memory,
world facts, feedback, negatives, resume points) kept beside them:

- **event**    -- append-only log of what happened (owned by CommonMemo)
- **session**  -- live working state of one conversation/run (owned by CommonMemo)
- **episodic** -- consolidated summaries of past sessions (delegated to an engine)
- **user**     -- durable facts about a user (delegated to an engine)

See docs/design.md for the full design.
"""

from __future__ import annotations

from .context import MemoryContext, MemoryPolicy
from .engines import Engine, Episode, Fact, LocalEngine
from .events import (
    Checkpoint,
    ExternalEvent,
    Feedback,
    Forgotten,
    MemoryEvent,
    Message,
    Negative,
    ToolInvoked,
    UserFact,
    WorldFact,
    event_from_dict,
)
from .memory import Memory
from .resume import ResumePoint, StepInDoubt
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
    "WorldFact",
    "Feedback",
    "Negative",
    "Forgotten",
    "Checkpoint",
    "ResumePoint",
    "StepInDoubt",
    "ToolProfile",
    "MemoryEvent",
    "Message",
    "ToolInvoked",
    "UserFact",
    "ExternalEvent",
    "event_from_dict",
    "Engine",
    "Episode",
    "Fact",
    "LocalEngine",
    "Store",
    "SQLiteStore",
    "__version__",
]
