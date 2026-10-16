"""CommonMemo -- framework-neutral memory for agents and workflows.

Four tiers behind one `Memory` handle:

- **event**    -- append-only log of what happened (owned by CommonMemo)
- **session**  -- live working state of one conversation/run (owned by CommonMemo)
- **episodic** -- consolidated summaries of past sessions (delegated to an engine)
- **user**     -- durable facts about a user (delegated to an engine)

See docs/design.md for the full design.
"""

from __future__ import annotations

from .context import MemoryContext
from .engines import Engine, Episode, Fact, LocalEngine
from .events import ExternalEvent, MemoryEvent, Message, ToolInvoked, UserFact, event_from_dict
from .memory import Memory
from .session import Session, current_session
from .stores import SQLiteStore, Store

__version__ = "0.0.1"

__all__ = [
    "Memory",
    "Session",
    "current_session",
    "MemoryContext",
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
