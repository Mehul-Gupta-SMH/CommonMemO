"""Runs: tag every event written inside one with its `run_id`.

    with mem.run("deploy-2026-10-17"):        # or: async with mem.run(...)
        mem.learn("refund_window", "30 days")   # run_id="deploy-2026-10-17"

A run is whatever produced a batch of memory: a deploy, an import, a
consolidation pass, one framework run. History uses it to answer "what did
run R change?" (`Memory.history()`), and reverting by cause will use it
(ADR 0004). An event that already carries a `run_id` keeps it.
"""

from __future__ import annotations

import contextvars
import uuid
from typing import Any, Optional

_current: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("commonmemo_current_run", default=None)


def current_run() -> Optional[str]:
    """The `run_id` of the innermost active run, if any."""
    return _current.get()


class Run:
    """Context manager from `Memory.run`; usable with `with` and `async with`."""

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "Run":
        self._token = _current.set(self.run_id)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    async def __aenter__(self) -> "Run":
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__()


__all__ = ["Run", "current_run"]
