"""What memory pushed into a harness's context, against re-sending what was said (#158).

Read-only and computed from the Log: no new events, nothing stored. For every
push at the start of a session or of a turn, the baseline is the text of all
messages said in that session before it, which is what carrying the history over
to the next harness (or the next turn) sends. Pulls, recalls and mid-run
refreshes cost memory characters but have no baseline of their own, so they are
counted as cost only.

Characters, not tokens (`TOKEN_CHARS` characters to a token is an estimate), and
only messages are in the baseline, not tool output, which makes the saving look
smaller than it is, not larger.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

from .events import ContextPushed, Message

if TYPE_CHECKING:
    from .memory import Memory

TOKEN_CHARS = 4
"""Characters to a token, for the estimate."""

BASELINE_REASONS = ("start", "turn")
"""Pushes that replace carrying the history over; the other reasons add cost only."""


@dataclasses.dataclass
class Row:
    """One harness (or one session) in the report."""

    name: str
    pushes: int = 0
    pushed: int = 0
    """Characters memory gave the harness (every reason)."""
    resend: int = 0
    """Characters of history that would have been sent for the start and turn pushes."""

    @property
    def saved(self) -> int:
        return self.resend - self.pushed

    @property
    def ratio(self) -> Optional[float]:
        """Pushed over re-sent, or None with no baseline."""
        return self.pushed / self.resend if self.resend else None

    @property
    def saved_pct(self) -> Optional[float]:
        return 100.0 * (1 - self.ratio) if self.ratio is not None else None

    def tokens(self, chars: int) -> int:
        return round(chars / TOKEN_CHARS)


@dataclasses.dataclass
class Savings:
    total: Row
    by_agent: list[Row]
    by_session: list[Row]
    by_reason: dict[str, int]

    def cost(self, row: Row, price_per_mtok: float) -> tuple[float, float]:
        """(dollars with memory, dollars re-sending), at one input price per million tokens."""
        return (row.tokens(row.pushed) * price_per_mtok / 1e6, row.tokens(row.resend) * price_per_mtok / 1e6)

    def lines(self, price_per_mtok: Optional[float] = None) -> list[str]:
        if not self.total.pushes:
            return ["no pushes recorded: nothing to compare"]
        out = [_line(self.total), ""]
        if len(self.by_agent) > 1 or any(r.name != "-" for r in self.by_agent):
            out += ["by harness:"] + ["  " + _line(r) for r in self.by_agent] + [""]
        out += ["by session:"] + ["  " + _line(r) for r in self.by_session[:20]]
        if len(self.by_session) > 20:
            out.append(f"  ... {len(self.by_session) - 20} more")
        out += ["", "memory characters by reason: " + ", ".join(f"{k} {v}" for k, v in sorted(self.by_reason.items()))]
        if price_per_mtok is not None:
            with_mem, without = self.cost(self.total, price_per_mtok)
            out.append(f"input at ${price_per_mtok:g} per million tokens: ${with_mem:.4f} with memory, ${without:.4f} re-sending")
        out += ["", f"Estimate: {TOKEN_CHARS} characters to a token. The baseline is the messages said earlier in the same session",
                "(not tool output), so it understates what re-sending costs. This measures input size, not answer quality."]
        return out


def _line(r: Row) -> str:
    pct = f"{r.saved_pct:.0f}% smaller" if r.saved_pct is not None else "no baseline"
    return (f"{r.name}: {r.pushes} push{"" if r.pushes == 1 else "es"}, {r.pushed} chars pushed (~{r.tokens(r.pushed)} tokens) "
            f"vs {r.resend} chars re-sent (~{r.tokens(r.resend)} tokens): {pct}")


def savings(memory: "Memory", *, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Savings:
    """The report for one session, one user's sessions, or (with neither) everything."""
    events = memory.store.events(session_id=session_id, user_id=user_id, kinds=[Message.kind, ContextPushed.kind])
    said: dict[Optional[str], int] = {}
    total = Row("all")
    agents: dict[str, Row] = {}
    sessions: dict[str, Row] = {}
    reasons: dict[str, int] = {}
    for e in events:
        if isinstance(e, Message):
            said[e.session_id] = said.get(e.session_id, 0) + len(e.content or "")
            continue
        if not isinstance(e, ContextPushed):
            continue
        base = said.get(e.session_id, 0) if e.reason in BASELINE_REASONS else 0
        reasons[e.reason] = reasons.get(e.reason, 0) + e.chars
        for row in (total, agents.setdefault(e.agent or "-", Row(e.agent or "-")),
                    sessions.setdefault(e.session_id or "-", Row(e.session_id or "-"))):
            row.pushes += 1
            row.pushed += e.chars
            row.resend += base
    return Savings(total, sorted(agents.values(), key=lambda r: r.name), list(sessions.values()), reasons)


__all__ = ["Row", "Savings", "savings", "TOKEN_CHARS"]
