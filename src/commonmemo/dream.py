"""Dream: extract from the sessions closed since the last pass (#159).

A session's facts are extracted at close only when the memory was built with an
extractor, and a harness that never set one up (a hook per tool call, no code of
yours) has closed sessions nobody read. `mem.dream()` is the batch version, run by
a person: it walks the sessions closed since it last ran, across harnesses, and
extracts each once with the same `ModelExtractor` the close path uses. It adds no
extraction logic of its own, so the redaction of forgotten values and the facts
it drops are the extractor's.

    report = mem.dream(complete=complete, budget=0.25)             # a dry run: writes nothing
    report = mem.dream(complete=complete, budget=0.25, apply=True)  # writes
    mem.revert(run_id=report.run_id)                                # takes it all back

- **Dry run by default.** It asks the model nothing and writes nothing: it plans the
  sessions, prices the prompts it would send, and says what stops it. `apply=True`
  writes. Opt-in: nothing calls it, there is no timer, cron or background process.
- **One run id.** Every write is `run_id="dream:<timestamp>"`, so the pass reverts as a
  unit. A user's facts are otherwise outside `revert` (a profile is theirs); a `dream:`
  run id is the exception, because it is the one unattended writer of them.
- **Once per session.** Each processed session gets a `Checkpoint(label="dream")` in the
  Log, bound to its user (so `forget_user` removes it) and to no session, whether or not
  the model found a fact in it. A re-run skips those, and the sessions the close path
  already read: an `extract:<session>` fact, or the `Checkpoint(label="extract")` it leaves
  when it found nothing. So a second pass does only new work. (A session whose close-time
  extraction found nothing before that mark existed has no trace of it, and is read once.) The marker is the Log's, not the pass's: a pass that
  stops on its budget leaves the rest to the next one, and `since` is a log position that
  narrows the walk further.
- **A spend cap stops the run and says where.** Before each session the next prompt's
  estimated cost is added to what was spent; past `budget` the pass stops, and the report
  names the position and the session it stopped before. The figure is an estimate (about 4
  characters a token, at the prices you pass); a `complete` that carries a running `.cost`
  in dollars is trusted over it, and the report prints both.
- **Not a rewrite of the past.** A batch pass over an old session must not replace what a
  newer session said, so a fact for a key whose current value was stated after the session
  closed is dropped. A value the user forgot is not extracted again (the extractor's rule).
- **The profile follows.** The local engine builds a user's profile when it ingests a
  session, so a pass re-ingests each session it wrote to. Other engines extract facts
  themselves and are refused.
- **No velocity alert.** A pass is a bulk write a person asked for, so its facts are left out
  of the per-source velocity alert (`reins.check_alerts`); the report says how many facts the
  pass wrote and the run id that takes them back. Every other writer is counted as before.

Only facts are extracted: no lessons, procedures or review items (the learner is a later
step), so nothing here is queued for review.
"""

from __future__ import annotations

import copy
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from .events import Checkpoint, UserFact
from .extraction import DONE_LABEL, ExtractedFact, Extractor, ModelExtractor, Sitting, extract
from .learning import episode

if TYPE_CHECKING:
    from .memory import Memory

log = logging.getLogger("commonmemo")

PREFIX = "dream:"
LABEL = "dream"
CHARS_PER_TOKEN = 4.0
"""Why the cost is an estimate: no tokenizer is imported for a batch command."""
OUT_TOKENS_PER_CALL = 300
"""A reply is a short JSON list; this is only the dry run's guess, and the metered run counts the real one."""
PRICE_IN = 1.0
PRICE_OUT = 5.0
"""Dollars per million tokens, Haiku's list prices; the CLI passes its own."""


@dataclass
class Planned:
    session_id: str
    user_id: str
    position: int
    """The log position of the session's last event."""
    calls: int
    chars: int
    cost: float
    """Estimated dollars for this session, at the prices given."""


@dataclass
class Dream:
    run_id: str
    applied: bool
    since: int
    budget: Optional[float]
    planned: list[Planned] = field(default_factory=list)
    """The sessions this pass would do (dry run) or did, in log order."""
    facts: dict[str, list[ExtractedFact]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    """Sessions not considered, by reason: open, no user, extracted at close, already dreamt, nothing to read."""
    stopped_before: Optional[Planned] = None
    """The first session the budget kept it from, or None when the pass finished."""
    position: int = 0
    """Where the pass got to: the last position it processed (`since` if none)."""
    estimated: float = 0.0
    spent: float = 0.0
    """Dollars metered over the prompts and replies of this pass (0 in a dry run)."""
    reported: Optional[float] = None
    """What the model function itself reported (`complete.cost`), when it does."""
    calls: int = 0

    @property
    def count(self) -> int:
        return sum(len(v) for v in self.facts.values())

    def lines(self) -> list[str]:
        done = len(self.facts) + len(self.failed) if self.applied else 0
        head = (f"dream {self.run_id}: " + (f"{done} of {len(self.planned)} session(s) processed" if self.applied
                else f"dry run, {len(self.planned)} session(s) would be read; nothing was written")
                + (f", {self.count} fact(s) {'written' if self.applied else 'proposed'}" if self.applied or self.facts else ""))
        out = [head]
        for p in self.planned:
            got = self.facts.get(p.session_id)
            note = ("" if got is None else f", {len(got)} fact(s)") + (f", FAILED: {self.failed[p.session_id]}"
                                                                         if p.session_id in self.failed else "")
            out.append(f"  {p.session_id}  user {p.user_id}  position {p.position}  {p.calls} call(s)  "
                       f"~${p.cost:.4f}{note}")
            for f in got or []:
                out.append(f"      {f.key}: {f.value}")
        out.append(f"estimated cost ${self.estimated:.4f}" + (f" of a ${self.budget:.4f} budget" if self.budget is not None else ""))
        if self.applied and self.count:
            # In place of a velocity alert, which a pass does not raise: the size of the bulk write and its undo.
            out.append(f"wrote {self.count} fact(s) in one batch; mem.revert(run_id=\"{self.run_id}\") takes them back")
        if self.applied:
            out.append(f"metered ${self.spent:.4f} over {self.calls} call(s)"
                       + (f"; the model function reported ${self.reported:.4f}" if self.reported is not None else ""))
        if self.stopped_before is not None:
            s = self.stopped_before
            out.append(f"STOPPED by the budget before {s.session_id} (position {s.position}); "
                       f"last processed position {self.position}; run again, or --since {s.position}")
        why = ", ".join(f"{n} {k}" for k, n in sorted(self.skipped.items()) if n)
        if why:
            out.append(f"not read: {why}")
        if not self.applied and self.planned:
            out.append("this was a dry run; --apply writes")
        return out


class _Meter:
    """Counts what goes through a model function, sync or async, without changing it."""

    def __init__(self, complete: Callable) -> None:
        self.complete = complete
        self.chars_in = self.chars_out = self.calls = 0

    def __call__(self, prompt: str):
        self.calls += 1
        self.chars_in += len(prompt)
        out = self.complete(prompt)
        if inspect.isawaitable(out):
            async def counted():
                text = await out
                self.chars_out += len(str(text))
                return text

            return counted()
        self.chars_out += len(str(out))
        return out

    def dollars(self, price_in: float, price_out: float) -> float:
        return (self.chars_in / CHARS_PER_TOKEN * price_in + self.chars_out / CHARS_PER_TOKEN * price_out) / 1e6

    @property
    def cost(self) -> Optional[float]:
        value = getattr(self.complete, "cost", None)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def done_sessions(memory: "Memory") -> set[str]:
    """Sessions a dream pass has processed (its Checkpoints) or the close path read: a fact it wrote, or
    the mark it leaves when it wrote none."""
    done = {e.data.get("session") for e in memory.store.events(kinds=[Checkpoint.kind])
            if e.label == LABEL or (e.label == DONE_LABEL and (e.run_id or "").startswith("extract:"))}
    done |= {e.session_id for e in memory.store.events(kinds=[UserFact.kind])
             if e.run_id and e.run_id.startswith("extract:") and e.session_id}
    return {s for s in done if s}


def _closed_since(memory: "Memory", since: int) -> dict[str, int]:
    """Session id -> the position of its last event, for every session with an event at or after `since`."""
    last: dict[str, int] = {}
    seq = max(since - 1, 0)
    while True:
        page = memory.store.events_after(seq, limit=5000)
        if not page:
            return last
        for pos, e in page:
            if e.session_id:
                last[e.session_id] = pos
        seq = page[-1][0]


def _estimate(extractor: Extractor, memory: "Memory", session_id: str) -> Optional[Planned]:
    ep = episode(memory, session_id)
    if ep is None or ep.user_id is None:
        return None
    record = memory.store.get_session(session_id)
    date = time.strftime("%Y-%m-%d", time.localtime(record.opened_at)) if record and record.opened_at else ""
    sitting = Sitting(ep, {}, date)
    if isinstance(extractor, ModelExtractor):
        chars = [len(extractor.prompt(sitting, c, {})) for c in extractor.chunks(ep.lines)]
    else:
        chars = [sum(len(line) + 1 for line in ep.lines)]
    return Planned(session_id, ep.user_id, 0, len(chars), sum(chars), 0.0)


def _price(p: Planned, price_in: float, price_out: float) -> float:
    return (p.chars / CHARS_PER_TOKEN * price_in + p.calls * OUT_TOKENS_PER_CALL * price_out) / 1e6


class ClaudeComplete:
    """A model function over the headless `claude -p`, for the CLI: one turn, no tools, and no settings (so
    no hooks, which would log this call as a session of its own). `cost` is the running total of the dollars
    the CLI says it billed, which `dream` trusts over its own estimate."""

    def __init__(self, model: str = "haiku", *, timeout: float = 300.0) -> None:
        self.model, self.timeout, self.cost, self.calls = model, timeout, 0.0, 0

    def __call__(self, prompt: str) -> str:
        import json
        import subprocess
        import tempfile

        cmd = ["claude", "-p", "--output-format", "json", "--max-turns", "1", "--tools", "", "--no-session-persistence",
               "--strict-mcp-config", "--disable-slash-commands", "--setting-sources", "", "--model", self.model]
        with tempfile.TemporaryDirectory(prefix="commonmemo-dream-") as cwd:
            proc = subprocess.run(cmd, input=prompt, cwd=cwd, capture_output=True, text=True, timeout=self.timeout,
                                  check=False)
        try:
            result = json.loads(proc.stdout)
        except ValueError:
            result = {}
        self.calls += 1
        self.cost += float(result.get("total_cost_usd") or 0.0)
        if proc.returncode != 0 or not isinstance(result, dict) or result.get("is_error"):
            raise RuntimeError(f"claude -p failed ({proc.returncode}): {(proc.stderr or proc.stdout)[:200]}")
        return str(result.get("result") or "")


async def reingest(memory: "Memory", session_ids: list[str]) -> None:
    """Hand each session's events back to the local engine, which rebuilds that session's facts."""
    for sid in session_ids:
        record = memory.store.get_session(sid)
        if record is not None:
            await memory.engine.ingest(record, memory.store.events(session_id=sid))


async def dream(memory: "Memory", *, since: int = 0, budget: Optional[float] = None, apply: bool = False,
                complete: Optional[Callable] = None, extractor: Optional[Extractor] = None,
                price_in: float = PRICE_IN, price_out: float = PRICE_OUT, now: Optional[float] = None) -> Dream:
    from .engines.local import LocalEngine

    if not isinstance(memory.engine, LocalEngine):
        raise ValueError("commonmemo: dream rebuilds the local engine's profile; the other engines extract facts themselves")
    if extractor is None:
        extractor = ModelExtractor(complete) if complete is not None else memory.extractor
    if extractor is None:
        raise ValueError("commonmemo: dream needs a model: pass complete= or extractor=, or build Memory(extractor=...)")
    meter = None
    if isinstance(extractor, ModelExtractor):
        extractor = copy.copy(extractor)
        meter = _Meter(extractor.complete)
        extractor.complete = meter

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now if now is not None else time.time()))
    report = Dream(f"{PREFIX}{stamp}", apply, since, budget, position=since)
    skip = done_sessions(memory)

    queue: list[Planned] = []
    for sid, pos in sorted(_closed_since(memory, since).items(), key=lambda kv: kv[1]):
        record = memory.store.get_session(sid)
        reason = ("open" if record is not None and record.is_open else
                  "no user" if record is None or record.user_id is None else
                  "already dreamt or extracted" if sid in skip else None)
        if reason:
            if record is not None:  # a session id with only a stray event is no session
                report.skipped[reason] = report.skipped.get(reason, 0) + 1
            continue
        planned = _estimate(extractor, memory, sid)
        if planned is None:
            report.skipped["nothing to read"] = report.skipped.get("nothing to read", 0) + 1
            continue
        planned.position, planned.cost = pos, _price(planned, price_in, price_out)
        queue.append(planned)

    spent = 0.0
    for planned in queue:
        if budget is not None and spent + planned.cost > budget:
            report.stopped_before = planned
            break
        spent += planned.cost
        report.planned.append(planned)
        report.estimated += planned.cost
        if not apply:
            continue
        record = memory.store.get_session(planned.session_id)
        try:
            facts = await extract(memory, planned.session_id, extractor=extractor, force=True, run_id=report.run_id,
                                  not_before=record.closed_at if record else None)
        except Exception as e:  # noqa: BLE001 -- one bad session must not cost the others; it is retried next pass
            log.exception("commonmemo: dream failed on %s", planned.session_id)
            report.failed[planned.session_id] = f"{type(e).__name__}: {e}"[:200]
            continue
        report.facts[planned.session_id] = facts
        report.position = planned.position
        if facts:
            await reingest(memory, [planned.session_id])
        memory._append(Checkpoint(label=LABEL, data={"session": planned.session_id, "position": planned.position,
                                                     "facts": len(facts)}).bind(user_id=planned.user_id,
                                                                                 run_id=report.run_id))
        if meter is not None:
            spent = meter.cost if meter.cost is not None else meter.dollars(price_in, price_out)
    if meter is not None and apply:
        report.calls, report.spent, report.reported = meter.calls, meter.dollars(price_in, price_out), meter.cost
    if not apply:
        report.position = since
    return report


__all__ = ["ClaudeComplete", "Dream", "Planned", "dream", "done_sessions", "reingest", "PREFIX", "LABEL"]
