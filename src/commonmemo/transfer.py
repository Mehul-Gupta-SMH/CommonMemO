"""Export and import the Log (#164): JSON lines, a header, one event per line, a closing line with a
count and SHA-256 of the event lines. Import checks them before writing anything, skips events the
target already holds (same id, so a repeat or a partial overlap is safe), and rebuilds projections.
Works between any two stores (SQLite file to Postgres and back)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Union

from .events import event_from_dict
from .stores.base import SessionRecord

if TYPE_CHECKING:
    from .memory import Memory

FORMAT = 1
BATCH = 500


def export_log(mem: "Memory", path: Union[str, Path]) -> int:
    """Write every event, oldest first, to `path`. Returns the count."""
    digest, n = hashlib.sha256(), 0
    seen = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"commonmemo_export": FORMAT}) + "\n")
        while True:
            batch = mem.store.events_after(seen, limit=1000)
            if not batch:
                break
            for seq, event in batch:
                line = json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False)
                digest.update(line.encode("utf-8") + b"\n")
                f.write(line + "\n")
                n, seen = n + 1, seq
        f.write(json.dumps({"count": n, "sha256": digest.hexdigest()}) + "\n")
    return n


def import_log(mem: "Memory", path: Union[str, Path]) -> tuple[int, int]:
    """Add the events in `path` that `mem` lacks. Returns (added, already_there).
    Raises ValueError, writing nothing, when the file is not an export or fails its checksum."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        head, tail, body = json.loads(lines[0]), json.loads(lines[-1]), lines[1:-1]
    except (IndexError, ValueError):
        raise ValueError("commonmemo: not an export file (unreadable header or trailer)") from None
    if head.get("commonmemo_export") != FORMAT:
        raise ValueError(f"commonmemo: unsupported export format {head.get('commonmemo_export')!r}")
    digest = hashlib.sha256()
    for line in body:
        digest.update(line.encode("utf-8") + b"\n")
    if tail.get("count") != len(body) or tail.get("sha256") != digest.hexdigest():
        raise ValueError("commonmemo: export is truncated or was changed (count or checksum does not match)")
    events = [event_from_dict(json.loads(line)) for line in body]
    added = 0
    for i in range(0, len(events), BATCH):
        added += sum(mem.store.append_many(events[i:i + BATCH]))
    # The session registry is not in the Log: register the sessions the events name, so forget_user finds them.
    first: dict = {}
    for e in events:
        if e.session_id and e.session_id not in first:
            first[e.session_id] = e
    for sid, e in first.items():
        if mem.store.get_session(sid) is None:
            mem.store.open_session(SessionRecord(session_id=sid, user_id=e.user_id, agent=e.agent, opened_at=e.ts))
    mem.knowledge.catch_up(mem.store)
    return added, len(events) - added


__all__ = ["export_log", "import_log"]
