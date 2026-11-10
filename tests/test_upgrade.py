"""A Log written by each released version opens and replays under the current code (#165).
Fixtures are made by tests/fixtures/upgrade/make.py run with the released package."""
import shutil
from pathlib import Path

import pytest

from commonmemo import Memory
from commonmemo.events import Message

pytestmark = pytest.mark.sqlite_only  # the old releases only wrote SQLite files

FIXTURES = Path(__file__).parent / "fixtures" / "upgrade"


@pytest.mark.parametrize("version", ["0.0.1", "0.0.2", "0.0.3"])
def test_old_log_opens_replays_and_accepts_writes(version, tmp_path):
    db = tmp_path / "old.db"
    shutil.copy(FIXTURES / f"v{version}.db", db)
    mem = Memory(f"sqlite:///{db}")
    said = [e.content for e in mem.store.events() if isinstance(e, Message)]
    assert "My name is Ada and I live in Lisbon." in said
    with mem.session(session_id="s2", user_id="ada") as s:  # a new session on the old Log
        s.message("user", "And I moved to Berlin.")
    again = Memory(f"sqlite:///{db}")  # reopening after the rebuild is stable
    texts = [e.content for e in again.store.events() if isinstance(e, Message)]
    assert texts[-1] == "And I moved to Berlin." and len(texts) == len(said) + 1
