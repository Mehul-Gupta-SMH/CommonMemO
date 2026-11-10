"""Writes a small Log with the commonmemo that runs this script:
   python make.py OUT.db
Used once per release (0.0.1, 0.0.2, 0.0.3) to make the fixtures `test_upgrade.py` opens."""
import sys

from commonmemo import Memory

mem = Memory(f"sqlite://{sys.argv[1]}"  # relative: 0.0.1 reads sqlite:///x as relative)
with mem.session(session_id="s1", user_id="ada", agent="claude-code") as s:
    s.message("user", "My name is Ada and I live in Lisbon.")
    s.message("assistant", "Nice to meet you, Ada.")
    mem.remember("ada", "city", "Lisbon") if hasattr(mem, "remember") else None
print("ok")
