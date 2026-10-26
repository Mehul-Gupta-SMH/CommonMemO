"""What a coding harness (Claude Code, Codex, OpenCode) does with CommonMemo, shown by hand.

A harness runs `commonmemo hook` on each of its events with the event's JSON on stdin. This sends
the same JSON a Claude Code session sends, so you can see the capture and the push without a harness.

    python examples/harness_hooks.py

To wire up a real harness:  commonmemo setup claude-code   (or `codex`)
                            commonmemo setup opencode > .opencode/plugin/commonmemo.js
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

store = Path(tempfile.mkdtemp(prefix="commonmemo-hooks-")) / "memory.db"
env = dict(os.environ, COMMONMEMO_STORE=f"sqlite:///{store}")


def hook(payload: dict) -> str:
    done = subprocess.run([sys.executable, "-m", "commonmemo", "hook", "--harness", "claude-code", "--user", "ada",
                           "--no-environment"], input=json.dumps(payload), text=True, capture_output=True, env=env,
                          check=True)
    return done.stdout


base = {"session_id": "demo-1", "cwd": "."}
hook({**base, "hook_event_name": "UserPromptSubmit", "prompt": "We deploy with make ship, never with scp."})
hook({**base, "hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": "make ship"},
      "tool_response": {"stdout": "shipped"}})
hook({**base, "hook_event_name": "Stop"})

# A later session in the same project: the prompt hook answers with context for the model.
out = hook({"session_id": "demo-2", "cwd": ".", "hook_event_name": "UserPromptSubmit",
            "prompt": "How do we deploy this?"})
print(out or "(nothing to add)")
print(f"store: {store}")
