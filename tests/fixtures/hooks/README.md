# Recorded hook payloads

What Claude Code and Codex send a hook command on stdin, one file per event,
used by `tests/test_hooks.py`.

- `claude-code/`: shaped after Claude Code 2.1.302's hook input schemas.
- `codex/`: checked in the tests against `codex-schema/`, the hook input and
  output JSON schemas built into the Codex CLI 0.163.0 binary
  (`codex-rs/hooks`, Apache-2.0), copied unchanged.

When a harness changes its payloads, refresh the files from the new version
rather than editing them by hand.
