# Changelog

## 0.0.1 — unreleased

Graphiti engine:

- `GraphitiEngine`: one Graphiti episode per closed session in the user's
  graph partition; retrieval ranked by matching graph facts; extracted facts
  alongside explicit ones, with provenance; re-ingest replaces; `forget_user`
  deletes the partition. `pip install "commonmemo[graphiti]"`.
- `Engine.bind(store)` so an engine can keep local state in the store's file.
- The `graphiti` extra declares `httpx`, which graphiti-core imports without
  declaring.

First skeleton:

- Event tier: `MemoryEvent` (subclass to define events, no decorator needed),
  built-ins `Message`, `ToolInvoked`, `UserFact`, `ExternalEvent`; `@mem.track`
  for sync and async functions.
- Session tier: resumable sessions with persisted state, ownership check on
  resume, TTL.
- `Engine` contract for episodic and user memory, with the offline `local`
  engine (rule-based episodes, keyword retrieval, superseding facts with history).
- `forget_user` across every tier; `prune` for event retention.
- CommonADK integration: one hook for all six CommonADK targets, plus
  `run_with_memory`.
