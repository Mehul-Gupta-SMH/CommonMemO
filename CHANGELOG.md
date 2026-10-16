# Changelog

## 0.0.1 — unreleased

Mem0 engine:

- `Mem0Engine`: each closed session is one Mem0 `add`, tagged with its
  session id; extracted memories join explicit facts (explicit first) with
  provenance through the origin session; retrieval ranks past sessions by
  matching memories; re-ingest replaces a session's memories;
  `forget_user` reaches Mem0. `pip install "commonmemo[mem0]"`.
- Episode ranking shared between the Graphiti and Mem0 engines
  (`engines/_ranking.py`).
- `scripts/live_engines.py`: a live run of both engines against real
  backends and an LLM.

### Fixed

- **The Graphiti engine could not create any episode.** It passed a
  session-derived `uuid=` to `Graphiti.add_episode`, which in Graphiti means
  "reprocess this existing episode" and raises `NodeNotFoundError` when
  there is none -- so every real ingest failed. The fake graph in the tests
  accepted the argument, so only the first live run exposed it. The engine
  now lets Graphiti assign the uuid and records it in the local index
  (`LocalEngine.set_external_id`); the backend test binds the call against
  graphiti-core's real signature.

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
