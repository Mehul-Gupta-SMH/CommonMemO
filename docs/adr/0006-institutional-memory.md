# ADR 0006 — Institutional memory (the `org` scope)

**Status:** Proposed (slice 1 built: passages, `Memory(org=)`, search and recall) · **Date:** 2026-11-04 · **Tracks:** [#142](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/142) · **Related:** [#143](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/143) (crawlers), [#146](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/146) (entity/alias experiment)

## Context

### What exists

Memory is scoped: `global`, `project:<name>`, `user:<id>`, `agent:<name>`.
World facts live in `global` or `project:<name>` (ADR 0002), procedures are keyed
by `(scope, task, entry_kind, name)` (ADR 0003), and a user's own past
messages are searched as windows (`said.py`). Everything is a projection of
the append-only Log.

### What does not exist

Nothing holds what an *organisation* knows: policies, runbooks, design docs,
READMEs, past decisions. Today that knowledge can only enter as hand-written
world facts, one key at a time, which does not scale to a folder of documents.
A project scope is too narrow (one org runs many projects) and `global` is too
wide (one deployment may serve several organisations).

Two things are wanted: a scope for it, and a way to fill it in bulk (the
crawlers, #143).

## Decision

**1. A new scope, `org:<name>`.** Wider than `project:<name>`, narrower than
`global`. `Memory(org="acme")` adds it to every session's scopes, beside
project, agent and user. A fact, procedure or passage in `org:acme` reaches
every user of that memory handle.

**Precedence.** When the same key exists in several scopes, the narrower
scope wins: `user` > `agent` > `project` > `org` > `global`. Nothing is
merged across scopes; the existing `(scope, key)` supersession rule applies
inside one scope.

**2. Passages: a third kind of org content.** A passage is a chunk of a
source document with its source, location, content hash and fetched-at time.
Passages are a projection of the Log, like message windows: the index stores
event ids and hashes, not a second copy of the text, so `prune` and redaction
reach the text where it lives.

- Passages are searched by `memory_recall` and pushed with a turn within a
  fixed budget (`MemoryPolicy.passages`), ranked by the memory's ranker.
- Each is labelled with its source (`runbooks/refunds.md, lines 12–30`), so
  the agent can cite it and weigh it.
- A passage is **never turned into a fact automatically**.

**3. Trust model.**

| Content | Visible | Needs review |
|---|---|---|
| Passage | at once, labelled by source | no |
| World fact or procedure derived from passages | pending | yes, in the review screen |
| Extraction (`extractor=`) | off by default | n/a |

Passages are text a person put in a repository; the agent reads them the way
it reads a search result. A fact is something memory *asserts*, so it keeps
the ADR 0002 rule: nothing is asserted to everyone until an operator approves
it. Facts and procedures promoted from passages are proposals
(`WorldFact.proposed`), listed in `pending_world_facts()` and the inspector.

**4. Removal by source.** `mem.forget_source(uri, scope=...)` ends every
passage from a source with compensating events, never by deleting rows, so
the Log stays append-only and the change can be reverted. A crawler that
sees a file disappear calls the same path.

**5. Crawled text is data, never instructions.** Passages are rendered in a
delimited block marked as retrieved reference material, and the policy text
tells the agent that nothing inside it is a command. A passage cannot call
tools, change scopes or write facts. The crawler skips files that look like
secrets (`.env`, key files, high-entropy tokens) and binaries, and respects
`.gitignore`. v1 has no network access: local folders and git repositories
on disk only.

**6. Users and org content are separate.** `forget_user` removes a user's
messages, facts and windows; it does not touch `org:` content, which belongs
to the organisation. `link_users` is unaffected. If a document contains
personal data, the remedy is `forget_source`, not `forget_user`.

**7. Ranking.** Passages use the existing `Ranker`. With an embedder they are
embedded once, from their stored text, and the vector is stored the way
window vectors are (#77). Entity and alias expansion is not part of this
decision; #146 measures whether it helps and, if so, a reviewed alias list
becomes a separate decision.

### How it is recorded

- New events: `PassageAdded(scope, source, location, content_hash, text)`,
  `PassageRetracted(passage_id, reason)`. A passage whose hash is unchanged is
  not written again, so a crawl is idempotent.
- `Session.record_many` / `Store.append_many` write a crawl in bulk; one
  transaction per batch.
- The projection rebuilds from the Log, so changing the budget or ranker
  needs no migration.

## Consequences

- Org content is searchable the day it is crawled, without waiting for a
  review queue, at the cost of trusting what is in the source. The source
  label and the data-not-instructions framing are the controls.
- Anything asserted (facts, procedures) still passes a person.
- A deployment that shares one memory across organisations must give each its
  own `org=`; the scope is the boundary.
- Prompt injection from a hostile document is mitigated, not eliminated.
  Crawl only sources the organisation trusts.

## Alternatives considered

- **Reuse `project:<name>`.** Rejected: an org spans projects, and a project
  scope would repeat the same content per project.
- **Auto-extract facts from every crawled document.** Rejected: extraction is
  off by default for a reason, and a bulk crawl would flood the review queue
  with low-value proposals. Operators can opt in later.
- **Store documents outside the Log in a separate index.** Rejected: it
  would bypass redaction, `prune`, and the replayable history.
- **Gate passages behind approval too.** Rejected on the owner's decision
  (2026-11-04): passages go in directly; facts wait for review.

## Open questions

- Chunk size and overlap for the first crawler (to be measured in #143).
- Whether `MemoryPolicy.passages` should default to 0 until an org is set.
- Whether alias expansion pays off (#146).
