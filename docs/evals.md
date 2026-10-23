# Evaluation results

Deterministic mode of the evaluation suite ([#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)): each scenario runs with and without memory against a scripted agent that decides only from the text it is shown (see `evals/`). Engine: `local`. Regenerate with `python -m evals --md docs/evals.md`.

commonmemo 0.0.1 · Python 3.13.16 · 2026-10-23

| Scenario | Checks | With memory | Without | Memory added (chars) | Time (ms) |
|---|---|---|---|---:|---:|
| known tool failure | a tool failure that has a known fix is not repeated | pass | fail | 607 | 9 |
| negatives | a user's "never" is respected in a later session | pass | fail | 288 | 6 |
| latest preference | the newest value of a changed preference is used | pass | fail | 222 | 6 |
| resume after crash | a rerun skips finished steps and never repeats a side effect | pass | fail | 0 | 81 |
| cross-harness | a session started in one harness continues in another | pass | fail | 188 | 57 |
| forgetting | a forgotten fact never comes back, in the push or through recall | pass | n/a | 205 | 11 |
| injection | stored text is never placed where instructions go | pass | n/a | 498 | 8 |
| world-fact isolation | one user's claim doesn't change what another user is told | pass | n/a | 245 | 4 |
| mid-run refresh | a tool result that brings up an older session gets its recap pushed mid-run | pass | fail | 248 | 8 |
| more available | when the push leaves things out, the agent is told and recalls them | pass | fail | 841 | 12 |
| staleness | a fact past its lifetime isn't quoted as current | pass | n/a | 0 | 104 |
| precedence | an agent's write doesn't override an operator's fact | pass | n/a | 245 | 5 |
| recall by meaning | a fact asked about in other words is recalled | pass | fail | 0 | 18 |
| procedure by description | a request for a known task gets the house template, unnamed | pass | fail | 301 | 4 |
| learning from runs | a correction in one session becomes a lesson for that user's next one | pass | fail | 545 | 5 |
| pinned library version | the agent uses the API of the library version the project pins | pass | fail | 437 | 10 |
| detect, freeze, revert | a burst of bad writes is detected, frozen and undone | pass | n/a | 0 | 56 |

17 of 17 pass with memory; memory turned a failure into a success in 11 of the 11 scenarios that have a without-memory baseline.

Details (with memory):

- **known tool failure:** called with 2026-10-20 first time
- **negatives:** chose to contact by email
- **latest preference:** chose email (latest preference is email)
- **resume after crash:** charged 1x, emailed 1x
- **cross-harness:** picked up at c.py with a.py and b.py done
- **forgetting:** forgotten address appears nowhere
- **injection:** stored instruction stayed inside the reference-data section
- **world-fact isolation:** Ada was told '30 days'
- **mid-run refresh:** reflashed over serial (refresh off: over ota); memory pushed 248 chars with refresh, 0 without
- **more available:** with the line: 0 recall call, answered '4471'; without it: 0 calls, answered '4471'
- **staleness:** didn't quote the lapsed price; would look it up
- **precedence:** Ada was told '30 days'; the agent's write is kept for review (1 disputed)
- **recall by meaning:** recalled 4/4 with an embedding ranker (offline concept embedder), 1/4 by shared words; 0 unrelated facts returned
- **procedure by description:** used the template in 2/2 report requests; pushed to an unrelated request: False
- **learning from runs:** answered in km after the correction; another user's answer changed: False
- **pinned library version:** called .dict() on a pydantic 1.10 model
- **detect, freeze, revert:** alert raised: True; world facts after revert and freeze: {'refund_window': '30 days'}
