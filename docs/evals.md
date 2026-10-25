# Evaluation results

Deterministic mode of the evaluation suite ([#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)): each scenario runs with and without memory against a scripted agent that decides only from the text it is shown (see `evals/`). Engine: `local`. Regenerate with `python -m evals --md docs/evals.md`.

commonmemo 0.0.1 · Python 3.13.16 · 2026-10-25

| Scenario | Checks | With memory | Without | Memory added (chars) | Time (ms) |
|---|---|---|---|---:|---:|
| known tool failure | a tool failure that has a known fix is not repeated | pass | fail | 607 | 11 |
| negatives | a user's "never" is respected in a later session | pass | fail | 288 | 7 |
| latest preference | the newest value of a changed preference is used | pass | fail | 222 | 6 |
| resume after crash | a rerun skips finished steps and never repeats a side effect | pass | fail | 0 | 116 |
| cross-harness | a session started in one harness continues in another | pass | fail | 188 | 87 |
| forgetting | a forgotten fact never comes back, in the push or through recall | pass | n/a | 336 | 11 |
| injection | stored text is never placed where instructions go | pass | n/a | 498 | 4 |
| world-fact isolation | one user's claim doesn't change what another user is told | pass | n/a | 245 | 4 |
| mid-run refresh | a tool result that brings up an older session gets its recap pushed mid-run | pass | fail | 248 | 10 |
| more available | when the push leaves things out, the agent is told and recalls them | pass | fail | 841 | 14 |
| staleness | a fact past its lifetime isn't quoted as current | pass | n/a | 0 | 105 |
| precedence | an agent's write doesn't override an operator's fact | pass | n/a | 245 | 5 |
| recall by meaning | a fact asked about in other words is recalled | pass | fail | 0 | 24 |
| procedure by description | a request for a known task gets the house template, unnamed | pass | fail | 301 | 12 |
| learning from runs | a correction in one session becomes a lesson for that user's next one | pass | fail | 545 | 6 |
| pinned library version | the agent uses the API of the library version the project pins | pass | fail | 437 | 15 |
| said once, recalled later | a fact stated in one session is recalled in the next, once extracted | pass | fail | 748 | 12 |
| detail in what was said | a detail a recap drops is found in the user's past messages | pass | fail | 483 | 8 |
| detect, freeze, revert | a burst of bad writes is detected, frozen and undone | pass | n/a | 0 | 58 |

19 of 19 pass with memory; memory turned a failure into a success in 13 of the 13 scenarios that have a without-memory baseline.

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
- **said once, recalled later:** answered 'Lena arrives on Friday 14 November' with extraction; 'unknown' without it (the recap digest doesn't hold it)
- **detail in what was said:** answered 'under the blue pot' with message windows; 'unknown' without them
- **detect, freeze, revert:** alert raised: True; world facts after revert and freeze: {'refund_window': '30 days'}
