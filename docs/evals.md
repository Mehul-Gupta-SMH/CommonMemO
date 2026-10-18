# Evaluation results

Deterministic mode of the evaluation suite ([#18](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/18)): each scenario runs with and without memory against a scripted agent that decides only from the text it is shown (see `evals/`). Engine: `local`. Regenerate with `python -m evals --md docs/evals.md`.

commonmemo 0.0.1 · Python 3.13.16 · 2026-10-18

| Scenario | Checks | With memory | Without | Memory added (chars) | Time (ms) |
|---|---|---|---|---:|---:|
| known tool failure | a tool failure that has a known fix is not repeated | pass | fail | 607 | 8 |
| negatives | a user's "never" is respected in a later session | pass | fail | 288 | 4 |
| latest preference | the newest value of a changed preference is used | pass | fail | 222 | 5 |
| resume after crash | a rerun skips finished steps and never repeats a side effect | pass | fail | 0 | 73 |
| cross-harness | a session started in one harness continues in another | pass | fail | 188 | 59 |
| forgetting | a forgotten fact never comes back, in the push or through recall | pass | n/a | 202 | 4 |
| injection | stored text is never placed where instructions go | pass | n/a | 535 | 3 |
| world-fact isolation | one user's claim doesn't change what another user is told | pass | n/a | 245 | 3 |
| detect, freeze, revert | a burst of bad writes is detected, frozen and undone | pass | n/a | 0 | 50 |

9 of 9 pass with memory; memory turned a failure into a success in 5 of the 5 scenarios that have a without-memory baseline.

Details (with memory):

- **known tool failure:** called with 2026-10-20 first time
- **negatives:** chose to contact by email
- **latest preference:** chose email (latest preference is email)
- **resume after crash:** charged 1x, emailed 1x
- **cross-harness:** picked up at c.py with a.py and b.py done
- **forgetting:** forgotten address appears nowhere
- **injection:** stored instruction stayed inside the reference-data section
- **world-fact isolation:** Ada was told '30 days'
- **detect, freeze, revert:** alert raised: True; world facts after revert and freeze: {'refund_window': '30 days'}
