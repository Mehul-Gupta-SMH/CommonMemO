# CommonMemo: LinkedIn posts

Five posts, a few days apart, in this order ([#79](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/79)). Every number is from `docs/live-runs.md`: ten LoCoMo conversations, 450 questions, Claude Haiku for everything; update a post when its numbers change. Repo link goes in the first comment: https://github.com/Mehul-Gupta-SMH/CommonMemO. Install line for any post: `pip install commonmemo` (https://pypi.org/project/commonmemo/).

---

## 1. More context isn't memory

Pasting an agent's entire conversation history into the prompt got 51% of questions right.

Giving it eight short excerpts of what the user actually said got 61%. At a fifth of the cost.

More context isn't memory. Picking the right context is.

That's the idea behind CommonMemo, an open-source memory layer I've been building for AI agents. It works with Claude Code, Codex, OpenCode, any MCP client, or plain Python, and isn't tied to a framework.

What it does differently:
- Every message and tool call goes into one append-only log. All memory is derived from it, so it can be inspected, rebuilt or deleted.
- Each turn, it searches the user's own past conversations and hands the agent only what the request is about.
- Anything a user asks to forget is redacted everywhere, and every line the model sees can be traced back to its source.

I tested it on LoCoMo, a public benchmark of long multi-session conversations: 450 questions across ten conversations, with the smallest Claude model doing all the work. Five of those conversations I never looked at while building. The result held on them too.

It's early (alpha, on PyPI as commonmemo), and I'd like people to try to break it.

Link in the comments.

---

## 2. Four smart ideas, zero wins

I tried four ideas to make agent memory smarter. I wrote down what would count as a win before running each. None of them beat doing less.

1. A model writes keywords and likely questions for every chunk of history. Retrieval got slightly better. Answers didn't.
2. A topic tree, searched coarse to fine. Worse than flat search.
3. Resolving "see you Friday" into an actual date. A tie.
4. A model rewriting "she" and "there" into names. No better, and sometimes it made things up.

One real lesson came out of number 3. When I ranked search results on the text with the dates added in, I lost the right answer for 18 questions and gained it for 1. Annotations help the reader. They confuse the ranker.

Setting the bar first is what let me say "didn't work" and move on, instead of tuning until something looked like a win.

---

## 3. I was wrong in my own README

For a while, my README said that having a model extract facts from conversations "only ties" simple search.

It was true on the five conversations I built against.

Then I ran the five I had never looked at. Same code, same prompts, nothing touched. Extraction got 149 of 228 questions right. Simple search got 109.

My claim held on half the data and failed on the other half. I don't know why yet, and I'm not going to pretend I do.

I fixed the README the same day. Extraction still ships off by default, because it costs a model call for every conversation, and on top of message search it adds a gain I can't yet tell apart from noise.

If you build evals: keep some data you never look at while building. Run it once, unchanged. It's the cheapest way to find out you were wrong before your users do.

---

## 4. Memory that forgets by accident

Agent memory has a quiet failure mode: the more a user talks to it, the worse it gets.

My first version searched past messages by scanning them. It took up to 93 ms, and it only ever looked at the latest 4,000 messages. I planted three facts deep in older history and asked for each one ten times. It found them zero times out of 30.

Now search goes through an index derived from the message log: about half a millisecond whether the user has 10,000 messages or 500,000, and it finds them 30 times out of 30. A second pass cut the index from 570 bytes a message to 195.

The index stores ids and word stems, never the text. So when a user asks to be forgotten, deleting them from the log reaches everything built from it.

One source of truth, everything else rebuildable. It's not exciting, and it's the reason the rest works.

---

## 5. What my agent memory remembers by default, and what it doesn't

An agent that remembers everything you said will eventually repeat something you wish it hadn't. So the defaults in CommonMemo are decisions, written down:

On by default: searching your own past messages. It was the biggest improvement I measured. One setting turns it off, and the README says so up front.

Off by default: having a model extract facts about you. It helped in my tests, but it sends every conversation to a model and writes that model's claims into your profile. That's your call, not a default's.

Always: only your own messages, never another user's. Anything you ask to forget is redacted from what the model sees. Any line the model was shown can be traced back to where it came from.

Memory earns trust by being inspectable, not impressive.

It's early and it will break. If you try it, tell me where.
