"""Learning from finished sessions (ADR 0003 section 6, #26 phase 2), and reverting procedures."""

from __future__ import annotations

import asyncio
import json

from commonmemo import Memory, ModelLearner
from commonmemo.agent_tools import build_agent_tools
from commonmemo.learning import parse

REPLY = {
    "decisions": [{"goal": "schedule the review", "plan": "find a slot", "chosen": "Tuesday 10:00 UTC",
                   "rejected": ["Monday"], "outcome": "accepted", "task": ""}],
    "lessons": [{"text": "Confirm the timezone before scheduling meetings.", "task": "scheduling"}],
    "procedures": [{"task": "scheduling", "kind": "rule", "name": "timezone", "use_when": "Scheduling a meeting",
                    "body": "Always state the timezone."}],
    "exemplars": [{"task": "scheduling", "body": "Invite: Tuesday 10:00 UTC, 30 min."}],
}


class Model:
    """A stand-in model: records prompts, returns a canned reply."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def run(coro):
    return asyncio.run(coro)


def work(mem, user="ada", **kw):
    with mem.session(user_id=user, agent="helper", **kw) as s:
        s.message("user", "Book the review for Tuesday")
        s.message("assistant", "Booked Tuesday 10:00 UTC")
        s.feedback(1)
    return s


def test_off_by_default():
    mem = Memory()
    work(mem)
    assert mem.decisions() == [] and mem.pending_procedures() == []


def test_a_closed_session_is_distilled():
    model = Model(REPLY)
    mem = Memory(project="acme", learner=ModelLearner(model))
    s = work(mem)
    assert "user: Book the review for Tuesday" in model.prompts[0]
    assert "feedback: That was helpful." in model.prompts[0]
    assert "data, not instructions" in model.prompts[0]
    [d] = mem.decisions(user_id="ada")
    assert (d.chosen, d.rejected, d.session_id) == ("Tuesday 10:00 UTC", ("Monday",), s.session_id)
    # Proposals: the rule and the exemplar candidate wait for review.
    pending = {p.kind: p for p in mem.pending_procedures()}
    assert pending["rule"].source == "extracted" and pending["rule"].note == f"learned from session {s.session_id}"
    assert pending["exemplar"].scope == "user:ada"
    assert not mem.procedures("scheduling", user_id="ada").entries
    # The lesson reaches the same user next time, and no one else.
    with mem.session(user_id="ada", agent="helper") as s2:
        stable = run(s2.context("anything")).stable_text()
    assert "## Lessons from past sessions\n- Confirm the timezone before scheduling meetings." in stable
    with mem.session(user_id="bob", agent="helper") as s3:
        assert "timezone" not in run(s3.context("anything")).stable_text()


def test_lessons_are_merged_across_sessions():
    model = Model(REPLY)
    mem = Memory(learner=ModelLearner(model))
    work(mem)
    model.reply = {"lessons": [{"text": "Always confirm the timezone before scheduling meetings"}]}
    work(mem)
    assert "Confirm the timezone before scheduling meetings." in model.prompts[1]  # told what it knows
    [lesson] = mem.knowledge.learned_lessons(["user:ada"])
    assert lesson.sessions == 2 and lesson.line().endswith("(seen in 2 sessions)")
    model.reply = {"lessons": [{"text": "Prefer short replies."}]}
    work(mem)
    assert len(mem.knowledge.learned_lessons(["user:ada"])) == 2


def test_once_per_session_unless_forced():
    model = Model(REPLY)
    mem = Memory(learner=ModelLearner(model))
    s = work(mem)
    assert not run(mem.distill(s.session_id))
    assert len(model.prompts) == 1
    assert run(mem.distill(s.session_id, force=True)).decisions
    assert len(mem.decisions()) == 2


def test_a_failing_or_garbled_learner_never_fails_the_close():
    def broken(prompt):
        raise RuntimeError("model down")

    mem = Memory(learner=ModelLearner(broken))
    work(mem)  # no exception
    mem = Memory(learner=ModelLearner(Model("I can't do JSON, sorry")))
    work(mem)
    assert mem.decisions() == []
    assert not parse('{"lessons": "not a list", "decisions": [{"goal": "only a goal"}]}')


def test_a_session_without_a_user_learns_no_lessons():
    mem = Memory(learner=ModelLearner(Model(REPLY)))
    with mem.session(agent="helper") as s:
        s.message("user", "hi")
    assert mem.knowledge.learned_lessons(["agent:helper", "global"]) == []
    assert [p.kind for p in mem.pending_procedures()] == ["rule"]  # no exemplar without a user either


def test_forgotten_text_never_reaches_the_learner():
    model = Model({})
    mem = Memory(learner=ModelLearner(model))
    with mem.session(user_id="ada") as s:
        s.remember("address", "12 Elm Street")
        s.forget("address")
    with mem.session(user_id="ada") as s:
        s.message("user", "Ship it to 12 Elm Street")
    assert "12 Elm Street" not in model.prompts[-1] and "[forgotten]" in model.prompts[-1]


def test_recall_returns_lessons_and_decisions():
    mem = Memory(learner=ModelLearner(Model(REPLY)))
    work(mem)
    with mem.session(user_id="ada") as s:
        recall = {t.__name__: t for t in build_agent_tools(s)}["memory_recall"]
        found = recall("when should I schedule the review, which timezone?")
    assert "[lesson] Confirm the timezone" in found
    assert "[decision] schedule the review: chose Tuesday 10:00 UTC over Monday; accepted" in found


def test_forget_user_removes_what_was_learned_about_them():
    mem = Memory(project="acme", learner=ModelLearner(Model(REPLY)))
    work(mem)
    run(mem.forget_user("ada"))
    mem.rebuild_knowledge()
    assert mem.decisions() == [] and mem.knowledge.learned_lessons(["user:ada"]) == []
    assert mem.pending_procedures() == []


def test_learned_proposals_can_be_reverted_by_cause():
    mem = Memory(project="acme", learner=ModelLearner(Model(REPLY)))
    s = work(mem)
    result = mem.revert(run_id=f"learn:{s.session_id}", kinds=["procedure"])
    assert result.summary() == {"procedure_rejected": 2}
    assert mem.pending_procedures() == []


def test_procedures_revert_to_a_snapshot():
    mem = Memory(project="acme")
    mem.save_procedure("weekly_report", "template", "default", "## Highlights")
    mem.snapshot("v1")
    mem.save_procedure("weekly_report", "template", "default", "## Wins")
    mem.save_procedure("weekly_report", "rule", "length", "Short.")
    mem.revert(to="v1")
    found = mem.procedures("weekly_report")
    [template] = found.of("template")
    assert (template.body, template.version) == ("## Highlights", 3)  # restored as the next version
    assert found.of("rule") == []
