"""Procedural memory (ADR 0003, #26)."""

from __future__ import annotations

import asyncio

import pytest

from commonmemo import Memory, MemoryPolicy
from commonmemo.agent_tools import build_agent_tools
from commonmemo.procedures import parse_steps, read

TEMPLATE = """---
kind: template
use_when: Writing the weekly status report for the team
---
## Highlights
## Risks
## Next week
"""

RULE = """---
kind: rule
name: length
---
Keep it under 300 words.
"""

FLOW = """---
kind: procedure
---
1. gather_metrics: pull the numbers from the dashboard
2. draft: write it up
3. review: ask the lead
"""


def run(coro):
    return asyncio.run(coro)


def tools(s):
    return {t.__name__: t for t in build_agent_tools(s)}


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "procedures" / "weekly_report"
    d.mkdir(parents=True)
    (d / "default.md").write_text(TEMPLATE)
    (d / "length.md").write_text(RULE)
    (d / "flow.md").write_text(FLOW)
    return tmp_path / "procedures"


@pytest.fixture
def mem(folder):
    m = Memory(project="acme")
    m.load_procedures(folder)
    return m


def test_files_are_read_with_task_from_directory_and_name_from_file(folder):
    specs = {s.name: s for s in read(folder)}
    assert set(specs) == {"default", "length", "flow"}
    assert all(s.task == "weekly_report" for s in specs.values())
    assert specs["default"].use_when == "Writing the weekly status report for the team"
    assert specs["flow"].steps == ("gather_metrics", "draft", "review")
    assert parse_steps("1. Gather metrics\n2) Write it up: carefully") == ("gather_metrics", "write_it_up")


def test_loading_is_idempotent_and_a_changed_body_is_a_new_version(mem, folder):
    again = mem.load_procedures(folder)
    assert (again.written, again.unchanged) == (0, 3)
    (folder / "weekly_report" / "length.md").write_text(RULE.replace("300", "250"))
    changed = mem.load_procedures(folder)
    assert (changed.written, changed.unchanged) == (1, 2)
    rule = mem.procedures("weekly_report").of("rule")[0]
    assert (rule.version, rule.body, rule.source) == (2, "Keep it under 250 words.", "file")
    versions = mem.knowledge.procedure_versions("project:acme", "weekly_report", "rule", "length")
    assert [(v.version, v.status) for v in versions] == [(1, "superseded"), (2, "current")]


def test_a_named_task_is_pushed_in_the_turn_half(mem):
    with mem.session(user_id="ada", task="weekly_report") as s:
        ctx = run(s.context("anything at all"))
    assert ctx.procedure.how == "named"
    turn = ctx.turn_text()
    assert "## How weekly_report is done" in turn
    assert "Rule (length, v1, this project's): Keep it under 300 words." in turn
    assert "## Risks" in turn
    assert "gather_metrics -> draft -> review" in turn
    assert "pull the numbers" not in turn  # a procedure is pushed as its step names
    assert "weekly_report" not in ctx.stable_text()


def test_a_request_is_matched_by_description(mem):
    with mem.session(user_id="ada") as s:
        hit = run(s.context("Can you write this week's status report?"))
        miss = run(s.context("What's the refund window?"))
    assert hit.procedure.how == "matched" and hit.procedure.score >= 0.3
    assert miss.procedure is None
    strict = Memory(project="acme", policy=MemoryPolicy(min_procedure_score=0.9))
    strict.load_procedures([{"task": "weekly_report", "kind": "rule", "name": "r", "body": "b",
                             "use_when": "Writing the weekly status report for the team"}])
    with strict.session(user_id="ada") as s:
        # Half the request's words match: enough by default, not at 0.9.
        assert run(s.context("Write the status report, then book flights to Paris")).procedure is None


def test_the_agent_can_pull_the_full_procedure(mem):
    with mem.session(user_id="ada") as s:
        t = tools(s)
        assert "- weekly_report: Writing the weekly status report" in t["memory_get_procedure"].__doc__
        full = t["memory_get_procedure"]("weekly_report")
        assert "pull the numbers from the dashboard" in full and "## Next week" in full
        assert "Known tasks:\n- weekly_report" in t["memory_get_procedure"]("expenses")
        assert mem.traces(session_id=s.session_id)[-1].reason == "pull"
        ctx = run(s.context("write the weekly status report"))
        s.recall_tool = True
    assert 'call memory_get_procedure("weekly_report")' in ctx.turn_text()


def test_no_tasks_means_no_procedure_tool():
    mem = Memory()
    with mem.session(user_id="ada") as s:
        assert "memory_get_procedure" not in tools(s)


def test_the_most_specific_scope_wins(mem):
    mem.save_procedure("weekly_report", "rule", "tone", "Formal.", scope="global")
    with mem.session(user_id="ada") as s:
        s.save_procedure("weekly_report", "rule", "length", "No limit for me.", for_user=True)
        mine = mem.procedures("weekly_report", user_id="ada")
    others = mem.procedures("weekly_report", user_id="bob")
    assert {e.name: e.body for e in mine.of("rule")} == {"tone": "Formal.", "length": "No limit for me."}
    assert {e.name: e.body for e in others.of("rule")}["length"] == "Keep it under 300 words."


def test_an_agents_entry_for_everyone_waits_for_review(mem):
    with mem.session(user_id="ada") as s:
        said = tools(s)["memory_save_procedure"]("weekly_report", "rule", "emoji", "No emoji.", for_everyone=True)
        assert said.startswith("Proposed")
        own = tools(s)["memory_save_procedure"]("weekly_report", "rule", "sign_off", "Sign as Ada.")
        assert own.startswith("Saved")
    assert "emoji" not in {e.name for e in mem.procedures("weekly_report", user_id="ada").entries}
    assert "sign_off" in {e.name for e in mem.procedures("weekly_report", user_id="ada").entries}
    [pending] = mem.pending_procedures()
    mem.confirm_procedure(pending.event_id)
    rule = {e.name: e for e in mem.procedures("weekly_report", user_id="bob").entries}["emoji"]
    assert (rule.version, rule.source, rule.scope) == (1, "confirmed", "project:acme")
    assert mem.knowledge.procedure_entry(pending.event_id).status == "confirmed"


def test_a_less_trusted_source_never_replaces_a_more_trusted_entry(mem):
    mem.save_procedure("weekly_report", "rule", "length", "Any length.", source="tool")
    assert mem.procedures("weekly_report").of("rule")[0].body == "Keep it under 300 words."
    [disputed] = mem.pending_procedures()
    assert disputed.status == "disputed" and "file entry" in disputed.note
    mem.reject_procedure(disputed.event_id, reason="no")
    assert mem.pending_procedures() == []
    with pytest.raises(ValueError):
        mem.reject_procedure(disputed.event_id)


def test_retiring_an_entry(mem):
    mem.retire_procedure("weekly_report", "rule", "length", reason="dropped")
    assert [e.name for e in mem.procedures("weekly_report").of("rule")] == []


def test_exemplars_stay_with_their_user_until_shared_scrubbed(mem):
    with mem.session(user_id="ada") as s:
        s.remember("manager", "Grace Hopper")
    with mem.session(user_id="ada") as s:
        ev = s.save_exemplar("weekly_report", "## Highlights\nShipped v2. Grace Hopper approved.")
    assert mem.procedures("weekly_report", user_id="ada").of("exemplar")
    assert not mem.procedures("weekly_report", user_id="bob").of("exemplar")
    with pytest.raises(ValueError, match="scrubbed=True"):
        mem.confirm_procedure(ev.id)
    mem.confirm_procedure(ev.id, scrubbed=True)
    [shared] = mem.procedures("weekly_report", user_id="bob").of("exemplar")
    assert shared.body == "## Highlights\nShipped v2. [redacted] approved."
    with pytest.raises(ValueError, match="user's scope"):
        mem.save_procedure("weekly_report", "exemplar", "x", "y", source="agent")


def test_forget_user_removes_their_entries_and_keeps_approved_copies(mem):
    with mem.session(user_id="ada") as s:
        kept = s.save_exemplar("weekly_report", "Shared one.", name="shared")
        s.save_exemplar("weekly_report", "Private one.", name="private")
        s.save_procedure("weekly_report", "rule", "mine", "My way.", for_user=True)
        tools(s)["memory_save_procedure"]("weekly_report", "rule", "idea", "An idea.", for_everyone=True)
    mem.confirm_procedure(kept.id, scrubbed=True)
    run(mem.forget_user("ada"))
    names = {e.name for e in mem.procedures("weekly_report", user_id="ada").entries}
    assert "shared" in names and not names & {"private", "mine", "idea"}
    assert mem.pending_procedures() == []
    mem.rebuild_knowledge()
    assert {e.name for e in mem.procedures("weekly_report", user_id="ada").entries} == names


def test_a_users_entry_written_outside_a_session_is_forgotten_too(mem):
    mem.save_procedure("weekly_report", "rule", "mine", "My way.", scope="user:ada")
    run(mem.forget_user("ada"))
    mem.rebuild_knowledge()
    assert "mine" not in {e.name for e in mem.procedures("weekly_report", user_id="ada").entries}


def test_read_only_tools_pull_without_recording(mem):
    with mem.session(user_id="ada") as s:
        pull = {t.__name__: t for t in build_agent_tools(s, writes=False)}["memory_get_procedure"]
        assert "## Next week" in pull("weekly_report")
        assert mem.traces(session_id=s.session_id) == []


def test_runs_are_recorded_with_skipped_steps_and_outcomes_per_version(mem):
    @mem.step
    def gather_metrics() -> int:
        return 3

    @mem.step
    def draft() -> str:
        return "draft"

    with mem.session(user_id="ada", task="weekly_report") as s:
        run(s.push_turn("write it"))
        gather_metrics()
        draft()
        s.feedback(1)
    with mem.session(user_id="bob", task="weekly_report") as s:
        run(s.push_turn("write it"))
        run(s.push_turn("again"))  # pushed twice, one run
        s.feedback(-1)
    outcomes = {(o.kind, o.name): o for o in mem.procedure_outcomes("weekly_report")}
    flow = outcomes[("procedure", "flow")]
    assert (flow.version, flow.runs, flow.rated, flow.positive, flow.negative) == (1, 2, 2, 1, 1)
    assert flow.skipped == {"review": 1}  # only ada's run had tracked steps
    assert outcomes[("rule", "length")].runs == 2


def test_an_explicit_follow_is_not_recorded_twice(mem):
    with mem.session(user_id="ada", task="weekly_report") as s:
        run(s.push_turn("write it"))
        s.follow("weekly_report", "flow", skipped=("review",))
    flow = {o.name: o for o in mem.procedure_outcomes("weekly_report")}["flow"]
    assert (flow.runs, flow.skipped) == (1, {"review": 1})


def test_why_traces_a_pushed_entry_to_its_source(mem):
    with mem.session(user_id="ada", task="weekly_report") as s:
        run(s.push_turn("write it"))
        found = mem.why("weekly_report/rule/length", session_id=s.session_id)
    assert found.item["version"] == 1 and found.sources[0].kind == "procedure_entry"


def test_the_harness_hook_pushes_a_matched_procedure(mem, tmp_path):
    from commonmemo.hooks import handle

    out = handle({"hook_event_name": "UserPromptSubmit", "session_id": "h1", "cwd": str(tmp_path),
                  "prompt": "Please write the weekly status report"}, mem, harness="claude-code", user="ada")
    assert "## How weekly_report is done" in out["hookSpecificOutput"]["additionalContext"]
