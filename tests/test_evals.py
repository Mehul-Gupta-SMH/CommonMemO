"""The evaluation suite (#18) in deterministic mode: every scenario passes
with memory, and memory makes the difference wherever there is a baseline."""

import pytest

from evals import run_all
from evals.scenarios import SCENARIOS


@pytest.fixture(scope="module")
def results():
    return {r.scenario.name: r for r in run_all()}


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS])
def test_scenario_passes_with_memory(results, name):
    r = results[name]
    assert r.with_memory.passed, r.with_memory.detail


@pytest.mark.parametrize("name", [s.name for s in SCENARIOS if s.baseline])
def test_memory_is_what_makes_it_pass(results, name):
    r = results[name]
    assert r.without_memory is not None and not r.without_memory.passed, (
        f"{name} also passes without memory, so it doesn't measure memory: {r.without_memory.detail}"
    )
