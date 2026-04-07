"""Tests for evaluation.conditions."""

from __future__ import annotations


from evaluation.conditions import EvalCondition, create_runner
from evaluation.skill_aware_runner import SkillAwarePlanExecuteRunner
from agent.plan_execute.runner import PlanExecuteRunner


class _FakeLLM:
    def generate(self, prompt: str, **kw) -> str:
        return "{}"


def test_eval_condition_values():
    assert EvalCondition.NO_SKILLS.value == "no_skills"
    assert EvalCondition.MCP_ONLY.value == "mcp_only"
    assert EvalCondition.MCP_SKILLS.value == "mcp_skills"


def test_create_runner_no_skills():
    runner = create_runner(EvalCondition.NO_SKILLS, _FakeLLM())
    assert isinstance(runner, PlanExecuteRunner)
    assert not isinstance(runner, SkillAwarePlanExecuteRunner)
    # No servers available
    assert runner._executor._server_paths == {}


def test_create_runner_mcp_only():
    runner = create_runner(EvalCondition.MCP_ONLY, _FakeLLM())
    assert isinstance(runner, PlanExecuteRunner)
    assert not isinstance(runner, SkillAwarePlanExecuteRunner)


def test_create_runner_mcp_skills(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    (d / "test.md").write_text(
        "---\nname: test\ndomain: iot\n---\n# Test Skill\nSome procedure.",
        encoding="utf-8",
    )
    runner = create_runner(EvalCondition.MCP_SKILLS, _FakeLLM(), skills_dir=d)
    assert isinstance(runner, SkillAwarePlanExecuteRunner)
