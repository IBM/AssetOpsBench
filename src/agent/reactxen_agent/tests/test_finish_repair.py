"""Offline coverage for wrapped evidence paths and reserved finish-only turns."""

import json
from types import SimpleNamespace

import pytest

from agent.reactxen_agent.reflection_plan import (
    ReflectionFinishParams,
    build_reflection_finish_tool,
)
from agent.stirrup_agent.finish_repair import finish_feedback, repair_finish
from agent.stirrup_agent.trajectory import build_trajectory


def fields():
    return {
        "selected_alternative": "alternative_1",
        "selection_reason": "verified from existing data",
        "alternative_assessments": [
            {
                "alternative_id": "alternative_1",
                "decision": "selected",
                "reason": "evidence supports it",
                "evidence_paths": {"item": ["mcp_results/orders.json"]},
            }
        ],
        "evidence_paths": {"item": ["review/verify.json"]},
        "correction_steps": ["Apply the verified calculation"],
    }


def assistant(arguments=None, tokens=3):
    calls = (
        []
        if arguments is None
        else [
            SimpleNamespace(
                name="finish",
                arguments=json.dumps(arguments),
                tool_call_id="finish-old",
            )
        ]
    )
    return SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=calls,
        token_usage=SimpleNamespace(input=tokens, output=2),
    )


def test_wrapped_top_level_and_nested_evidence_paths_become_arrays():
    report = ReflectionFinishParams(**fields())
    assert report.evidence_paths == ["review/verify.json"]
    assert report.alternative_assessments[0].evidence_paths == [
        "mcp_results/orders.json"
    ]
    assert json.loads(report.answer)["evidence_paths"] == ["review/verify.json"]


@pytest.mark.parametrize(
    "bad",
    [
        {"item": "a.json"},
        {"item": ["a"], "other": 1},
        {"item": [123]},
        {"item": [""]},
        "a.json",
    ],
)
def test_other_invalid_paths_are_not_silently_coerced(bad):
    data = fields()
    data["evidence_paths"] = bad
    with pytest.raises(ValueError):
        ReflectionFinishParams(**data)


@pytest.mark.anyio
async def test_reserved_repairs_reuse_context_expose_only_finish_and_count_tokens():
    tool = build_reflection_finish_tool(["alternative_1"])
    invalid = fields()
    invalid["selection_reason"] = "x" * 2001
    history = [[assistant(invalid, tokens=5)]]
    original_tools = {"finish": tool, "code_exec": object()}

    class Session:
        _active_tools = original_tools
        attempts = 0

        async def step(self, messages, metadata):
            self.attempts += 1
            assert self._active_tools == {"finish": tool}
            assert messages is history[-1]
            assert "selection_reason" in messages[-1].content or self.attempts == 2
            return (
                assistant(tokens=7),
                [],
                ReflectionFinishParams(**fields()) if self.attempts == 2 else None,
            )

    session = Session()
    params, attempts = await repair_finish(session, tool, history, 2)
    assert params.selected_alternative == "alternative_1"
    assert attempts == 2
    assert session._active_tools is original_tools
    assert (
        build_trajectory(history).total_input_tokens == 19
    )  # original 5 plus two repairs of 7


@pytest.mark.anyio
async def test_repair_stops_at_bound_and_restores_tools_on_error():
    tool = build_reflection_finish_tool(["alternative_1"])
    original_tools = {"finish": tool, "code_exec": object()}

    class Session:
        _active_tools = original_tools
        calls = 0

        async def step(self, messages, metadata):
            self.calls += 1
            return assistant(), [], None

    session = Session()
    params, count = await repair_finish(session, tool, [[assistant()]], 2)
    assert params is None and count == session.calls == 2
    assert session._active_tools is original_tools

    async def fail(*args):
        raise RuntimeError("provider failed")

    session.step = fail
    with pytest.raises(RuntimeError):
        await repair_finish(session, tool, [[assistant()]], 2)
    assert session._active_tools is original_tools


def test_specific_validation_feedback_identifies_nested_path_field():
    data = fields()
    data["alternative_assessments"][0]["evidence_paths"] = {"wrong": []}
    feedback = finish_feedback(
        [assistant(data)], build_reflection_finish_tool(["alternative_1"])
    )
    assert "alternative_assessments.0.evidence_paths" in feedback


def test_finish_feedback_surfaces_missing_export_file_errors():
    from agent.reactxen_agent.review_plan import build_review_finish_tool

    fields = dict(
        outcome="partially_accomplished",
        arithmetic_verified=True,
        population_verified=False,
        question_alignment_verified=False,
        alignment_findings="Original scope unresolved",
        critique="scope unresolved",
        alternative_search_summary="compared filters",
        alternatives=[],
        paths=["review/round-000/"],
    )
    history = [
        assistant(fields),
        SimpleNamespace(
            role="tool",
            name="finish",
            content="Cannot finish: reported output files do not exist: ['review/round-000/']",
        ),
    ]
    assert "review/round-000/" in finish_feedback(history, build_review_finish_tool())


def test_review_and_reflect_both_have_bounded_finish_repair():
    from agent.reactxen_agent.runner import ReActXenAgentRunner

    runner = ReActXenAgentRunner()
    assert runner._critic("review", object())._finish_repair_attempts == 2
    assert runner._critic("reflect", object())._finish_repair_attempts == 2
