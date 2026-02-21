"""Tests for PlanExecuteRunner and Executor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from plan_execute.executor import (
    Executor,
    _has_placeholders,
    _parse_json,
    _parse_tool_call,
    _resolve_args,
    _resolve_args_with_llm,
)
from plan_execute.models import Plan, PlanStep, StepResult
from plan_execute.runner import PlanExecuteRunner

# ── shared plan strings ───────────────────────────────────────────────────────

_TWO_STEP_PLAN = """\
#Task1: Get IoT sites
#Agent1: IoTAgent
#Tool1: sites
#Args1: {}
#Dependency1: None
#ExpectedOutput1: List of site names

#Task2: Get current datetime
#Agent2: Utilities
#Tool2: current_date_time
#Args2: {}
#Dependency2: None
#ExpectedOutput2: Current date and time"""

_FINAL_ANSWER = "Sites: MAIN. Current time: 2026-02-18T13:00:00."

_MOCK_TOOLS = [
    {"name": "sites", "description": "List IoT sites", "parameters": []},
    {"name": "current_date_time", "description": "Get current datetime", "parameters": []},
]
_TOOL_RESPONSE = json.dumps({"sites": ["MAIN"]})


# ── helper to patch MCP helpers ───────────────────────────────────────────────


def _patch_mcp(tool_response: str = _TOOL_RESPONSE):
    return (
        patch("plan_execute.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch(
            "plan_execute.executor._call_tool", new=AsyncMock(return_value=tool_response)
        ),
    )


def _make_step(
    n: int,
    agent: str = "IoTAgent",
    tool: str = "sites",
    tool_args: dict | None = None,
    deps: list[int] | None = None,
) -> PlanStep:
    return PlanStep(
        step_number=n,
        task=f"Task {n}",
        agent=agent,
        tool=tool,
        tool_args=tool_args or {},
        dependencies=deps or [],
        expected_output=f"output {n}",
    )


# ── orchestrator tests ────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_orchestrator_run_returns_result(sequential_llm):
    llm = sequential_llm([
        _TWO_STEP_PLAN,  # planner call
        _FINAL_ANSWER,   # summarisation
    ])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteRunner(llm).run("What are the IoT sites?")

    assert result.question == "What are the IoT sites?"
    assert result.answer == _FINAL_ANSWER
    assert len(result.plan.steps) == 2
    assert len(result.history) == 2


@pytest.mark.anyio
async def test_orchestrator_all_steps_succeed(sequential_llm):
    llm = sequential_llm([_TWO_STEP_PLAN, _FINAL_ANSWER])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteRunner(llm).run("Q")

    assert all(r.success for r in result.history)


@pytest.mark.anyio
async def test_orchestrator_unknown_agent_recorded_as_error(sequential_llm):
    bad_plan = (
        "#Task1: Do something\n"
        "#Agent1: GhostAgent\n"
        "#Tool1: ghost_tool\n"
        "#Args1: {}\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: Result\n"
    )
    llm = sequential_llm([bad_plan, _FINAL_ANSWER])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteRunner(llm).run("Q")

    assert len(result.history) == 1
    assert result.history[0].success is False
    assert "GhostAgent" in result.history[0].error


@pytest.mark.anyio
async def test_orchestrator_no_tool_returns_expected_output(sequential_llm):
    """A step with tool=none returns expected_output directly (no MCP call)."""
    plan_with_no_tool = (
        "#Task1: Answer from context\n"
        "#Agent1: IoTAgent\n"
        "#Tool1: none\n"
        "#Args1: {}\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: 42\n"
    )
    llm = sequential_llm([plan_with_no_tool, "Final: 42"])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteRunner(llm).run("Simple Q")

    assert result.history[0].response == "42"
    assert result.history[0].success is True


# ── executor unit tests ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_executor_unknown_agent(mock_llm):
    llm = mock_llm("")
    executor = Executor(llm, server_paths={})  # no servers registered

    plan = Plan(steps=[_make_step(1)], raw="")
    with _patch_mcp()[0], _patch_mcp()[1]:
        results = await executor.execute_plan(plan, "Q")

    assert results[0].success is False
    assert "IoTAgent" in results[0].error


@pytest.mark.anyio
async def test_executor_get_agent_descriptions(mock_llm):
    llm = mock_llm()
    executor = Executor(llm, server_paths={"TestServer": None})

    with patch(
        "plan_execute.executor._list_tools",
        new=AsyncMock(
            return_value=[{"name": "foo", "description": "does foo", "parameters": []}]
        ),
    ):
        descs = await executor.get_agent_descriptions()

    assert "TestServer" in descs
    assert "foo" in descs["TestServer"]


@pytest.mark.anyio
async def test_executor_resolves_placeholder_via_llm(mock_llm):
    """Steps with {step_N} placeholders use an LLM call to resolve arg values."""
    from pathlib import Path

    resolved_json = json.dumps({"asset_id": "CH-1"})
    llm = mock_llm(resolved_json)
    executor = Executor(llm, server_paths={"IoTAgent": Path("/fake/server.py")})

    plan = Plan(
        steps=[
            _make_step(1, tool="sites", tool_args={}),
            _make_step(2, tool="sensors",
                       tool_args={"site_name": "MAIN", "asset_id": "{step_1}"},
                       deps=[1]),
        ],
        raw="",
    )
    site_resp = json.dumps({"sites": ["MAIN"]})
    sensor_resp = json.dumps({"sensors": ["temp"]})

    call_mock = AsyncMock(side_effect=[site_resp, sensor_resp])
    with (
        patch("plan_execute.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("plan_execute.executor._call_tool", new=call_mock),
    ):
        results = await executor.execute_plan(plan, "Q")

    assert all(r.success for r in results)
    # Step 2 tool call should receive LLM-resolved asset_id
    step2_args = call_mock.call_args_list[1].args[2]
    assert step2_args["site_name"] == "MAIN"   # known arg passed through
    assert step2_args["asset_id"] == "CH-1"    # resolved by LLM


@pytest.mark.anyio
async def test_pipeline_resolves_placeholder_from_planner_output(sequential_llm):
    """Regression test for the {step_N} placeholder regex bug.

    The planner LLM returns a plan string with {step_N} (single braces) in
    Args — exactly what the LLM produces after _PLAN_PROMPT.format() renders
    {{step_N}} -> {step_N}.  The executor must detect the placeholder, call the
    LLM to resolve it, and forward the resolved value (not the placeholder
    string) to the tool.

    With the old regex r"\\{\\{step_(\\d+)\\}\\}" this test failed because
    _has_placeholders() returned False for single-brace args, so the literal
    string "{step_1}" was passed as site_name to the tool.
    """
    planner_output = (
        "#Task1: Get IoT sites\n"
        "#Agent1: IoTAgent\n"
        "#Tool1: sites\n"
        "#Args1: {}\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: List of site names\n\n"
        "#Task2: Get assets at the site from step 1\n"
        "#Agent2: IoTAgent\n"
        "#Tool2: assets\n"
        '#Args2: {"site_name": "{step_1}"}\n'
        "#Dependency2: #S1\n"
        "#ExpectedOutput2: List of assets"
    )
    llm = sequential_llm([
        planner_output,            # planner call
        '{"site_name": "MAIN"}',   # arg resolution for step 2
        "Final answer.",           # summarisation
    ])

    site_resp = '{"sites": ["MAIN"]}'
    asset_resp = '{"assets": ["CH-1"]}'
    call_mock = AsyncMock(side_effect=[site_resp, asset_resp])

    with (
        patch("plan_execute.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("plan_execute.executor._call_tool", new=call_mock),
    ):
        result = await PlanExecuteRunner(llm).run("List all assets at site MAIN")

    assert all(r.success for r in result.history)
    # Step 2 must be called with the resolved value, not the placeholder string
    step2_args = call_mock.call_args_list[1].args[2]
    assert step2_args["site_name"] == "MAIN"
    assert "{step_1}" not in str(step2_args)


@pytest.mark.anyio
async def test_executor_no_placeholder_skips_llm(mock_llm):
    """Steps without placeholders do not trigger an LLM call."""
    from pathlib import Path

    llm = mock_llm("")
    executor = Executor(llm, server_paths={"IoTAgent": Path("/fake/server.py")})

    plan = Plan(steps=[_make_step(1, tool="sites", tool_args={})], raw="")
    call_mock = AsyncMock(return_value=_TOOL_RESPONSE)
    with (
        patch("plan_execute.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("plan_execute.executor._call_tool", new=call_mock),
    ):
        await executor.execute_plan(plan, "Q")

    # LLM generate should never have been called
    assert llm._response == "" and call_mock.call_count == 1


# ── _has_placeholders tests ───────────────────────────────────────────────────


def test_has_placeholders_true():
    assert _has_placeholders({"asset_id": "{step_1}"}) is True


def test_has_placeholders_false():
    assert _has_placeholders({"site_name": "MAIN"}) is False


def test_has_placeholders_empty():
    assert _has_placeholders({}) is False


def test_has_placeholders_non_string_ignored():
    assert _has_placeholders({"count": 5}) is False


# ── _resolve_args_with_llm tests ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_resolve_args_with_llm_resolves_placeholder(mock_llm):
    llm = mock_llm('{"asset_id": "CH-1"}')
    ctx = {1: StepResult(step_number=1, task="t", agent="a",
                         response='{"assets": ["CH-1", "CH-2"]}')}
    result = await _resolve_args_with_llm(
        "get sensors", "sensors",
        {"site_name": "MAIN", "asset_id": "{step_1}"},
        ctx, llm,
    )
    assert result["site_name"] == "MAIN"   # known arg unchanged
    assert result["asset_id"] == "CH-1"    # resolved by LLM


@pytest.mark.anyio
async def test_resolve_args_with_llm_fallback_on_bad_json(mock_llm):
    llm = mock_llm("I cannot determine the value.")
    ctx = {1: StepResult(step_number=1, task="t", agent="a", response="data")}
    result = await _resolve_args_with_llm(
        "task", "tool", {"x": "{step_1}"}, ctx, llm
    )
    # Bad JSON → empty dict merged with known args (none here) → x absent
    assert result == {}


# ── _resolve_args tests (simple substitution, kept for reference) ─────────────


def test_resolve_args_no_placeholders():
    args = {"site_name": "MAIN", "limit": 10}
    assert _resolve_args(args, {}) == args


def test_resolve_args_replaces_placeholder():
    ctx = {1: StepResult(step_number=1, task="t", agent="a", response="MAIN")}
    resolved = _resolve_args({"site_name": "{step_1}"}, ctx)
    assert resolved["site_name"] == "MAIN"


def test_resolve_args_missing_step_keeps_placeholder():
    resolved = _resolve_args({"site_name": "{step_9}"}, {})
    assert resolved["site_name"] == "{step_9}"


def test_resolve_args_non_string_values_unchanged():
    args = {"count": 5, "flag": True}
    assert _resolve_args(args, {}) == args


# ── _parse_json tests ─────────────────────────────────────────────────────────


def test_parse_json_plain():
    assert _parse_json('{"a": "b"}') == {"a": "b"}


def test_parse_json_markdown_fence():
    assert _parse_json('```json\n{"a": "b"}\n```') == {"a": "b"}


def test_parse_json_embedded():
    assert _parse_json('Result: {"a": "b"} done.') == {"a": "b"}


def test_parse_json_unrecoverable_returns_empty():
    assert _parse_json("no json here") == {}


# ── _parse_tool_call tests ────────────────────────────────────────────────────


def test_parse_tool_call_plain_json():
    raw = '{"tool": "sites", "args": {}}'
    result = _parse_tool_call(raw)
    assert result["tool"] == "sites"
    assert result["args"] == {}


def test_parse_tool_call_with_markdown_fence():
    raw = '```json\n{"tool": "history", "args": {"site_name": "MAIN"}}\n```'
    result = _parse_tool_call(raw)
    assert result["tool"] == "history"
    assert result["args"]["site_name"] == "MAIN"


def test_parse_tool_call_null_tool():
    raw = '{"tool": null, "answer": "42"}'
    result = _parse_tool_call(raw)
    assert result["tool"] is None
    assert result["answer"] == "42"


def test_parse_tool_call_embedded_json():
    raw = 'Here is my response: {"tool": "sites", "args": {}} done.'
    result = _parse_tool_call(raw)
    assert result["tool"] == "sites"


def test_parse_tool_call_unrecoverable_returns_direct_answer():
    raw = "I cannot decide which tool to use."
    result = _parse_tool_call(raw)
    assert result["tool"] is None
    assert result["answer"] == raw
