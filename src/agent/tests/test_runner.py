"""Tests for PlanExecuteRunner and Executor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agent.executor import (
    Executor,
    _parse_json,
    _parse_tool_call,
    _resolve_args,
    _resolve_args_with_llm,
)
from agent.models import Plan, PlanStep, StepResult
from agent.runner import PlanExecuteRunner

# ── shared plan strings ───────────────────────────────────────────────────────

_TWO_STEP_PLAN = """\
#Task1: Get IoT sites
#Server1: iot
#Tool1: sites
#Dependency1: None
#ExpectedOutput1: List of site names

#Task2: Get current datetime
#Server2: utilities
#Tool2: current_date_time
#Dependency2: None
#ExpectedOutput2: Current date and time"""

_FINAL_ANSWER = "Sites: MAIN. Current time: 2026-02-18T13:00:00."

_MOCK_TOOLS = [
    {"name": "sites", "description": "List IoT sites", "parameters": []},
    {"name": "current_date_time", "description": "Get current datetime", "parameters": []},
]
_TOOL_RESPONSE = json.dumps({"sites": ["MAIN"]})

# Arg-resolution responses for the two tool steps in _TWO_STEP_PLAN.
_STEP1_ARGS = "{}"
_STEP2_ARGS = "{}"


# ── helpers ───────────────────────────────────────────────────────────────────


def _patch_mcp(tool_response: str = _TOOL_RESPONSE):
    return (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch(
            "agent.executor._call_tool", new=AsyncMock(return_value=tool_response)
        ),
    )


def _make_step(
    n: int,
    server: str = "iot",
    tool: str = "sites",
    deps: list[int] | None = None,
    expected_output: str = "",
) -> PlanStep:
    return PlanStep(
        step_number=n,
        task=f"Task {n}",
        server=server,
        tool=tool,
        tool_args={},
        dependencies=deps or [],
        expected_output=expected_output or f"output {n}",
    )


class _CapturingLLM:
    """Records every generate() prompt and returns a canned response."""

    def __init__(self, response: str = "{}") -> None:
        self.prompts: list[str] = []
        self._response = response

    def generate(self, prompt: str, **_kw) -> str:
        self.prompts.append(prompt)
        return self._response


# ── orchestrator tests ────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_orchestrator_run_returns_result(sequential_llm):
    llm = sequential_llm([
        _TWO_STEP_PLAN,  # planner call
        _STEP1_ARGS,     # arg resolution for step 1
        _STEP2_ARGS,     # arg resolution for step 2
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
    llm = sequential_llm([_TWO_STEP_PLAN, _STEP1_ARGS, _STEP2_ARGS, _FINAL_ANSWER])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteRunner(llm).run("Q")

    assert all(r.success for r in result.history)


@pytest.mark.anyio
async def test_orchestrator_unknown_server_recorded_as_error(sequential_llm):
    bad_plan = (
        "#Task1: Do something\n"
        "#Server1: ghost\n"
        "#Tool1: ghost_tool\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: Result\n"
    )
    # Unknown server returns early — no arg-resolution LLM call for that step.
    llm = sequential_llm([bad_plan, _FINAL_ANSWER])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteRunner(llm).run("Q")

    assert len(result.history) == 1
    assert result.history[0].success is False
    assert "ghost" in result.history[0].error


@pytest.mark.anyio
async def test_orchestrator_no_tool_returns_expected_output(sequential_llm):
    """A step with tool=none returns expected_output without any MCP or LLM call."""
    plan_with_no_tool = (
        "#Task1: Answer from context\n"
        "#Server1: iot\n"
        "#Tool1: none\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: 42\n"
    )
    # Only planner + summarisation — no arg-resolution call for the no-tool step.
    llm = sequential_llm([plan_with_no_tool, "Final: 42"])
    with _patch_mcp()[0], _patch_mcp()[1]:
        result = await PlanExecuteRunner(llm).run("Simple Q")

    assert result.history[0].response == "42"
    assert result.history[0].success is True


# ── executor unit tests ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_executor_unknown_server(mock_llm):
    executor = Executor(mock_llm(""), server_paths={})

    plan = Plan(steps=[_make_step(1)], raw="")
    with _patch_mcp()[0], _patch_mcp()[1]:
        results = await executor.execute_plan(plan, "Q")

    assert results[0].success is False
    assert "iot" in results[0].error


@pytest.mark.anyio
async def test_executor_get_server_descriptions(mock_llm):
    executor = Executor(mock_llm(), server_paths={"TestServer": None})

    with patch(
        "agent.executor._list_tools",
        new=AsyncMock(
            return_value=[{"name": "foo", "description": "does foo", "parameters": []}]
        ),
    ):
        descs = await executor.get_server_descriptions()

    assert "TestServer" in descs
    assert "foo" in descs["TestServer"]


@pytest.mark.anyio
async def test_executor_no_tool_step_skips_llm():
    """tool=none must not trigger an LLM call for arg resolution."""
    from pathlib import Path

    llm = _CapturingLLM()
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})  # type: ignore[arg-type]

    step = _make_step(1, tool="none", expected_output="42")
    result = await executor.execute_step(step, {}, "Q")

    assert result.response == "42"
    assert result.success is True
    assert llm.prompts == []  # LLM was never called


@pytest.mark.anyio
async def test_executor_step_result_carries_resolved_args(sequential_llm):
    """StepResult.tool_args must reflect the args the LLM generated, not {}."""
    from pathlib import Path

    llm = sequential_llm(['{"site_name": "MAIN"}'])
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})

    step = _make_step(1, tool="assets")
    with (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("agent.executor._call_tool", new=AsyncMock(return_value="{}")),
    ):
        result = await executor.execute_step(step, {}, "List assets at MAIN")

    assert result.tool_args == {"site_name": "MAIN"}


@pytest.mark.anyio
async def test_executor_tool_call_exception_recorded_as_error(sequential_llm):
    """If _call_tool raises, the error is captured in StepResult (no crash)."""
    from pathlib import Path

    llm = sequential_llm(['{}'])
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})

    step = _make_step(1, tool="sites")
    with (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("agent.executor._call_tool", new=AsyncMock(side_effect=RuntimeError("timeout"))),
    ):
        result = await executor.execute_step(step, {}, "Q")

    assert result.success is False
    assert "timeout" in result.error


@pytest.mark.anyio
async def test_executor_calls_llm_to_generate_args(sequential_llm):
    """Each tool step triggers exactly one LLM call for arg generation."""
    from pathlib import Path

    llm = sequential_llm([
        '{}',                                       # step 1: sites (no args)
        '{"site_name": "MAIN", "asset_id": "CH-1"}',  # step 2: sensors
    ])
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})

    plan = Plan(
        steps=[
            _make_step(1, tool="sites"),
            _make_step(2, tool="sensors", deps=[1]),
        ],
        raw="",
    )
    call_mock = AsyncMock(side_effect=[
        json.dumps({"sites": ["MAIN"]}),
        json.dumps({"sensors": ["temp"]}),
    ])
    with (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("agent.executor._call_tool", new=call_mock),
    ):
        results = await executor.execute_plan(plan, "Q")

    assert all(r.success for r in results)
    step2_args = call_mock.call_args_list[1].args[2]
    assert step2_args["site_name"] == "MAIN"
    assert step2_args["asset_id"] == "CH-1"


@pytest.mark.anyio
async def test_executor_prior_step_results_in_llm_prompt():
    """Prior step results appear in the LLM prompt for dependent steps."""
    from pathlib import Path

    llm = _CapturingLLM('{"asset_id": "CH-1"}')
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})  # type: ignore[arg-type]

    plan = Plan(
        steps=[
            _make_step(1, tool="sites"),
            _make_step(2, tool="sensors", deps=[1]),
        ],
        raw="",
    )
    site_resp = json.dumps({"sites": ["MAIN"]})
    call_mock = AsyncMock(side_effect=[site_resp, '{"sensors": []}'])
    with (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("agent.executor._call_tool", new=call_mock),
    ):
        await executor.execute_plan(plan, "List sensors for CH-1")

    # Step 2's LLM prompt (index 1) must contain step 1's tool response.
    assert site_resp in llm.prompts[1]


@pytest.mark.anyio
async def test_executor_no_prior_context_shows_none_in_prompt():
    """When no prior steps exist the prompt contains the literal '(none)'."""
    from pathlib import Path

    llm = _CapturingLLM('{}')
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})  # type: ignore[arg-type]

    step = _make_step(1, tool="sites")
    with (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("agent.executor._call_tool", new=AsyncMock(return_value="{}")),
    ):
        await executor.execute_step(step, {}, "Q")

    assert "(none)" in llm.prompts[0]


@pytest.mark.anyio
async def test_executor_context_accumulates_across_steps():
    """Step 3's LLM prompt contains results from both steps 1 and 2."""
    from pathlib import Path

    llm = _CapturingLLM('{}')
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})  # type: ignore[arg-type]

    plan = Plan(
        steps=[
            _make_step(1, tool="sites"),
            _make_step(2, tool="assets", deps=[1]),
            _make_step(3, tool="sensors", deps=[2]),
        ],
        raw="",
    )
    resp1, resp2, resp3 = '{"sites":["MAIN"]}', '{"assets":["CH-1"]}', '{"sensors":[]}'
    call_mock = AsyncMock(side_effect=[resp1, resp2, resp3])
    with (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("agent.executor._call_tool", new=call_mock),
    ):
        await executor.execute_plan(plan, "Q")

    step3_prompt = llm.prompts[2]
    assert resp1 in step3_prompt
    assert resp2 in step3_prompt


@pytest.mark.anyio
async def test_pipeline_uses_llm_args_for_each_step(sequential_llm):
    """End-to-end: executor generates args via LLM for every tool step."""
    planner_output = (
        "#Task1: Get IoT sites\n"
        "#Server1: iot\n"
        "#Tool1: sites\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: List of site names\n\n"
        "#Task2: Get assets at the site from step 1\n"
        "#Server2: iot\n"
        "#Tool2: assets\n"
        "#Dependency2: #S1\n"
        "#ExpectedOutput2: List of assets"
    )
    llm = sequential_llm([
        planner_output,            # planner call
        '{}',                      # arg resolution for step 1 (sites needs no args)
        '{"site_name": "MAIN"}',   # arg resolution for step 2 (uses step 1 result)
        "Final answer.",           # summarisation
    ])

    call_mock = AsyncMock(side_effect=['{"sites": ["MAIN"]}', '{"assets": ["CH-1"]}'])
    with (
        patch("agent.executor._list_tools", new=AsyncMock(return_value=_MOCK_TOOLS)),
        patch("agent.executor._call_tool", new=call_mock),
    ):
        result = await PlanExecuteRunner(llm).run("List all assets at site MAIN")

    assert all(r.success for r in result.history)
    step2_args = call_mock.call_args_list[1].args[2]
    assert step2_args["site_name"] == "MAIN"


# ── _resolve_args_with_llm tests ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_resolve_args_with_llm_uses_context(mock_llm):
    llm = mock_llm('{"asset_id": "CH-1"}')
    ctx = {1: StepResult(step_number=1, task="t", server="a",
                         response='{"assets": ["CH-1", "CH-2"]}')}
    result = await _resolve_args_with_llm(
        "What sensors does CH-1 have?", "get sensors", "sensors", "", ctx, llm,
    )
    assert result["asset_id"] == "CH-1"


@pytest.mark.anyio
async def test_resolve_args_with_llm_fallback_on_bad_json(mock_llm):
    llm = mock_llm("I cannot determine the value.")
    ctx = {1: StepResult(step_number=1, task="t", server="a", response="data")}
    result = await _resolve_args_with_llm("task", "task", "tool", "", ctx, llm)
    assert result == {}


@pytest.mark.anyio
async def test_resolve_args_with_llm_question_in_prompt():
    llm = _CapturingLLM('{"site_name": "MAIN"}')
    await _resolve_args_with_llm(
        "What sites exist?", "List sites", "sites", "", {}, llm  # type: ignore[arg-type]
    )
    assert "What sites exist?" in llm.prompts[0]


@pytest.mark.anyio
async def test_resolve_args_with_llm_tool_in_prompt():
    llm = _CapturingLLM('{}')
    await _resolve_args_with_llm("Q", "List IoT sites", "sites", "", {}, llm)  # type: ignore[arg-type]
    assert "sites" in llm.prompts[0]


@pytest.mark.anyio
async def test_resolve_args_with_llm_schema_in_prompt():
    """Tool parameter schema appears in the prompt so LLM uses correct names."""
    llm = _CapturingLLM('{"site_name": "MAIN"}')
    await _resolve_args_with_llm(  # type: ignore[arg-type]
        "Q", "List assets", "assets", "site_name: string", {}, llm
    )
    assert "site_name: string" in llm.prompts[0]


@pytest.mark.anyio
async def test_resolve_args_with_llm_unknown_schema_shows_sentinel():
    """Empty schema renders as '(unknown)' in the prompt."""
    llm = _CapturingLLM('{}')
    await _resolve_args_with_llm("Q", "task", "tool", "", {}, llm)  # type: ignore[arg-type]
    assert "(unknown)" in llm.prompts[0]


@pytest.mark.anyio
async def test_resolve_args_with_llm_context_in_prompt():
    """Prior step results appear verbatim in the generated prompt."""
    llm = _CapturingLLM('{}')
    ctx = {1: StepResult(step_number=1, task="t", server="a", response="step-one-result")}
    await _resolve_args_with_llm("Q", "task", "tool", "", ctx, llm)  # type: ignore[arg-type]
    assert "step-one-result" in llm.prompts[0]


@pytest.mark.anyio
async def test_resolve_args_with_llm_empty_context_shows_none():
    llm = _CapturingLLM('{}')
    await _resolve_args_with_llm("Q", "task", "tool", "", {}, llm)  # type: ignore[arg-type]
    assert "(none)" in llm.prompts[0]


# ── _resolve_args tests (simple substitution, kept for reference) ─────────────


def test_resolve_args_no_placeholders():
    args = {"site_name": "MAIN", "limit": 10}
    assert _resolve_args(args, {}) == args


def test_resolve_args_replaces_placeholder():
    ctx = {1: StepResult(step_number=1, task="t", server="a", response="MAIN")}
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


def test_parse_json_unrecoverable_returns_none():
    assert _parse_json("no json here") is None


def test_parse_json_empty_object_returns_empty_dict():
    assert _parse_json("{}") == {}


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
