"""Deterministic adaptive-routing and bounded-recovery fixtures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.plan_execute.executor import Executor, _tool_annotations
from agent.plan_execute.models import Plan, PlanStep
from agent.plan_execute.runner import PlanExecuteRunner
from llm import LLMBackend


class RecordingLLM(LLMBackend):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_mcp_retry_safety_annotations_are_extracted_without_guessing():
    annotated = SimpleNamespace(
        annotations=SimpleNamespace(
            readOnlyHint=True, destructiveHint=False, idempotentHint=True
        )
    )

    assert _tool_annotations(annotated) == {
        "read_only": True,
        "destructive": False,
        "idempotent": True,
    }
    assert _tool_annotations(SimpleNamespace()) == {
        "read_only": None,
        "destructive": None,
        "idempotent": None,
    }


def _step(
    *,
    tool: str = "read_asset",
    server: str = "iot",
    dependencies: list[int] | None = None,
    task: str = "Read asset",
) -> PlanStep:
    return PlanStep(
        step_number=1,
        task=task,
        server=server,
        tool=tool,
        tool_args={},
        dependencies=dependencies or [],
        expected_output="asset evidence",
    )


def _tool_spec(
    *,
    name: str = "read_asset",
    read_only: bool | None = True,
    destructive: bool | None = False,
    required: bool = False,
):
    return {
        "name": name,
        "description": "fixture tool",
        "parameters": [
            {"name": "site_name", "type": "string", "required": required}
        ],
        "annotations": {
            "read_only": read_only,
            "destructive": destructive,
            "idempotent": read_only,
        },
    }


def _plan_text(
    *,
    server: str = "iot",
    tool: str = "read_asset",
    task: str = "Read asset evidence",
) -> str:
    return (
        f"#Task1: {task}\n"
        f"#Server1: {server}\n"
        f"#Tool1: {tool}\n"
        "#Dependency1: None\n"
        "#ExpectedOutput1: Asset evidence\n"
    )


@pytest.mark.anyio
async def test_successful_shallow_execution_does_not_escalate():
    llm = RecordingLLM([_plan_text(), "{}", "asset answer"])
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch(
            "agent.plan_execute.executor._call_tool",
            new=AsyncMock(return_value='{"asset":"A-1"}'),
        ),
    ):
        result = await PlanExecuteRunner(
            llm, server_paths={"iot": Path("/fake.py")}, adaptive_escalation=True
        ).run("Read A-1")

    assert result.escalation_action == "none"
    assert len(llm.prompts) == 3


@pytest.mark.anyio
async def test_successful_specialist_execution_does_not_escalate():
    llm = RecordingLLM(
        [_plan_text(server="vibration", tool="diagnose"), "{}", "healthy"]
    )
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec(name="diagnose")]),
        ),
        patch(
            "agent.plan_execute.executor._call_tool",
            new=AsyncMock(return_value='{"severity":"low"}'),
        ),
    ):
        result = await PlanExecuteRunner(
            llm,
            server_paths={"vibration": Path("/fake.py")},
            adaptive_escalation=True,
        ).run("Diagnose vibration")

    assert result.escalation_action == "none"


@pytest.mark.anyio
async def test_domain_words_alone_do_not_escalate():
    llm = RecordingLLM([_plan_text(task="Read work order failure"), "{}", "done"])
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch(
            "agent.plan_execute.executor._call_tool",
            new=AsyncMock(return_value='{"status":"complete"}'),
        ),
    ):
        result = await PlanExecuteRunner(
            llm, server_paths={"iot": Path("/fake.py")}, adaptive_escalation=True
        ).run("Review the work order failure")

    assert result.escalation_action == "none"


@pytest.mark.anyio
async def test_failed_read_only_tool_is_retried_once():
    llm = RecordingLLM(["{}", "{}"])
    call = AsyncMock(side_effect=[RuntimeError("timeout"), '{"asset":"A-1"}'])
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step()], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.success is True
    assert result.attempt_count == 2
    assert result.recovery_succeeded is True
    assert call.await_count == 2


@pytest.mark.anyio
async def test_semantically_wrong_arguments_are_repaired_after_explicit_error():
    llm = RecordingLLM(
        ['{"site_name":"main site"}', '{"site_name":"MAIN"}']
    )
    call = AsyncMock(
        side_effect=['{"error":"unknown site main site"}', '{"asset":"A-1"}']
    )
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec(required=True)]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step()], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.success is True
    assert result.tool_args == {"site_name": "MAIN"}
    assert call.call_args_list[0].args[2] == {"site_name": "main site"}
    assert call.call_args_list[1].args[2] == {"site_name": "MAIN"}


@pytest.mark.anyio
async def test_malformed_arguments_are_corrected_before_tool_call():
    llm = RecordingLLM(["not json", '{"site_name":"MAIN"}'])
    call = AsyncMock(return_value='{"asset":"A-1"}')
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec(required=True)]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step()], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.success is True
    assert result.initial_error == "Argument generation returned no parseable JSON object"
    assert result.tool_args == {"site_name": "MAIN"}
    assert call.await_count == 1


@pytest.mark.anyio
async def test_unadvertised_argument_is_repaired_before_tool_call():
    llm = RecordingLLM(['{"Description":"MAIN"}', '{}'])
    call = AsyncMock(return_value='{"sites":["MAIN"]}')
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step()], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.success is True
    assert result.initial_error == "Unexpected argument(s): Description"
    assert result.tool_args == {}
    assert call.await_count == 1


@pytest.mark.anyio
async def test_missing_artifact_retries_once_then_reports_failure():
    llm = RecordingLLM([_plan_text(), "{}", "{}"])
    call = AsyncMock(return_value='{"error":"No such file: evidence.json"}')
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = await PlanExecuteRunner(
            llm, server_paths={"iot": Path("/fake.py")}, adaptive_escalation=True
        ).run("Read evidence.json")

    assert result.escalation_action == "report_failure"
    assert result.trajectory[0].retry_exhausted is True
    assert call.await_count == 2
    assert result.answer.startswith("Unable to complete the request")


@pytest.mark.anyio
async def test_side_effecting_tool_failure_is_not_replayed():
    llm = RecordingLLM(["{}"])
    call = AsyncMock(side_effect=RuntimeError("connection dropped"))
    write_spec = _tool_spec(
        name="create_workorder", read_only=False, destructive=True
    )
    executor = Executor(llm, server_paths={"wo": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[write_spec]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step(server="wo", tool="create_workorder")], raw=""),
            "Q",
            adaptive_recovery=True,
        ))[0]

    assert result.success is False
    assert result.retry_blocked is True
    assert call.await_count == 1
    assert len(llm.prompts) == 1


@pytest.mark.anyio
async def test_unknown_tool_safety_prohibits_replay():
    llm = RecordingLLM(["{}"])
    call = AsyncMock(side_effect=RuntimeError("outcome unknown"))
    unknown_spec = _tool_spec(read_only=None, destructive=None)
    executor = Executor(llm, server_paths={"custom": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[unknown_spec]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step(server="custom")], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.retry_safety == "unknown"
    assert result.retry_blocked is True
    assert call.await_count == 1


@pytest.mark.anyio
async def test_empty_read_only_output_is_retried_once():
    llm = RecordingLLM(["{}", "{}"])
    call = AsyncMock(side_effect=["", '{"asset":"A-1"}'])
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step()], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.recovery_succeeded is True
    assert result.initial_error == "Tool returned empty output"
    assert call.await_count == 2


@pytest.mark.anyio
async def test_changed_recovery_arguments_can_legitimately_return_zero_results():
    llm = RecordingLLM(
        ['{"site_name":"main site"}', '{"site_name":"MAIN"}']
    )
    call = AsyncMock(
        side_effect=[RuntimeError("temporary read failure"), '{"total":0,"assets":[]}']
    )
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec(required=True)]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step()], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.success is True
    assert result.recovery_succeeded is True


@pytest.mark.anyio
async def test_reasoning_only_step_is_handled_normally():
    executor = Executor(RecordingLLM([]), server_paths={})
    result = await executor.execute_step(
        _step(server="none", tool="none", task="Reason over evidence"), {}, "Q"
    )

    assert result.success is True
    assert result.response == "asset evidence"


@pytest.mark.anyio
async def test_verifier_failure_falls_back_to_deterministic_failure():
    llm = RecordingLLM(
        [_plan_text(server="wo", tool="create_workorder"), "{}", RuntimeError("LLM down")]
    )
    write_spec = _tool_spec(
        name="create_workorder", read_only=False, destructive=True
    )
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[write_spec]),
        ),
        patch(
            "agent.plan_execute.executor._call_tool",
            new=AsyncMock(side_effect=RuntimeError("outcome unknown")),
        ),
    ):
        result = await PlanExecuteRunner(
            llm, server_paths={"wo": Path("/fake.py")}, adaptive_escalation=True
        ).run("Create a work order")

    assert result.escalation_action == "verify"
    assert result.answer.startswith("Unable to complete the request")


@pytest.mark.anyio
async def test_recovery_model_failure_is_captured_and_bounded():
    llm = RecordingLLM(["{}", RuntimeError("recovery model down")])
    call = AsyncMock(side_effect=RuntimeError("read timeout"))
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = (await executor.execute_plan(
            Plan([_step()], raw=""), "Q", adaptive_recovery=True
        ))[0]

    assert result.success is False
    assert result.recovery_attempted is True
    assert result.retry_exhausted is True
    assert result.error == "recovery model down"
    assert call.await_count == 1


@pytest.mark.anyio
async def test_recovery_evidence_reaches_final_summarization():
    llm = RecordingLLM(
        [_plan_text(), "not json", '{"site_name":"MAIN"}', "A-1 recovered"]
    )
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec(required=True)]),
        ),
        patch(
            "agent.plan_execute.executor._call_tool",
            new=AsyncMock(return_value='{"asset":"A-1","site":"MAIN"}'),
        ),
    ):
        result = await PlanExecuteRunner(
            llm, server_paths={"iot": Path("/fake.py")}, adaptive_escalation=True
        ).run("Read A-1")

    assert result.answer == "A-1 recovered"
    assert result.escalation_action == "retry_step"
    assert '"asset":"A-1"' in llm.prompts[-1]
    assert 'args: {"site_name": "MAIN"}' in llm.prompts[-1]


@pytest.mark.anyio
async def test_failed_recovery_is_bounded_and_accurately_reported():
    llm = RecordingLLM([_plan_text(), "{}", "{}"])
    call = AsyncMock(side_effect=RuntimeError("still unavailable"))
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = await PlanExecuteRunner(
            llm, server_paths={"iot": Path("/fake.py")}, adaptive_escalation=True
        ).run("Read A-1")

    assert call.await_count == 2
    assert result.trajectory[0].attempt_count == 2
    assert result.trajectory[0].retry_exhausted is True
    assert "still unavailable" in result.answer


@pytest.mark.anyio
async def test_adaptive_disabled_preserves_legacy_bad_json_fallback():
    llm = RecordingLLM([_plan_text(), "not json", "legacy answer"])
    call = AsyncMock(return_value='{"asset":"legacy"}')
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec(required=True)]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        result = await PlanExecuteRunner(
            llm, server_paths={"iot": Path("/fake.py")}, adaptive_escalation=False
        ).run("Read A-1")

    assert result.answer == "legacy answer"
    assert result.escalation_action is None
    assert result.trajectory[0].tool_args == {}
    assert call.await_count == 1


@pytest.mark.anyio
async def test_failed_dependency_is_not_executed_in_adaptive_mode():
    first = _step()
    second = PlanStep(
        step_number=2,
        task="Use asset evidence",
        server="iot",
        tool="read_asset",
        tool_args={},
        dependencies=[1],
        expected_output="derived evidence",
    )
    llm = RecordingLLM(["{}", "{}"])
    call = AsyncMock(side_effect=RuntimeError("offline"))
    executor = Executor(llm, server_paths={"iot": Path("/fake.py")})
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=[_tool_spec()]),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call),
    ):
        results = await executor.execute_plan(
            Plan([first, second], raw=""), "Q", adaptive_recovery=True
        )

    assert call.await_count == 2  # initial call plus one bounded retry for step 1
    assert results[1].failure_kind == "failed_dependency"
    assert results[1].attempt_count == 0
