"""Tests for timing fields on trajectory models and plan-execute wiring.

Timing fields (``duration_ms`` on ``ToolCall`` / ``TurnRecord`` / ``StepResult``,
``started_at`` on ``Trajectory``) must default to ``None`` so a runner that
cannot measure them cleanly still produces a valid trajectory.  The
plan-execute executor always populates ``StepResult.duration_ms``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.models import ToolCall, Trajectory, TurnRecord
from agent.plan_execute.executor import Executor
from agent.plan_execute.models import Plan, PlanStep, StepResult


def test_toolcall_duration_defaults_none():
    tc = ToolCall(name="sensors", input={})
    assert tc.duration_ms is None


def test_turnrecord_duration_defaults_none():
    turn = TurnRecord(index=0, text="")
    assert turn.duration_ms is None


def test_trajectory_started_at_defaults_none():
    traj = Trajectory()
    assert traj.started_at is None


def test_stepresult_duration_defaults_none():
    step = StepResult(step_number=1, task="t", server="iot", response="ok")
    assert step.duration_ms is None


@pytest.mark.anyio
async def test_executor_records_per_step_duration():
    """execute_plan must populate StepResult.duration_ms for every step."""
    plan = Plan(
        steps=[
            PlanStep(
                step_number=1,
                task="get sites",
                server="iot",
                tool="",
                tool_args={},
                dependencies=[],
                expected_output="MAIN",
            ),
        ],
        raw="",
    )
    executor = Executor(llm=AsyncMock(), server_paths={"iot": "iot-mcp-server"})

    with patch.object(executor, "get_server_descriptions", AsyncMock(return_value={})):
        results = await executor.execute_plan(plan, question="q")

    assert len(results) == 1
    assert results[0].duration_ms is not None
    assert results[0].duration_ms >= 0
