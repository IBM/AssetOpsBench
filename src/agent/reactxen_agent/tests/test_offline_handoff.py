"""Regression cases for network isolation and incomplete reflection handoffs."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.models import AgentResult, ToolCall, Trajectory, TurnRecord
from agent.reactxen_agent.models import Outcome, ReviewVerdict, RoundRecord
from agent.reactxen_agent.reflection_plan import (
    ReflectionFinishParams,
    build_reflection_finish_tool,
)
from agent.reactxen_agent.runner import ReActXenAgentRunner
from agent.stirrup_agent.trajectory import final_answer


def plan(**overrides):
    value = {
        "selected_alternative": "alternative_1",
        "selection_reason": "C is supported by the failure-code filter",
        "alternative_assessments": [
            {
                "alternative_id": "alternative_1",
                "decision": "selected",
                "reason": "explicit breakdown evidence",
                "evidence_paths": ["mcp_results/orders.json"],
            },
            {
                "alternative_id": "alternative_2",
                "decision": "rejected",
                "reason": "undefined servicing flag",
                "evidence_paths": ["review/verify.json"],
            },
        ],
        "evidence_paths": ["mcp_results/orders.json"],
        "candidate_answer": {"r2_pit": "C"},
        "correction_steps": ["Recompute the selected filter and verify candidate C"],
    }
    value.update(overrides)
    return value


@pytest.mark.anyio
async def test_docker_has_no_network_or_automatic_image_download(monkeypatch, tmp_path):
    import agent.stirrup_agent.offline_docker as module

    client = MagicMock()
    client.images.get.return_value.id = "sha256:already-local"
    monkeypatch.setattr(module.docker, "from_env", lambda: client)
    provider = module.OfflineDockerCodeExecToolProvider(
        "assetops-code", temp_base_dir=tmp_path
    )
    monkeypatch.setattr(provider, "_fix_file_ownership", AsyncMock())
    async with provider:
        kwargs = client.containers.run.call_args.kwargs
        assert kwargs["network_mode"] == "none"
        assert kwargs["environment"] is None
        assert "NET_ADMIN" in kwargs["cap_drop"]
    client.images.get.assert_called_once_with("assetops-code")
    client.images.pull.assert_not_called()


def test_no_finish_does_not_fall_back_to_unfinished_text():
    history = [
        [SimpleNamespace(role="assistant", content="unfinished draft", tool_calls=[])]
    ]
    assert final_answer(history, None) == "unfinished draft"  # legacy behavior
    assert final_answer(history, None, require_finish=True) == ""
    assert (
        final_answer(
            history, SimpleNamespace(answer="final report"), require_finish=True
        )
        == "final report"
    )


@pytest.mark.anyio
async def test_finish_rejects_omitted_alternative_and_accepts_complete_selection():
    tool = build_reflection_finish_tool(["alternative_1", "alternative_2"])
    params = ReflectionFinishParams(
        **plan(alternative_assessments=plan()["alternative_assessments"][:1])
    )
    result = await tool.executor(params)
    assert not result.success
    assert "every offered alternative" in result.content
    result = await tool.executor(ReflectionFinishParams(**plan()))
    assert result.success


def test_reflection_finish_rejects_unstructured_or_deliberative_output():
    with pytest.raises(ValueError):
        ReflectionFinishParams(answer="unfinished draft")
    with pytest.raises(ValueError):
        ReflectionFinishParams(answer="<think>unfinished</think>")


@pytest.mark.anyio
async def test_incomplete_reflection_never_produces_guidance():
    runner = ReActXenAgentRunner()
    workspace = MagicMock(write_file_bytes=AsyncMock())
    r = RoundRecord(
        0,
        "actor answer",
        Trajectory(),
        review=ReviewVerdict(Outcome.PARTIAL, alternatives=["try C", "try D"]),
    )
    trajectory = Trajectory(
        turns=[TurnRecord(0, "", [ToolCall("code_exec", {})], 13, 5)]
    )
    critic = MagicMock()
    critic.run = AsyncMock(
        return_value=AgentResult(
            "task", "unfinished draft", trajectory, completed=False
        )
    )
    runner._critic = lambda *a, **k: critic
    guidance, in_tokens, out_tokens = await runner._react_reflect("q", r, workspace)
    assert guidance == ""
    assert (in_tokens, out_tokens) == (13, 5)
    assert "reflect" in r.phase_errors
    saved = json.loads(workspace.write_file_bytes.call_args.args[1])
    assert not saved["completed"] and saved["guidance"] == ""


@pytest.mark.anyio
async def test_completed_reflection_names_selection_and_rejected_alternative():
    runner = ReActXenAgentRunner()
    workspace = MagicMock(write_file_bytes=AsyncMock())
    r = RoundRecord(
        0,
        "actor answer",
        Trajectory(),
        review=ReviewVerdict(Outcome.PARTIAL, alternatives=["try C", "try D"]),
    )
    critic = MagicMock()
    critic.run = AsyncMock(
        return_value=AgentResult("task", json.dumps(plan()), Trajectory())
    )
    runner._critic = lambda *a, **k: critic
    guidance, _, _ = await runner._react_reflect("q", r, workspace)
    assert "Selected alternative: alternative_1" in guidance
    assert "alternative_2: rejected" in guidance
    assert '"r2_pit": "C"' in guidance
    assert r.reflection_plan["selected_alternative"] == "alternative_1"


def test_reactxen_rejects_local_backend_for_offline_execution():
    with pytest.raises(ValueError, match="network-disabled Docker"):
        ReActXenAgentRunner(code_backend="local")
