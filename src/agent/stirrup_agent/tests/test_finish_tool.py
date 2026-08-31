"""Tests for the AssetOps-specific Stirrup finish contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from stirrup import Agent
from stirrup.core.models import AssistantMessage, ToolCall

from agent.stirrup_agent.finish_tool import (
    ASSETOPS_FINISH_TOOL,
    AssetOpsFinishParams,
    structured_finish_answer,
)


def test_finish_params_require_a_non_empty_answer():
    with pytest.raises(ValidationError):
        AssetOpsFinishParams(answer="")
    with pytest.raises(ValidationError):
        AssetOpsFinishParams(answer="  \n")


def test_finish_params_keep_reason_and_paths_optional():
    params = AssetOpsFinishParams(answer='{"count":7}')

    assert params.answer == '{"count":7}'
    assert params.reason == ""
    assert params.paths == []


@pytest.mark.anyio
async def test_finish_tool_returns_the_user_facing_answer():
    params = AssetOpsFinishParams(
        answer='["PMP-01"]',
        reason="Located the requested pump.",
    )

    result = await ASSETOPS_FINISH_TOOL.executor(params)

    assert result.success is True
    assert result.content == '["PMP-01"]'


def test_structured_finish_answer_rejects_legacy_params():
    class _LegacyFinish:
        reason = "done"

    assert structured_finish_answer(_LegacyFinish()) is None


def test_structured_finish_answer_strips_boundary_whitespace():
    params = AssetOpsFinishParams(answer="  42\n")

    assert structured_finish_answer(params) == "42"


@pytest.mark.anyio
async def test_stirrup_agent_returns_custom_finish_params():
    class _Client:
        max_tokens = 100_000
        context_window_tokens = 100_000
        model_slug = "fake/custom-finish"

        async def generate(self, messages, tools):
            assert tools["finish"].parameters is AssetOpsFinishParams
            return AssistantMessage(
                blocks=[
                    ToolCall(
                        name="finish",
                        arguments=(
                            '{"answer":"[1,2]","reason":"done","paths":[]}'
                        ),
                        tool_call_id="finish-1",
                    )
                ],
            )

    agent = Agent(
        client=_Client(),
        name="assetops-test",
        tools=[],
        finish_tool=ASSETOPS_FINISH_TOOL,
        max_turns=2,
    )

    finish_params, history, _metadata = await agent.run("Return a JSON array.")

    assert isinstance(finish_params, AssetOpsFinishParams)
    assert finish_params.answer == "[1,2]"
    assert history[-1][-1].content == "[1,2]"
