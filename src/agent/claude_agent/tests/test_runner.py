"""Unit tests for ClaudeAgentRunner.

These tests patch claude_agent_sdk.query so no real API calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.claude_agent.models import Trajectory
from agent.claude_agent.runner import ClaudeAgentRunner, _build_mcp_servers, _resolve_model, _sdk_env
from agent.models import AgentResult


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_strips_litellm_prefix():
    assert _resolve_model("litellm_proxy/aws/claude-opus-4-6") == "aws/claude-opus-4-6"


def test_resolve_model_passthrough():
    assert _resolve_model("claude-opus-4-6") == "claude-opus-4-6"


def test_resolve_model_stored_on_runner():
    runner = ClaudeAgentRunner(model="litellm_proxy/aws/claude-opus-4-6")
    assert runner._model == "aws/claude-opus-4-6"


def test_sdk_env_no_prefix_returns_none():
    assert _sdk_env("claude-opus-4-6") is None


def test_sdk_env_litellm_prefix_maps_vars(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-1234")
    env = _sdk_env("litellm_proxy/aws/claude-opus-4-6")
    assert env == {
        "ANTHROPIC_BASE_URL": "http://localhost:4000",
        "ANTHROPIC_API_KEY": "sk-1234",
    }


def test_sdk_env_missing_litellm_vars_returns_none(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    assert _sdk_env("litellm_proxy/aws/claude-opus-4-6") is None


# ---------------------------------------------------------------------------
# _build_mcp_servers
# ---------------------------------------------------------------------------


def test_build_mcp_servers_entrypoint():
    specs = {"iot": "iot-mcp-server", "utilities": "utilities-mcp-server"}
    result = _build_mcp_servers(specs)
    assert result["iot"] == {"command": "uv", "args": ["run", "iot-mcp-server"]}
    assert result["utilities"] == {
        "command": "uv",
        "args": ["run", "utilities-mcp-server"],
    }


def test_build_mcp_servers_path():
    p = Path("/some/server.py")
    result = _build_mcp_servers({"custom": p})
    assert result["custom"] == {"command": "uv", "args": ["run", "/some/server.py"]}


def test_build_mcp_servers_empty():
    assert _build_mcp_servers({}) == {}


# ---------------------------------------------------------------------------
# ClaudeAgentRunner.__init__
# ---------------------------------------------------------------------------


def test_runner_defaults():
    runner = ClaudeAgentRunner()
    assert runner._model == "claude-opus-4-6"
    assert runner._max_turns == 30
    assert runner._permission_mode == "bypassPermissions"
    assert "iot" in runner._resolved_server_paths


def test_runner_custom_server_paths():
    paths = {"iot": "iot-mcp-server"}
    runner = ClaudeAgentRunner(server_paths=paths)
    assert runner._resolved_server_paths == paths


# ---------------------------------------------------------------------------
# ClaudeAgentRunner.run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_returns_orchestrator_result():
    from claude_agent_sdk import ResultMessage

    mock_result = MagicMock(spec=ResultMessage)
    mock_result.result = "42 sensors found"
    mock_result.stop_reason = "end_turn"

    async def fake_query(prompt, options):
        yield mock_result

    with patch("agent.claude_agent.runner.query", side_effect=fake_query):
        runner = ClaudeAgentRunner(server_paths={"iot": "iot-mcp-server"})
        result = await runner.run("How many sensors are there?")

    assert isinstance(result, AgentResult)
    assert result.question == "How many sensors are there?"
    assert result.answer == "42 sensors found"
    assert isinstance(result.trajectory, Trajectory)
    assert result.trajectory.total_input_tokens == 0
    assert result.trajectory.total_output_tokens == 0


@pytest.mark.anyio
async def test_run_collects_trajectory():
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

    mock_tool = MagicMock(spec=ToolUseBlock)
    mock_tool.name = "sensors"
    mock_tool.input = {"asset_id": "CH-6"}
    mock_tool.id = "tu_123"

    mock_text = MagicMock(spec=TextBlock)
    mock_text.text = "Calling sensors tool..."

    mock_assistant = MagicMock(spec=AssistantMessage)
    mock_assistant.content = [mock_text, mock_tool]
    mock_assistant.usage = {"input_tokens": 100, "output_tokens": 20}

    mock_result = MagicMock(spec=ResultMessage)
    mock_result.result = "Chiller 6 has 5 sensors."
    mock_result.stop_reason = "end_turn"

    async def fake_query(prompt, options):
        yield mock_assistant
        yield mock_result

    with patch("agent.claude_agent.runner.query", side_effect=fake_query):
        runner = ClaudeAgentRunner(server_paths={})
        result = await runner.run("What sensors are on Chiller 6?")

    traj = result.trajectory
    assert isinstance(traj, Trajectory)
    assert len(traj.turns) == 1
    turn = traj.turns[0]
    assert turn.text == "Calling sensors tool..."
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "sensors"
    assert turn.tool_calls[0].input == {"asset_id": "CH-6"}
    assert turn.input_tokens == 100
    assert turn.output_tokens == 20
    assert traj.total_input_tokens == 100
    assert traj.total_output_tokens == 20
    assert len(traj.all_tool_calls) == 1


@pytest.mark.anyio
async def test_run_tool_output_captured():
    """PostToolUse hook output is attached to the matching ToolCall."""
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

    mock_tool = MagicMock(spec=ToolUseBlock)
    mock_tool.name = "sensors"
    mock_tool.input = {"asset_id": "CH-6"}
    mock_tool.id = "tu_456"

    mock_text = MagicMock(spec=TextBlock)
    mock_text.text = ""

    mock_assistant1 = MagicMock(spec=AssistantMessage)
    mock_assistant1.content = [mock_text, mock_tool]
    mock_assistant1.usage = {"input_tokens": 50, "output_tokens": 10}

    mock_assistant2 = MagicMock(spec=AssistantMessage)
    mock_assistant2.content = [MagicMock(spec=TextBlock, text="Done.")]
    mock_assistant2.usage = {"input_tokens": 60, "output_tokens": 5}

    mock_result = MagicMock(spec=ResultMessage)
    mock_result.result = "5 sensors."
    mock_result.stop_reason = "end_turn"

    async def fake_query(prompt, options):
        # Simulate hook firing between turns by calling it directly
        hook_matcher = options.hooks["PostToolUse"][0]
        hook_fn = hook_matcher.hooks[0]
        yield mock_assistant1
        await hook_fn(
            {"tool_response": {"content": [{"type": "text", "text": "sensor data"}]}},
            "tu_456",
            {},
        )
        yield mock_assistant2
        yield mock_result

    with patch("agent.claude_agent.runner.query", side_effect=fake_query):
        runner = ClaudeAgentRunner(server_paths={})
        result = await runner.run("What sensors are on Chiller 6?")

    traj = result.trajectory
    assert len(traj.turns) == 2
    tc = traj.turns[0].tool_calls[0]
    assert tc.id == "tu_456"
    assert tc.output == [{"type": "text", "text": "sensor data"}]


@pytest.mark.anyio
async def test_run_tool_output_string_response():
    """PostToolUse hook handles string tool_response (no .get)."""
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

    mock_tool = MagicMock(spec=ToolUseBlock)
    mock_tool.name = "sites"
    mock_tool.input = {}
    mock_tool.id = "tu_789"

    mock_assistant = MagicMock(spec=AssistantMessage)
    mock_assistant.content = [MagicMock(spec=TextBlock, text=""), mock_tool]
    mock_assistant.usage = {"input_tokens": 10, "output_tokens": 5}

    mock_result = MagicMock(spec=ResultMessage)
    mock_result.result = "MAIN"
    mock_result.stop_reason = "end_turn"

    async def fake_query(prompt, options):
        hook_fn = options.hooks["PostToolUse"][0].hooks[0]
        yield mock_assistant
        # Simulate SDK passing tool_response as a plain string
        await hook_fn(
            {"tool_response": '{"sites": ["MAIN"]}'},
            "tu_789",
            {},
        )
        yield mock_result

    with patch("agent.claude_agent.runner.query", side_effect=fake_query):
        runner = ClaudeAgentRunner(server_paths={})
        result = await runner.run("What sites?")

    tc = result.trajectory.turns[0].tool_calls[0]
    assert tc.output == '{"sites": ["MAIN"]}'


@pytest.mark.anyio
async def test_run_empty_result():
    async def fake_query(prompt, options):
        return
        yield  # make it an async generator

    with patch("agent.claude_agent.runner.query", side_effect=fake_query):
        runner = ClaudeAgentRunner(server_paths={})
        result = await runner.run("What time is it?")

    assert result.answer == ""
    assert isinstance(result.trajectory, Trajectory)
    assert result.trajectory.turns == []
