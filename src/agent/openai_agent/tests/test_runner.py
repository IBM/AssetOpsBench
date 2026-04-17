"""Unit tests for OpenAIAgentRunner.

These tests patch agents.Runner.run so no real API calls are made.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.openai_agent.models import Trajectory
from agent.openai_agent.runner import (
    OpenAIAgentRunner,
    _build_mcp_servers,
    _build_run_config,
    _build_trajectory,
    _resolve_model,
)
from agent.models import AgentResult


# ---------------------------------------------------------------------------
# _build_mcp_servers
# ---------------------------------------------------------------------------


def test_build_mcp_servers_entrypoint():
    specs = {"iot": "iot-mcp-server", "utilities": "utilities-mcp-server"}
    result = _build_mcp_servers(specs)
    assert len(result) == 2
    assert result[0].name == "iot"
    assert result[1].name == "utilities"


def test_build_mcp_servers_path():
    p = Path("/some/server.py")
    result = _build_mcp_servers({"custom": p})
    assert len(result) == 1
    assert result[0].name == "custom"


def test_build_mcp_servers_empty():
    assert _build_mcp_servers({}) == []


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_strips_litellm_prefix():
    assert _resolve_model("litellm_proxy/Azure/gpt-5-2025-08-07") == "Azure/gpt-5-2025-08-07"


def test_resolve_model_passthrough():
    assert _resolve_model("gpt-4o") == "gpt-4o"


# ---------------------------------------------------------------------------
# _build_run_config
# ---------------------------------------------------------------------------


def test_build_run_config_no_prefix_returns_none():
    assert _build_run_config("gpt-4o") is None


def test_build_run_config_litellm_prefix(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    config = _build_run_config("litellm_proxy/Azure/gpt-5-2025-08-07")
    assert config is not None
    assert config.model_provider is not None


def test_build_run_config_missing_env_raises(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LITELLM_BASE_URL"):
        _build_run_config("litellm_proxy/Azure/gpt-5-2025-08-07")


# ---------------------------------------------------------------------------
# OpenAIAgentRunner.__init__
# ---------------------------------------------------------------------------


def test_runner_defaults(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    runner = OpenAIAgentRunner()
    assert runner._model == "azure/gpt-5.4"
    assert runner._run_config is not None
    assert runner._max_turns == 30
    assert "iot" in runner._resolved_server_paths


def test_runner_custom_server_paths(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    paths = {"iot": "iot-mcp-server"}
    runner = OpenAIAgentRunner(server_paths=paths)
    assert runner._resolved_server_paths == paths


def test_runner_custom_model():
    runner = OpenAIAgentRunner(model="gpt-4.1-mini")
    assert runner._model == "gpt-4.1-mini"


def test_runner_litellm_model(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    runner = OpenAIAgentRunner(model="litellm_proxy/Azure/gpt-5-2025-08-07")
    assert runner._model == "Azure/gpt-5-2025-08-07"
    assert runner._run_config is not None


# ---------------------------------------------------------------------------
# _build_trajectory
# ---------------------------------------------------------------------------


def _make_message_item(text: str):
    """Create a fake MessageOutputItem."""
    text_part = SimpleNamespace(text=text)
    raw = SimpleNamespace(content=[text_part])
    return SimpleNamespace(type="message_output_item", raw_item=raw)


def _make_tool_call_item(name: str, args: str, call_id: str = "call_1"):
    """Create a fake ToolCallItem."""
    raw = SimpleNamespace(name=name, arguments=args, call_id=call_id)
    return SimpleNamespace(type="tool_call_item", raw_item=raw)


def _make_tool_output_item(output):
    """Create a fake ToolCallOutputItem."""
    return SimpleNamespace(type="tool_call_output_item", output=output)


def _make_usage(input_tokens: int, output_tokens: int):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def _make_run_result(items, raw_responses=None):
    return SimpleNamespace(
        new_items=items,
        raw_responses=raw_responses or [],
        final_output="test answer",
    )


def test_build_trajectory_empty():
    result = _make_run_result([])
    traj = _build_trajectory(result)
    assert isinstance(traj, Trajectory)
    assert traj.turns == []


def test_build_trajectory_message_only():
    result = _make_run_result([_make_message_item("Hello world")])
    traj = _build_trajectory(result)
    assert len(traj.turns) == 1
    assert traj.turns[0].text == "Hello world"
    assert traj.turns[0].tool_calls == []


def test_build_trajectory_tool_calls():
    items = [
        _make_tool_call_item("sensors", '{"asset_id": "CH-6"}', "call_1"),
        _make_tool_output_item("5 sensors found"),
        _make_message_item("Chiller 6 has 5 sensors."),
    ]
    result = _make_run_result(items)
    traj = _build_trajectory(result)
    assert len(traj.turns) == 2
    # First turn: tool call + output
    assert len(traj.turns[0].tool_calls) == 1
    tc = traj.turns[0].tool_calls[0]
    assert tc.name == "sensors"
    assert tc.input == {"asset_id": "CH-6"}
    assert tc.id == "call_1"
    assert tc.output == "5 sensors found"
    # Second turn: message
    assert traj.turns[1].text == "Chiller 6 has 5 sensors."


def test_build_trajectory_token_usage():
    items = [_make_message_item("Hello")]
    raw_responses = [SimpleNamespace(usage=_make_usage(100, 25))]
    result = _make_run_result(items, raw_responses)
    traj = _build_trajectory(result)
    assert traj.turns[0].input_tokens == 100
    assert traj.turns[0].output_tokens == 25
    assert traj.total_input_tokens == 100
    assert traj.total_output_tokens == 25


def test_build_trajectory_invalid_json_args():
    items = [
        _make_tool_call_item("sensors", "not-json", "call_1"),
    ]
    result = _make_run_result(items)
    traj = _build_trajectory(result)
    assert traj.turns[0].tool_calls[0].input == {"raw": "not-json"}


def test_build_trajectory_multiple_tool_calls():
    items = [
        _make_tool_call_item("sites", "{}", "call_1"),
        _make_tool_output_item(["MAIN"]),
        _make_tool_call_item("assets", '{"site_id": "MAIN"}', "call_2"),
        _make_tool_output_item(["Chiller 6"]),
        _make_message_item("Found Chiller 6 at site MAIN."),
    ]
    # Two turns: (tool calls) and (message), so two raw_responses
    raw = [
        SimpleNamespace(usage=_make_usage(50, 10)),
        SimpleNamespace(usage=_make_usage(80, 15)),
    ]
    result = _make_run_result(items, raw)
    traj = _build_trajectory(result)
    # Both tool calls land in the same turn (no message between them)
    assert len(traj.turns) == 2
    assert len(traj.all_tool_calls) == 2
    assert traj.all_tool_calls[0].name == "sites"
    assert traj.all_tool_calls[0].output == ["MAIN"]
    assert traj.all_tool_calls[1].name == "assets"
    assert traj.all_tool_calls[1].output == ["Chiller 6"]
    assert traj.total_input_tokens == 50 + 80
    assert traj.total_output_tokens == 10 + 15


# ---------------------------------------------------------------------------
# OpenAIAgentRunner.run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_returns_agent_result():
    fake_result = _make_run_result(
        [_make_message_item("42 sensors found")],
    )
    fake_result.final_output = "42 sensors found"

    with (
        patch("agent.openai_agent.runner.Runner") as MockRunner,
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=[]),
        patch("agent.openai_agent.runner._build_run_config", return_value=None),
        patch("agent.openai_agent.runner._managed_servers") as MockCtx,
    ):
        MockRunner.run = AsyncMock(return_value=fake_result)
        MockCtx.return_value.__aenter__ = AsyncMock(return_value=[])
        MockCtx.return_value.__aexit__ = AsyncMock(return_value=False)

        runner = OpenAIAgentRunner(server_paths={})
        result = await runner.run("How many sensors are there?")

    assert isinstance(result, AgentResult)
    assert result.question == "How many sensors are there?"
    assert result.answer == "42 sensors found"
    assert isinstance(result.trajectory, Trajectory)


@pytest.mark.anyio
async def test_run_collects_trajectory():
    items = [
        _make_tool_call_item("sensors", '{"asset_id": "CH-6"}', "call_1"),
        _make_tool_output_item("sensor data"),
        _make_message_item("Chiller 6 has 5 sensors."),
    ]
    raw_responses = [
        SimpleNamespace(usage=_make_usage(100, 20)),
        SimpleNamespace(usage=_make_usage(150, 30)),
    ]
    fake_result = _make_run_result(items, raw_responses)
    fake_result.final_output = "Chiller 6 has 5 sensors."

    with (
        patch("agent.openai_agent.runner.Runner") as MockRunner,
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=[]),
        patch("agent.openai_agent.runner._build_run_config", return_value=None),
        patch("agent.openai_agent.runner._managed_servers") as MockCtx,
    ):
        MockRunner.run = AsyncMock(return_value=fake_result)
        MockCtx.return_value.__aenter__ = AsyncMock(return_value=[])
        MockCtx.return_value.__aexit__ = AsyncMock(return_value=False)

        runner = OpenAIAgentRunner(server_paths={})
        result = await runner.run("What sensors are on Chiller 6?")

    traj = result.trajectory
    assert len(traj.turns) == 2
    assert len(traj.all_tool_calls) == 1
    assert traj.all_tool_calls[0].name == "sensors"
    assert traj.total_input_tokens == 100 + 150
    assert traj.total_output_tokens == 20 + 30


@pytest.mark.anyio
async def test_run_empty_result():
    fake_result = _make_run_result([])
    fake_result.final_output = ""

    with (
        patch("agent.openai_agent.runner.Runner") as MockRunner,
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=[]),
        patch("agent.openai_agent.runner._build_run_config", return_value=None),
        patch("agent.openai_agent.runner._managed_servers") as MockCtx,
    ):
        MockRunner.run = AsyncMock(return_value=fake_result)
        MockCtx.return_value.__aenter__ = AsyncMock(return_value=[])
        MockCtx.return_value.__aexit__ = AsyncMock(return_value=False)

        runner = OpenAIAgentRunner(server_paths={})
        result = await runner.run("What time is it?")

    assert result.answer == ""
    assert isinstance(result.trajectory, Trajectory)
    assert result.trajectory.turns == []
