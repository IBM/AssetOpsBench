"""Unit tests for ClaudeAgentRunner.

These tests patch claude_agent_sdk.query so no real API calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.claude_agent.runner import ClaudeAgentRunner, _build_mcp_servers
from agent.plan_execute.models import AgentResult, StepResult


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
    assert runner._permission_mode == "default"
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
    assert len(result.plan.steps) == 0
    assert len(result.history) == 1
    assert isinstance(result.history[0], StepResult)
    assert result.history[0].response == "42 sensors found"
    assert result.history[0].server == "claude-agent-sdk"


@pytest.mark.anyio
async def test_run_empty_result():
    async def fake_query(prompt, options):
        return
        yield  # make it an async generator

    with patch("agent.claude_agent.runner.query", side_effect=fake_query):
        runner = ClaudeAgentRunner(server_paths={})
        result = await runner.run("What time is it?")

    assert result.answer == ""
    assert result.history[0].response == ""
