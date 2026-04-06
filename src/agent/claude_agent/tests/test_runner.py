"""Unit tests for ClaudeAgentRunner.

These tests patch claude_agent_sdk.query so no real API calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    assert result.history is None


@pytest.mark.anyio
async def test_run_empty_result():
    async def fake_query(prompt, options):
        return
        yield  # make it an async generator

    with patch("agent.claude_agent.runner.query", side_effect=fake_query):
        runner = ClaudeAgentRunner(server_paths={})
        result = await runner.run("What time is it?")

    assert result.answer == ""
    assert result.history is None
