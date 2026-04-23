"""Unit tests for DeepAgentRunner.

These tests patch deep-agents and langchain-mcp-adapters so no real API calls
or MCP subprocesses are started.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.deep_agent.runner import (
    DeepAgentRunner,
    _build_chat_model,
    _build_mcp_connections,
    _build_trajectory,
)
from agent.models import AgentResult, Trajectory


# ---------------------------------------------------------------------------
# _build_mcp_connections
# ---------------------------------------------------------------------------


def test_build_mcp_connections_entrypoint():
    conns = _build_mcp_connections({"iot": "iot-mcp-server"})
    assert set(conns.keys()) == {"iot"}
    iot = conns["iot"]
    assert iot["transport"] == "stdio"
    assert iot["command"] == "uv"
    assert iot["args"] == ["run", "iot-mcp-server"]
    assert "cwd" in iot


def test_build_mcp_connections_path():
    p = Path("/some/server.py")
    conns = _build_mcp_connections({"custom": p})
    assert conns["custom"]["args"] == ["run", "/some/server.py"]


def test_build_mcp_connections_empty():
    assert _build_mcp_connections({}) == {}


# ---------------------------------------------------------------------------
# _build_chat_model
# ---------------------------------------------------------------------------


def test_build_chat_model_litellm(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    model = _build_chat_model("litellm_proxy/aws/claude-opus-4-6")
    assert model is not None
    assert getattr(model, "model_name", None) == "aws/claude-opus-4-6"


def test_build_chat_model_missing_env_raises(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LITELLM_BASE_URL"):
        _build_chat_model("litellm_proxy/aws/claude-opus-4-6")


# ---------------------------------------------------------------------------
# DeepAgentRunner.__init__
# ---------------------------------------------------------------------------


def test_runner_defaults():
    runner = DeepAgentRunner()
    assert runner._model_id == "litellm_proxy/aws/claude-opus-4-6"
    assert runner._recursion_limit == 100
    assert "iot" in runner._resolved_server_paths


def test_runner_custom_server_paths():
    paths = {"iot": "iot-mcp-server"}
    runner = DeepAgentRunner(server_paths=paths)
    assert runner._resolved_server_paths == paths


def test_runner_custom_model():
    runner = DeepAgentRunner(model="anthropic:claude-sonnet-4-6")
    assert runner._model_id == "anthropic:claude-sonnet-4-6"


def test_runner_custom_recursion_limit():
    runner = DeepAgentRunner(recursion_limit=50)
    assert runner._recursion_limit == 50


# ---------------------------------------------------------------------------
# _build_trajectory
# ---------------------------------------------------------------------------


def test_build_trajectory_empty():
    traj = _build_trajectory([])
    assert isinstance(traj, Trajectory)
    assert traj.turns == []


def test_build_trajectory_message_only():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="Hello world"),
    ]
    traj = _build_trajectory(messages)
    assert len(traj.turns) == 1
    assert traj.turns[0].text == "Hello world"
    assert traj.turns[0].tool_calls == []


def test_build_trajectory_tool_calls_and_outputs():
    messages = [
        HumanMessage(content="question"),
        AIMessage(
            content="",
            tool_calls=[{"name": "sensors", "args": {"asset_id": "CH-6"}, "id": "c1"}],
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        ),
        ToolMessage(content="5 sensors found", tool_call_id="c1"),
        AIMessage(
            content="Chiller 6 has 5 sensors.",
            usage_metadata={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
        ),
    ]
    traj = _build_trajectory(messages)
    assert len(traj.turns) == 2
    # First turn: tool call
    assert len(traj.turns[0].tool_calls) == 1
    tc = traj.turns[0].tool_calls[0]
    assert tc.name == "sensors"
    assert tc.input == {"asset_id": "CH-6"}
    assert tc.id == "c1"
    assert tc.output == "5 sensors found"
    assert traj.turns[0].input_tokens == 100
    assert traj.turns[0].output_tokens == 20
    # Second turn: final message
    assert traj.turns[1].text == "Chiller 6 has 5 sensors."
    assert traj.total_input_tokens == 250
    assert traj.total_output_tokens == 50


def test_build_trajectory_list_content():
    messages = [
        AIMessage(content=[{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}])
    ]
    traj = _build_trajectory(messages)
    assert traj.turns[0].text == "part one part two"


def test_build_trajectory_orphan_tool_message():
    messages = [
        ToolMessage(content="stray", tool_call_id="missing"),
    ]
    traj = _build_trajectory(messages)
    # No AIMessage → no turns, and orphan ToolMessage is safely ignored.
    assert traj.turns == []


def test_build_trajectory_multiple_tool_calls_one_turn():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "sites", "args": {}, "id": "c1"},
                {"name": "assets", "args": {"site_id": "MAIN"}, "id": "c2"},
            ],
            usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
        ),
        ToolMessage(content=["MAIN"], tool_call_id="c1"),
        ToolMessage(content=["Chiller 6"], tool_call_id="c2"),
        AIMessage(
            content="Found Chiller 6 at site MAIN.",
            usage_metadata={"input_tokens": 80, "output_tokens": 15, "total_tokens": 95},
        ),
    ]
    traj = _build_trajectory(messages)
    assert len(traj.turns) == 2
    assert len(traj.all_tool_calls) == 2
    assert traj.all_tool_calls[0].output == ["MAIN"]
    assert traj.all_tool_calls[1].output == ["Chiller 6"]
    assert traj.total_input_tokens == 130
    assert traj.total_output_tokens == 25


# ---------------------------------------------------------------------------
# DeepAgentRunner.run
# ---------------------------------------------------------------------------


def _fake_agent(final_state):
    """Return a mock with an ``ainvoke`` coroutine that yields *final_state*."""
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value=final_state)
    return agent


def _fake_client(tools=None):
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=tools or [])
    return client


@pytest.mark.anyio
async def test_run_returns_agent_result():
    final_state = {
        "messages": [
            HumanMessage(content="How many sensors are there?"),
            AIMessage(content="42 sensors found"),
        ]
    }

    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=MagicMock()),
        patch("deepagents.create_deep_agent", return_value=_fake_agent(final_state)),
        patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=_fake_client([]),
        ),
    ):
        runner = DeepAgentRunner(server_paths={})
        result = await runner.run("How many sensors are there?")

    assert isinstance(result, AgentResult)
    assert result.question == "How many sensors are there?"
    assert result.answer == "42 sensors found"
    assert isinstance(result.trajectory, Trajectory)
    assert len(result.trajectory.turns) == 1


@pytest.mark.anyio
async def test_run_collects_trajectory():
    final_state = {
        "messages": [
            HumanMessage(content="What sensors are on Chiller 6?"),
            AIMessage(
                content="",
                tool_calls=[{"name": "sensors", "args": {"asset_id": "CH-6"}, "id": "c1"}],
                usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            ),
            ToolMessage(content="sensor data", tool_call_id="c1"),
            AIMessage(
                content="Chiller 6 has 5 sensors.",
                usage_metadata={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
            ),
        ]
    }

    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=MagicMock()),
        patch("deepagents.create_deep_agent", return_value=_fake_agent(final_state)),
        patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=_fake_client([]),
        ),
    ):
        runner = DeepAgentRunner(server_paths={})
        result = await runner.run("What sensors are on Chiller 6?")

    assert result.answer == "Chiller 6 has 5 sensors."
    traj = result.trajectory
    assert len(traj.turns) == 2
    assert len(traj.all_tool_calls) == 1
    assert traj.all_tool_calls[0].name == "sensors"
    assert traj.all_tool_calls[0].output == "sensor data"
    assert traj.total_input_tokens == 250
    assert traj.total_output_tokens == 50


@pytest.mark.anyio
async def test_run_no_servers_skips_mcp_client():
    final_state = {
        "messages": [
            HumanMessage(content="What time is it?"),
            AIMessage(content="It is noon."),
        ]
    }

    client_ctor = MagicMock()

    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=MagicMock()),
        patch("deepagents.create_deep_agent", return_value=_fake_agent(final_state)),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", client_ctor),
    ):
        runner = DeepAgentRunner(server_paths={})
        result = await runner.run("What time is it?")

    client_ctor.assert_not_called()
    assert result.answer == "It is noon."


@pytest.mark.anyio
async def test_run_empty_messages():
    final_state: dict = {"messages": []}

    with (
        patch("agent.deep_agent.runner._build_chat_model", return_value=MagicMock()),
        patch("deepagents.create_deep_agent", return_value=_fake_agent(final_state)),
        patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=_fake_client([]),
        ),
    ):
        runner = DeepAgentRunner(server_paths={})
        result = await runner.run("q")

    assert result.answer == ""
    assert result.trajectory.turns == []
