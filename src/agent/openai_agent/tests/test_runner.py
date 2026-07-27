"""Unit tests for OpenAIAgentRunner.

These tests patch agents.Runner.run so no real API calls are made.
"""

from __future__ import annotations

import argparse
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio
import pytest
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel

from agent.models import AgentResult, Trajectory
from agent.openai_agent.cli import _build_parser, _parse_mcp_tool_permission
from agent.openai_agent.runner import (
    OpenAIAgentRunner,
    _build_mcp_servers,
    _build_run_config,
    _build_trajectory,
    _enter_mcp_servers,
    _normalize_mcp_tool_allowlist,
    _uses_responses_api,
)

# ---------------------------------------------------------------------------
# _build_mcp_servers
# ---------------------------------------------------------------------------


def test_build_mcp_servers_entrypoint():
    specs = {"iot": "iot-mcp-server", "utilities": "utilities-mcp-server"}
    result = _build_mcp_servers(specs)
    assert len(result) == 2
    assert result[0].name == "iot"
    assert result[1].name == "utilities"
    assert result[0].tool_filter is None
    assert result[0]._needs_approval_policy is False


def test_build_mcp_servers_path():
    p = Path("/some/server.py")
    result = _build_mcp_servers({"custom": p})
    assert len(result) == 1
    assert result[0].name == "custom"


def test_build_mcp_servers_empty():
    assert _build_mcp_servers({}) == []


def test_build_mcp_servers_applies_fail_closed_tool_allowlist():
    specs = {"iot": "iot-mcp-server", "utilities": "utilities-mcp-server"}
    result = _build_mcp_servers(
        specs,
        mcp_tool_allowlist={"iot": {"sites", "asset_ids"}},
    )

    assert len(result) == 1
    assert result[0].tool_filter == {"allowed_tool_names": ["asset_ids", "sites"]}
    assert all(server._needs_approval_policy is False for server in result)


def test_normalize_mcp_tool_allowlist_rejects_unknown_server():
    with pytest.raises(ValueError, match="unknown servers: typo"):
        _normalize_mcp_tool_allowlist(
            {"iot": "iot-mcp-server"},
            {"typo": {"sites"}},
        )


def test_normalize_mcp_tool_allowlist_rejects_string_value():
    with pytest.raises(TypeError, match="must be a collection"):
        _normalize_mcp_tool_allowlist(
            {"iot": "iot-mcp-server"},
            {"iot": "sites"},
        )


@pytest.mark.anyio
async def test_enter_mcp_servers_connects_concurrently():
    entered: list[str] = []
    exited: list[str] = []
    both_entered = anyio.Event()

    class FakeServer:
        def __init__(self, name: str):
            self.name = name

        async def __aenter__(self):
            entered.append(self.name)
            if len(entered) == 2:
                both_entered.set()
            await both_entered.wait()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            exited.append(self.name)

    servers = [FakeServer("one"), FakeServer("two")]
    async with AsyncExitStack() as stack:
        with anyio.fail_after(1):
            active = await _enter_mcp_servers(stack, servers)
        assert active == servers

    assert set(entered) == {"one", "two"}
    assert set(exited) == {"one", "two"}


# ---------------------------------------------------------------------------
# _build_run_config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("tokenrouter/openai/gpt-5.5", True),
        ("tokenrouter/openai/gpt-5.6-sol", True),
        ("tokenrouter/openai/gpt-4.1", False),
        ("tokenrouter/anthropic/claude-opus-4.8", False),
        ("litellm_proxy/openai/gpt-5.5", False),
        ("gpt-5.5", False),
    ],
)
def test_uses_responses_api(model_id, expected):
    assert _uses_responses_api(model_id) is expected


def test_build_run_config_no_prefix_raises():
    with pytest.raises(ValueError, match="must start with"):
        _build_run_config("gpt-4o")


def test_build_run_config_litellm_prefix(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    config = _build_run_config("litellm_proxy/Azure/gpt-5-2025-08-07")
    model = config.model_provider.get_model("Azure/gpt-5-2025-08-07")
    assert isinstance(model, OpenAIChatCompletionsModel)


def test_build_run_config_tokenrouter_openai_gpt5_uses_responses(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "http://localhost:4001")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "sk-test")
    config = _build_run_config("tokenrouter/openai/gpt-5.6-sol")
    model = config.model_provider.get_model("openai/gpt-5.6-sol")
    assert isinstance(model, OpenAIResponsesModel)
    assert config.tracing_disabled is True


def test_build_run_config_other_tokenrouter_model_uses_chat_completions(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "http://localhost:4001")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "sk-test")
    config = _build_run_config("tokenrouter/MiniMax-M3")
    model = config.model_provider.get_model("MiniMax-M3")
    assert isinstance(model, OpenAIChatCompletionsModel)


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
    assert "iot" in runner._server_paths


def test_runner_custom_server_paths(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    paths = {"iot": "iot-mcp-server"}
    runner = OpenAIAgentRunner(server_paths=paths)
    assert runner._server_paths == paths


def test_runner_normalizes_mcp_tool_allowlist(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    runner = OpenAIAgentRunner(
        server_paths={"iot": "iot-mcp-server", "utilities": "utilities-mcp-server"},
        mcp_tool_allowlist={"iot": {"sites"}},
    )

    assert runner._mcp_tool_allowlist == {
        "iot": ("sites",),
        "utilities": (),
    }


def test_runner_custom_model(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "http://localhost:4001")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "sk-test")
    runner = OpenAIAgentRunner(model="tokenrouter/openai/gpt-4.1-mini")
    assert runner._model == "openai/gpt-4.1-mini"


def test_runner_unprefixed_model_raises():
    with pytest.raises(ValueError, match="must start with"):
        OpenAIAgentRunner(model="gpt-4.1-mini")


def test_runner_litellm_model(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    runner = OpenAIAgentRunner(model="litellm_proxy/Azure/gpt-5-2025-08-07")
    assert runner._model == "Azure/gpt-5-2025-08-07"
    assert runner._run_config is not None


# ---------------------------------------------------------------------------
# CLI permissions
# ---------------------------------------------------------------------------


def test_parse_mcp_tool_permission():
    assert _parse_mcp_tool_permission("iot/sites") == ("iot", "sites")


@pytest.mark.parametrize("value", ["sites", "/sites", "iot/"])
def test_parse_mcp_tool_permission_rejects_invalid_value(value):
    with pytest.raises(argparse.ArgumentTypeError, match="expected SERVER/TOOL"):
        _parse_mcp_tool_permission(value)


def test_cli_collects_mcp_tool_permissions():
    args = _build_parser().parse_args(
        [
            "--allow-mcp-tool",
            "iot/sites",
            "--allow-mcp-tool",
            "utilities/current_date_time",
            "question",
        ]
    )

    assert args.allow_mcp_tool == [
        ("iot", "sites"),
        ("utilities", "current_date_time"),
    ]


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


def _make_tool_output_item(output, call_id: str | None = None):
    """Create a fake ToolCallOutputItem."""
    raw_item = {"call_id": call_id} if call_id else None
    return SimpleNamespace(
        type="tool_call_output_item",
        raw_item=raw_item,
        output=output,
    )


def _make_raw_message(text: str):
    content = SimpleNamespace(text=text)
    return SimpleNamespace(type="message", content=[content])


def _make_raw_tool_call(name: str, args: str, call_id: str = "call_1"):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=args,
        call_id=call_id,
    )


def _make_usage(input_tokens: int, output_tokens: int):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


def _make_raw_response(outputs=None, usage=None):
    return SimpleNamespace(output=outputs or [], usage=usage)


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
    raw = [_make_raw_response([_make_raw_message("Hello world")])]
    result = _make_run_result([_make_message_item("Hello world")], raw)
    traj = _build_trajectory(result)
    assert len(traj.turns) == 1
    assert traj.turns[0].text == "Hello world"
    assert traj.turns[0].tool_calls == []


def test_build_trajectory_tool_calls():
    items = [
        _make_tool_call_item("sensors", '{"asset_id": "CH-6"}', "call_1"),
        _make_tool_output_item("5 sensors found", "call_1"),
        _make_message_item("Chiller 6 has 5 sensors."),
    ]
    raw = [
        _make_raw_response(
            [_make_raw_tool_call("sensors", '{"asset_id": "CH-6"}', "call_1")]
        ),
        _make_raw_response([_make_raw_message("Chiller 6 has 5 sensors.")]),
    ]
    result = _make_run_result(items, raw)
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
    raw_responses = [
        _make_raw_response([_make_raw_message("Hello")], _make_usage(100, 25))
    ]
    result = _make_run_result(items, raw_responses)
    traj = _build_trajectory(result)
    assert traj.turns[0].input_tokens == 100
    assert traj.turns[0].output_tokens == 25
    assert traj.total_input_tokens == 100
    assert traj.total_output_tokens == 25


def test_build_trajectory_invalid_json_args():
    raw = [_make_raw_response([_make_raw_tool_call("sensors", "not-json", "call_1")])]
    result = _make_run_result([], raw)
    traj = _build_trajectory(result)
    assert traj.turns[0].tool_calls[0].input == {"raw": "not-json"}


def test_build_trajectory_multiple_tool_calls():
    items = [
        _make_tool_call_item("sites", "{}", "call_1"),
        _make_tool_call_item("assets", '{"site_id": "MAIN"}', "call_2"),
        _make_tool_output_item(["MAIN"], "call_1"),
        _make_tool_output_item(["Chiller 6"], "call_2"),
        _make_message_item("Found Chiller 6 at site MAIN."),
    ]
    # Two turns: (tool calls) and (message), so two raw_responses
    raw = [
        _make_raw_response(
            [
                _make_raw_tool_call("sites", "{}", "call_1"),
                _make_raw_tool_call("assets", '{"site_id": "MAIN"}', "call_2"),
            ],
            _make_usage(50, 10),
        ),
        _make_raw_response(
            [_make_raw_message("Found Chiller 6 at site MAIN.")],
            _make_usage(80, 15),
        ),
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


def test_build_trajectory_keeps_preamble_and_tool_call_in_same_turn():
    items = [
        _make_message_item("I'll check. "),
        _make_tool_call_item("sites", "{}", "call_1"),
        _make_tool_output_item(["MAIN"], "call_1"),
        _make_message_item("Found site MAIN."),
    ]
    raw = [
        _make_raw_response(
            [
                _make_raw_message("I'll check. "),
                _make_raw_tool_call("sites", "{}", "call_1"),
            ],
            _make_usage(50, 10),
        ),
        _make_raw_response(
            [_make_raw_message("Found site MAIN.")],
            _make_usage(80, 15),
        ),
    ]
    traj = _build_trajectory(_make_run_result(items, raw))

    assert len(traj.turns) == 2
    assert traj.turns[0].text == "I'll check. "
    assert traj.turns[0].tool_calls[0].output == ["MAIN"]
    assert traj.turns[0].input_tokens == 50
    assert traj.turns[1].text == "Found site MAIN."
    assert traj.turns[1].input_tokens == 80


def test_build_trajectory_matches_parallel_outputs_by_call_id():
    items = [
        _make_tool_call_item("sites", "{}", "call_1"),
        _make_tool_call_item("assets", "{}", "call_2"),
        _make_tool_output_item(["Chiller 6"], "call_2"),
        _make_tool_output_item(["MAIN"], "call_1"),
    ]
    raw = [
        _make_raw_response(
            [
                _make_raw_tool_call("sites", "{}", "call_1"),
                _make_raw_tool_call("assets", "{}", "call_2"),
            ]
        )
    ]
    traj = _build_trajectory(_make_run_result(items, raw))

    assert traj.all_tool_calls[0].output == ["MAIN"]
    assert traj.all_tool_calls[1].output == ["Chiller 6"]


def test_build_trajectory_preserves_usage_for_response_without_visible_items():
    raw = [_make_raw_response([], _make_usage(12, 3))]
    traj = _build_trajectory(_make_run_result([], raw))

    assert len(traj.turns) == 1
    assert traj.total_input_tokens == 12
    assert traj.total_output_tokens == 3


# ---------------------------------------------------------------------------
# OpenAIAgentRunner.run
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_run_returns_agent_result():
    fake_result = _make_run_result(
        [_make_message_item("42 sensors found")],
        [_make_raw_response([_make_raw_message("42 sensors found")])],
    )
    fake_result.final_output = "42 sensors found"

    with (
        patch("agent.openai_agent.runner.Runner") as MockRunner,
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=[]),
        patch("agent.openai_agent.runner._build_run_config", return_value=None),
    ):
        MockRunner.run = AsyncMock(return_value=fake_result)

        runner = OpenAIAgentRunner(server_paths={})
        result = await runner.run("How many sensors are there?")

    assert isinstance(result, AgentResult)
    assert result.question == "How many sensors are there?"
    assert result.answer == "42 sensors found"
    assert isinstance(result.trajectory, Trajectory)
    agent = MockRunner.run.await_args.args[0]
    assert agent.tools == []
    assert agent.mcp_config == {"include_server_in_tool_names": True}


@pytest.mark.anyio
async def test_run_collects_trajectory():
    items = [
        _make_tool_call_item("sensors", '{"asset_id": "CH-6"}', "call_1"),
        _make_tool_output_item("sensor data"),
        _make_message_item("Chiller 6 has 5 sensors."),
    ]
    raw_responses = [
        _make_raw_response(
            [_make_raw_tool_call("sensors", '{"asset_id": "CH-6"}', "call_1")],
            _make_usage(100, 20),
        ),
        _make_raw_response(
            [_make_raw_message("Chiller 6 has 5 sensors.")],
            _make_usage(150, 30),
        ),
    ]
    fake_result = _make_run_result(items, raw_responses)
    fake_result.final_output = "Chiller 6 has 5 sensors."

    with (
        patch("agent.openai_agent.runner.Runner") as MockRunner,
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=[]),
        patch("agent.openai_agent.runner._build_run_config", return_value=None),
    ):
        MockRunner.run = AsyncMock(return_value=fake_result)

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
    ):
        MockRunner.run = AsyncMock(return_value=fake_result)

        runner = OpenAIAgentRunner(server_paths={})
        result = await runner.run("What time is it?")

    assert result.answer == ""
    assert isinstance(result.trajectory, Trajectory)
    assert result.trajectory.turns == []


@pytest.mark.anyio
async def test_async_context_reuses_and_closes_mcp_servers():
    fake_result = _make_run_result(
        [_make_message_item("done")],
        [_make_raw_response([_make_raw_message("done")])],
    )
    entered = 0
    exited = 0

    class FakeServer:
        name = "fake"

        async def __aenter__(self):
            nonlocal entered
            entered += 1
            return self

        async def __aexit__(self, exc_type, exc, tb):
            nonlocal exited
            exited += 1

    fake_server = FakeServer()
    with (
        patch("agent.openai_agent.runner.Runner") as MockRunner,
        patch(
            "agent.openai_agent.runner._build_mcp_servers",
            return_value=[fake_server],
        ) as build_servers,
        patch("agent.openai_agent.runner._build_run_config", return_value=None),
    ):
        MockRunner.run = AsyncMock(return_value=fake_result)
        runner = OpenAIAgentRunner(server_paths={})

        async with runner:
            await runner.run("first")
            await runner.run("second")

    assert build_servers.call_count == 1
    assert MockRunner.run.await_count == 2
    assert entered == 1
    assert exited == 1
