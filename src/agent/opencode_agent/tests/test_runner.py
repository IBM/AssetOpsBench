"""Unit tests for OpenCodeAgentRunner helpers."""

from __future__ import annotations

from pathlib import Path

from agent.models import ToolCall
from agent.opencode_agent.runner import (
    OpenCodeAgentRunner,
    _build_mcp_config,
    _build_opencode_config,
    _build_permissions,
    _build_trajectory_from_events,
    _json_events,
    _resolve_opencode_model_and_provider,
)


def test_build_mcp_config_entrypoint():
    config = _build_mcp_config({"iot": "iot-mcp-server"}, cwd=Path("/repo"))
    assert config["iot"] == {
        "type": "local",
        "command": ["uv", "run", "iot-mcp-server"],
        "cwd": "/repo",
        "enabled": True,
        "timeout": 30000,
    }


def test_build_mcp_config_path():
    config = _build_mcp_config({"custom": Path("/tmp/server.py")}, cwd=Path("/repo"))
    assert config["custom"]["command"] == ["uv", "run", "/tmp/server.py"]


def test_build_permissions_default_safe():
    permission = _build_permissions(["iot", "wo"])
    assert permission["iot_*"] == "allow"
    assert permission["wo_*"] == "allow"
    assert permission["bash"] == "deny"
    assert permission["edit"] == "deny"
    assert permission["websearch"] == "deny"
    assert permission["question"] == "deny"


def test_build_permissions_allows_opt_in_tools():
    permission = _build_permissions(
        ["iot"], allow_bash=True, allow_edit=True, allow_web=True
    )
    assert permission["bash"] == "allow"
    assert permission["edit"] == "allow"
    assert permission["webfetch"] == "allow"
    assert permission["websearch"] == "allow"


def test_resolve_direct_opencode_model():
    model, provider, env = _resolve_opencode_model_and_provider("opencode/gpt-5")
    assert model == "opencode/gpt-5"
    assert provider == {}
    assert env == {}


def test_resolve_litellm_model(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    model, provider, env = _resolve_opencode_model_and_provider(
        "litellm_proxy/azure/gpt-5.4"
    )
    assert model == "litellm-proxy/azure/gpt-5.4"
    assert provider["litellm-proxy"]["npm"] == "@ai-sdk/openai-compatible"
    assert provider["litellm-proxy"]["options"]["baseURL"] == "http://localhost:4000"
    assert provider["litellm-proxy"]["models"]["azure/gpt-5.4"]["name"] == "azure/gpt-5.4"
    assert env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "sk-test"


def test_resolve_tokenrouter_model(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-test")
    model, provider, env = _resolve_opencode_model_and_provider("tokenrouter/MiniMax-M3")
    assert model == "tokenrouter/MiniMax-M3"
    assert provider["tokenrouter"]["npm"] == "@ai-sdk/openai-compatible"
    assert provider["tokenrouter"]["options"]["baseURL"] == "https://router.example/v1"
    assert provider["tokenrouter"]["models"]["MiniMax-M3"]["name"] == "MiniMax-M3"
    assert env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "tr-test"


def test_build_opencode_config_includes_agent_and_mcp():
    config, env, opencode_model = _build_opencode_config(
        model="opencode/gpt-5",
        agent_name="assetops",
        max_steps=7,
        server_paths={"iot": "iot-mcp-server"},
    )
    assert env == {}
    assert opencode_model == "opencode/gpt-5"
    assert config["agent"]["assetops"]["steps"] == 7
    assert config["agent"]["assetops"]["permission"]["iot_*"] == "allow"
    assert config["mcp"]["iot"]["command"] == ["uv", "run", "iot-mcp-server"]


def test_json_events_parses_ndjson_and_plain_lines():
    events, plain = _json_events('{"type":"a"}\nnot-json\n{"type":"b"}\n')
    assert [event["type"] for event in events] == ["a", "b"]
    assert plain == ["not-json"]


def test_build_trajectory_from_text_and_tool_parts():
    events = [
        {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "id": "tool_1",
                    "type": "tool",
                    "tool": "iot_get_asset",
                    "input": {"asset_id": "CH-6"},
                    "output": {"name": "Chiller 6"},
                }
            },
        },
        {
            "type": "message.part.updated",
            "properties": {
                "part": {
                    "id": "text_1",
                    "type": "text",
                    "text": "Chiller 6 is online.",
                }
            },
        },
        {"usage": {"input_tokens": 100, "output_tokens": 25}},
    ]
    answer, trajectory = _build_trajectory_from_events(events, [])

    assert answer == "Chiller 6 is online."
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].input_tokens == 100
    assert trajectory.turns[0].output_tokens == 25
    assert isinstance(trajectory.turns[0].tool_calls[0], ToolCall)
    assert trajectory.turns[0].tool_calls[0].name == "iot_get_asset"


def test_runner_defaults():
    runner = OpenCodeAgentRunner(server_paths={}, model="opencode/gpt-5")
    assert runner._model_id == "opencode/gpt-5"
    assert runner._opencode_model == "opencode/gpt-5"
    assert runner._agent_name == "assetops"
