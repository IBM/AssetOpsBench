"""Unit tests for OpenCodeAgentRunner helpers."""

from __future__ import annotations

from pathlib import Path

from agent.models import ToolCall
from agent.opencode_agent.runner import (
    OpenCodeAgentRunner,
    _REPO_ROOT,
    _build_mcp_config,
    _build_opencode_config,
    _build_permissions,
    _build_trajectory_from_events,
    _json_events,
    _resolve_run_dir,
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
    assert permission["read"] == "deny"
    assert permission["glob"] == "deny"
    assert permission["grep"] == "deny"
    assert permission["lsp"] == "deny"
    assert permission["bash"] == "deny"
    assert permission["edit"] == "deny"
    assert permission["webfetch"] == "deny"
    assert permission["websearch"] == "deny"
    assert permission["external_directory"] == "deny"
    assert permission["question"] == "deny"


def test_build_permissions_allows_opt_in_tools():
    permission = _build_permissions(
        ["iot"],
        allow_bash=True,
        allow_edit=True,
        allow_web=True,
        allow_files=True,
    )
    assert permission["read"] == "allow"
    assert permission["glob"] == "allow"
    assert permission["grep"] == "allow"
    assert permission["lsp"] == "allow"
    assert permission["bash"] == "allow"
    assert permission["edit"] == "allow"
    assert permission["webfetch"] == "allow"
    assert permission["websearch"] == "allow"
    assert permission["external_directory"] == "deny"


def test_resolve_run_dir_defaults_to_repo_root():
    run_dir = _resolve_run_dir()
    assert run_dir == _REPO_ROOT


def test_resolve_run_dir_requires_workspace_for_file_or_code_tools():
    import pytest

    with pytest.raises(ValueError, match="workspace-dir"):
        _resolve_run_dir(allow_files=True)
    with pytest.raises(ValueError, match="workspace-dir"):
        _resolve_run_dir(allow_bash=True)
    with pytest.raises(ValueError, match="workspace-dir"):
        _resolve_run_dir(allow_edit=True)


def test_resolve_run_dir_creates_workspace(tmp_path):
    workspace = tmp_path / "opencode-run"
    run_dir = _resolve_run_dir(workspace_dir=workspace, allow_files=True)
    assert run_dir == workspace.resolve()
    assert run_dir.exists()


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


def test_resolve_rits_model_from_root_url(monkeypatch):
    monkeypatch.setenv("RITS_BASE_URL", "https://rits.example")
    monkeypatch.setenv("RITS_API_KEY", "rits-test")
    model, provider, env = _resolve_opencode_model_and_provider(
        "rits/qwen3-30b-a3b-thinking-2507"
    )
    assert model == "rits/qwen3-30b-a3b-thinking-2507"
    assert provider["rits"]["npm"] == "@ai-sdk/openai-compatible"
    assert (
        provider["rits"]["options"]["baseURL"]
        == "https://rits.example/qwen3-30b-a3b-thinking-2507/v1"
    )
    assert provider["rits"]["options"]["headers"] == {
        "RITS_API_KEY": "{env:ASSETOPSBENCH_OPENCODE_API_KEY}",
    }
    assert (
        provider["rits"]["models"]["qwen3-30b-a3b-thinking-2507"]["name"]
        == "qwen3-30b-a3b-thinking-2507"
    )
    assert env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "rits-test"


def test_resolve_rits_model_with_served_model_name(monkeypatch):
    monkeypatch.setenv(
        "RITS_BASE_URL",
        "https://rits.example/byom-gb-40365e1f",
    )
    monkeypatch.setenv("RITS_API_KEY", "rits-test")
    monkeypatch.setenv(
        "RITS_SERVED_MODEL_NAME",
        "ibm-granite/granite-guardian-4-1-8b-Q2_K",
    )
    model, provider, env = _resolve_opencode_model_and_provider(
        "rits/byom-gb-40365e1f"
    )
    assert model == "rits/ibm-granite/granite-guardian-4-1-8b-Q2_K"
    assert provider["rits"]["options"]["baseURL"] == (
        "https://rits.example/byom-gb-40365e1f/v1"
    )
    assert provider["rits"]["models"][
        "ibm-granite/granite-guardian-4-1-8b-Q2_K"
    ]["name"] == "ibm-granite/granite-guardian-4-1-8b-Q2_K"
    assert env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "rits-test"


def test_resolve_rits_model_with_custom_auth_header(monkeypatch):
    monkeypatch.setenv("RITS_BASE_URL", "https://rits.example")
    monkeypatch.setenv("RITS_API_KEY", "rits-test")
    monkeypatch.setenv("RITS_AUTH_HEADER", "user_key")
    model, provider, env = _resolve_opencode_model_and_provider(
        "rits/qwen3-30b-a3b-thinking-2507"
    )
    assert model == "rits/qwen3-30b-a3b-thinking-2507"
    assert provider["rits"]["options"]["headers"] == {
        "user_key": "{env:ASSETOPSBENCH_OPENCODE_API_KEY}",
    }
    assert env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "rits-test"


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
    assert config["agent"]["assetops"]["permission"]["read"] == "deny"
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


def test_build_trajectory_uses_last_visible_text_not_concatenated_thinking():
    events = [
        {
            "type": "text",
            "part": {
                "id": "text_1",
                "type": "text",
                "text": "<think>Planning with tools.</think>\nI will inspect the data.",
            },
        },
        {
            "type": "tool_use",
            "part": {
                "id": "tool_1",
                "type": "tool",
                "tool": "wo_list_workorders",
                "input": {"asset_num": "C"},
                "output": {"total": 10},
            },
        },
        {
            "type": "text",
            "part": {
                "id": "text_2",
                "type": "text",
                "text": "<think>Now prepare the final answer.</think>\nExcavator C",
            },
        },
    ]

    answer, trajectory = _build_trajectory_from_events(events, [])

    assert answer == "Excavator C"
    assert trajectory.turns[0].text == "Excavator C"
    assert len(trajectory.turns[0].tool_calls) == 1


def test_build_trajectory_from_opencode_step_finish_usage():
    events = [
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {
                    "input": 13084,
                    "output": 33,
                    "reasoning": 61,
                    "cache": {"read": 128, "write": 2},
                },
            },
        },
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {
                    "input": 209,
                    "output": 84,
                    "reasoning": 17,
                    "cache": {"read": 13184, "write": 0},
                },
            },
        },
    ]

    answer, trajectory = _build_trajectory_from_events(events, [])

    assert answer == ""
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].input_tokens == 26607
    assert trajectory.turns[0].output_tokens == 195


def test_runner_defaults():
    runner = OpenCodeAgentRunner(server_paths={}, model="opencode/gpt-5")
    assert runner._model_id == "opencode/gpt-5"
    assert runner._opencode_model == "opencode/gpt-5"
    assert runner._agent_name == "assetops"
    assert runner._run_dir == _REPO_ROOT
    assert runner._thinking is False
    assert runner._variant is None


def test_runner_accepts_thinking_and_variant():
    runner = OpenCodeAgentRunner(
        server_paths={},
        model="opencode/gpt-5",
        thinking=True,
        variant="high",
    )
    assert runner._thinking is True
    assert runner._variant == "high"


def test_runner_workspace_mode(tmp_path):
    workspace = tmp_path / "run-401"
    runner = OpenCodeAgentRunner(
        server_paths={},
        model="opencode/gpt-5",
        allow_files=True,
        workspace_dir=workspace,
    )
    assert runner._run_dir == workspace.resolve()
    assert runner._config["agent"]["assetops"]["permission"]["read"] == "allow"
