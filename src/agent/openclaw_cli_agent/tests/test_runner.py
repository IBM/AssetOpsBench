"""Unit tests for OpenClawCliAgentRunner helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.models import ToolCall
from agent.openclaw_cli_agent.runner import (
    OpenClawCliAgentRunner,
    _answer_from_envelope,
    _build_mcp_config,
    _build_openclaw_config,
    _disabled_tools,
    _extract_json_object,
    _parse_transcript,
    _REPO_ROOT,
    _resolve_openclaw_model_and_auth,
    _resolve_run_dir,
    _stage_openclaw_home,
    _usage_from_envelope,
)


def test_build_mcp_config_entrypoint() -> None:
    config = _build_mcp_config({"iot": "iot-mcp-server"}, cwd=Path("/repo"))
    assert config["iot"] == {
        "command": "uv",
        "args": ["run", "iot-mcp-server"],
        "cwd": "/repo",
        "enabled": True,
        "timeout": 30,
    }


def test_build_mcp_config_path() -> None:
    config = _build_mcp_config({"custom": Path("/tmp/server.py")}, cwd=Path("/repo"))
    assert config["custom"]["args"] == ["run", "/tmp/server.py"]


def test_disabled_tools_default_safe() -> None:
    disabled = _disabled_tools()
    assert "web_search" in disabled
    assert "web_fetch" in disabled
    assert "read_file" in disabled
    assert "write_file" in disabled
    assert "exec" in disabled
    assert "bash" in disabled


def test_disabled_tools_respects_opt_in_flags() -> None:
    disabled = _disabled_tools(
        allow_files=True,
        allow_bash=True,
        allow_edit=True,
        allow_web=True,
    )
    assert "web_search" not in disabled
    assert "web_fetch" not in disabled
    assert "read_file" not in disabled
    assert "write_file" not in disabled
    assert "exec" not in disabled
    assert "bash" not in disabled


def test_disabled_tools_allows_read_only_files() -> None:
    disabled = _disabled_tools(allow_files=True, allow_edit=False)
    assert "read_file" not in disabled
    assert "write_file" in disabled
    assert "edit" in disabled


def test_resolve_run_dir_defaults_to_repo_root() -> None:
    run_dir = _resolve_run_dir()
    assert run_dir == _REPO_ROOT


def test_resolve_run_dir_requires_workspace_for_file_or_code_tools() -> None:
    with pytest.raises(ValueError, match="workspace-dir"):
        _resolve_run_dir(allow_files=True)
    with pytest.raises(ValueError, match="workspace-dir"):
        _resolve_run_dir(allow_bash=True)
    with pytest.raises(ValueError, match="workspace-dir"):
        _resolve_run_dir(allow_edit=True)


def test_resolve_run_dir_creates_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "openclaw-run"
    run_dir = _resolve_run_dir(workspace_dir=workspace, allow_files=True)
    assert run_dir == workspace.resolve()
    assert run_dir.exists()


def test_resolve_tokenrouter_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-test")

    model, provider, key, base_url = _resolve_openclaw_model_and_auth(
        "tokenrouter/MiniMax-M3"
    )

    assert model == "openai/MiniMax-M3"
    assert provider == "openai"
    assert key == "tr-test"
    assert base_url == "https://api.tokenrouter.com/v1"


def test_resolve_openai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    model, provider, key, base_url = _resolve_openclaw_model_and_auth("openai/gpt-5.1")

    assert model == "openai/gpt-5.1"
    assert provider == "openai"
    assert key == "sk-test"
    assert base_url is None


def test_build_openclaw_config_includes_mcp_and_tokenrouter_base_url() -> None:
    config = _build_openclaw_config(
        model_ref="openai/MiniMax-M3",
        provider="openai",
        base_url="https://api.tokenrouter.com/v1",
        server_paths={"iot": "iot-mcp-server"},
        timeout_s=900,
    )

    assert config["agents"]["defaults"]["model"]["primary"] == "openai/MiniMax-M3"
    assert config["plugins"]["allow"] == ["openai", "memory-core"]
    assert config["mcp"]["servers"]["iot"]["args"] == ["run", "iot-mcp-server"]
    assert config["models"]["providers"]["openai"]["baseUrl"] == (
        "https://api.tokenrouter.com/v1"
    )
    assert config["models"]["providers"]["openai"]["models"] == [
        {"id": "MiniMax-M3", "name": "MiniMax-M3"}
    ]
    assert "read_file" in config["tools"]["deny"]
    assert "exec" in config["tools"]["deny"]


def test_stage_openclaw_home_writes_config_and_auth(tmp_path: Path) -> None:
    config = _build_openclaw_config(
        model_ref="openai/MiniMax-M3",
        provider="openai",
        base_url="https://api.tokenrouter.com/v1",
        server_paths={},
        timeout_s=900,
    )
    home = _stage_openclaw_home(
        home_root=tmp_path,
        agent_name="assetops",
        config=config,
        provider="openai",
        api_key="tr-test",
    )

    assert (home / ".openclaw" / "openclaw.json").exists()
    auth_path = home / ".openclaw" / "agents" / "assetops" / "agent" / "auth-profiles.json"
    assert auth_path.exists()
    assert "tr-test" in auth_path.read_text(encoding="utf-8")
    assert (home / ".openclaw" / "workspace" / ".openclaw" / "workspace-state.json").exists()


def test_extract_json_object_parses_plain_json_and_preamble() -> None:
    assert _extract_json_object('{"ok": true}') == {"ok": True}
    assert _extract_json_object('warning\n{"answer": "done"}\n') == {"answer": "done"}


def test_answer_and_usage_from_envelope() -> None:
    envelope = {
        "meta": {
            "finalAssistantVisibleText": "The answer.",
            "agentMeta": {"usage": {"input": 100, "output": 25}},
        }
    }

    assert _answer_from_envelope(envelope) == "The answer."
    assert _usage_from_envelope(envelope) == (100, 25)


def test_parse_transcript_extracts_tool_calls(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                '{"message":{"content":[{"type":"tool_call","id":"tool_1","name":"iot_get_asset","input":{"asset_id":"A1"}}]}}',
                '{"message":{"content":[{"type":"tool_result","toolCallId":"tool_1","output":{"name":"Pump 1"}}]}}',
            ]
        ),
        encoding="utf-8",
    )

    events, tool_calls = _parse_transcript(transcript)

    assert len(events) == 2
    assert isinstance(tool_calls[0], ToolCall)
    assert tool_calls[0].name == "iot_get_asset"
    assert tool_calls[0].input == {"asset_id": "A1"}
    assert tool_calls[0].output == {"name": "Pump 1"}


def test_runner_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-test")

    runner = OpenClawCliAgentRunner(server_paths={})

    assert runner._model_id == "tokenrouter/MiniMax-M3"
    assert runner._openclaw_model == "openai/MiniMax-M3"
    assert runner._provider == "openai"
    assert runner._run_dir == _REPO_ROOT
    assert "read_file" in runner._config["tools"]["deny"]
    assert "exec" in runner._config["tools"]["deny"]


def test_runner_workspace_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://api.tokenrouter.com/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-test")
    workspace = tmp_path / "run-401"

    runner = OpenClawCliAgentRunner(
        server_paths={},
        allow_files=True,
        allow_bash=True,
        workspace_dir=workspace,
    )

    assert runner._run_dir == workspace.resolve()
    assert "read_file" not in runner._config["tools"]["deny"]
    assert "exec" not in runner._config["tools"]["deny"]
    assert (workspace / ".openclaw_home" / ".openclaw" / "openclaw.json").exists()
