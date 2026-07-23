"""Unit tests for OpenCodeAgentRunner helpers."""

from __future__ import annotations

import json
from pathlib import Path

from agent.models import ToolCall
from agent.opencode_agent.runner import (
    OpenCodeAgentRunner,
    _REPO_ROOT,
    _build_mcp_config,
    _build_opencode_config,
    _build_permissions,
    _build_trajectory_from_events,
    _command_for_log,
    _event_type_counts,
    _json_events,
    _needs_reasoning_effort_none,
    _opencode_error_message,
    _permission_log_summary,
    _resolve_run_dir,
    _resolve_opencode_model_and_provider,
    _stream_stats,
    _stream_tail,
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
    assert permission["*"] == "deny"
    assert permission["iot_*"] == "allow"
    assert permission["wo_*"] == "allow"
    assert permission["read"] == "deny"
    assert permission["glob"] == "deny"
    assert permission["grep"] == "deny"
    assert permission["lsp"] == "deny"
    assert permission["list"] == "deny"
    assert permission["bash"] == "deny"
    assert permission["edit"] == "deny"
    assert permission["todowrite"] == "deny"
    assert permission["webfetch"] == "deny"
    assert permission["websearch"] == "deny"
    assert permission["codesearch"] == "deny"
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
    assert permission["list"] == "allow"
    assert permission["bash"] == "allow"
    assert permission["edit"] == "allow"
    assert permission["todowrite"] == "deny"
    assert permission["webfetch"] == "allow"
    assert permission["websearch"] == "allow"
    assert permission["codesearch"] == "deny"
    assert permission["external_directory"] == "deny"


def test_build_permissions_allows_writes_with_bash_workspace_mode():
    permission = _build_permissions(["iot"], allow_bash=True)

    assert permission["bash"] == "allow"
    assert permission["edit"] == "allow"


def test_build_permissions_allows_edits_without_bash():
    permission = _build_permissions(["iot"], allow_edit=True)

    assert permission["edit"] == "allow"
    assert permission["bash"] == "deny"


def test_permission_log_summary_excludes_mcp_tool_rules():
    permission = _build_permissions(["iot"], allow_bash=True, allow_files=True)
    summary = _permission_log_summary(permission)

    assert summary == {
        "*": "deny",
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "lsp": "allow",
        "list": "allow",
        "edit": "allow",
        "bash": "allow",
        "todowrite": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "codesearch": "deny",
        "external_directory": "deny",
        "question": "deny",
    }
    assert "iot_*" not in summary


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
    assert (
        provider["litellm-proxy"]["models"]["azure/gpt-5.4"]["name"] == "azure/gpt-5.4"
    )
    assert (
        env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "sk-test"
    )  # pragma: allowlist secret


def test_resolve_tokenrouter_model(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-test")
    model, provider, env = _resolve_opencode_model_and_provider(
        "tokenrouter/MiniMax-M3"
    )
    assert model == "tokenrouter/MiniMax-M3"
    assert provider["tokenrouter"]["npm"] == "@ai-sdk/openai-compatible"
    assert provider["tokenrouter"]["options"]["baseURL"] == "https://router.example/v1"
    assert provider["tokenrouter"]["models"]["MiniMax-M3"]["name"] == "MiniMax-M3"
    assert (
        env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "tr-test"
    )  # pragma: allowlist secret


def test_resolve_tokenrouter_anthropic_model(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-test")
    model, provider, env = _resolve_opencode_model_and_provider(
        "tokenrouter/anthropic/claude-opus-4.8"
    )
    assert model == "tokenrouter/anthropic/claude-opus-4.8"
    assert provider["tokenrouter"]["npm"] == "@ai-sdk/anthropic"
    assert provider["tokenrouter"]["options"]["baseURL"] == "https://router.example/v1"
    assert (
        provider["tokenrouter"]["models"]["anthropic/claude-opus-4.8"]
        == {
            "name": "anthropic/claude-opus-4.8",
            "options": {"toolStreaming": False},
        }
    )
    assert (
        env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "tr-test"
    )  # pragma: allowlist secret


def test_tokenrouter_gpt5_models_need_reasoning_effort_none():
    assert _needs_reasoning_effort_none("tokenrouter", "openai/gpt-5.5")
    assert _needs_reasoning_effort_none("tokenrouter", "openai/gpt-5.6-sol")
    assert not _needs_reasoning_effort_none("tokenrouter", "MiniMax-M3")
    assert not _needs_reasoning_effort_none("litellm-proxy", "openai/gpt-5.6-sol")


def test_resolve_tokenrouter_gpt5_disables_reasoning_effort(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tr-test")
    for model_id in ("tokenrouter/openai/gpt-5.5", "tokenrouter/openai/gpt-5.6-sol"):
        model, provider, env = _resolve_opencode_model_and_provider(model_id)
        model_name = model_id.removeprefix("tokenrouter/")
        assert model == model_id
        model_config = provider["tokenrouter"]["models"][model_name]
        assert model_config["options"] == {"reasoningEffort": "none"}
        assert (
            env["ASSETOPSBENCH_OPENCODE_API_KEY"] == "tr-test"
        )  # pragma: allowlist secret


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
    assert config["agent"]["assetops"]["temperature"] == 0.1
    assert config["agent"]["assetops"]["permission"]["iot_*"] == "allow"
    assert config["agent"]["assetops"]["permission"]["read"] == "deny"
    assert config["mcp"]["iot"]["command"] == ["uv", "run", "iot-mcp-server"]


def test_build_opencode_config_uses_requested_temperature():
    config, _, _ = _build_opencode_config(
        model="opencode/gpt-5",
        agent_name="assetops",
        max_steps=7,
        temperature=0.0,
        server_paths={"iot": "iot-mcp-server"},
    )

    assert config["agent"]["assetops"]["temperature"] == 0.0


def test_json_events_parses_ndjson_and_plain_lines():
    events, plain = _json_events('{"type":"a"}\nnot-json\n{"type":"b"}\n')
    assert [event["type"] for event in events] == ["a", "b"]
    assert plain == ["not-json"]


def test_command_for_log_omits_question():
    question = "question with sensitive context"
    cmd = [
        "opencode",
        "run",
        "--format",
        "json",
        "--model",
        "opencode/gpt-5",
        question,
    ]

    logged = _command_for_log(cmd)

    assert logged == [
        "opencode",
        "run",
        "--format",
        "json",
        "--model",
        "opencode/gpt-5",
        "<question omitted>",
    ]
    assert question not in repr(logged)
    assert cmd[-1] == question


def test_stream_tail_truncates_long_streams():
    assert _stream_tail("short", limit=10) == "short"
    assert _stream_tail("0123456789", limit=4) == "<truncated 6 chars>\n6789"


def test_stream_stats_counts_decoded_chars_and_lines():
    assert _stream_stats("one\ntwo\n") == (8, 2)
    assert _stream_stats("") == (0, 0)


def test_event_type_counts_summarizes_events_without_payloads():
    events = [
        {"type": "message.part.updated", "payload": {"text": "secret"}},
        {"type": "message.part.updated"},
        {"payload": {"type": "not-counted"}},
    ]

    assert _event_type_counts(events) == {
        "message.part.updated": 2,
        "<missing>": 1,
    }


def test_opencode_error_message_extracts_api_error():
    message = _opencode_error_message(
        [
            {
                "type": "error",
                "error": {
                    "name": "APIError",
                    "data": {
                        "statusCode": 400,
                        "message": "Function tools are not supported.",
                    },
                },
            }
        ]
    )

    assert message == "OpenCode error APIError (400): Function tools are not supported."


def test_opencode_error_message_ignores_null_error_field():
    assert _opencode_error_message([{"type": "step_finish", "error": None}]) is None


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
    assert len(trajectory.turns) == 2
    assert trajectory.turns[0].input_tokens == 13214
    assert trajectory.turns[0].output_tokens == 94
    assert trajectory.turns[1].input_tokens == 13393
    assert trajectory.turns[1].output_tokens == 101
    assert trajectory.total_input_tokens == 26607
    assert trajectory.total_output_tokens == 195


def test_runner_defaults():
    runner = OpenCodeAgentRunner(server_paths={}, model="opencode/gpt-5")
    assert runner._model_id == "opencode/gpt-5"
    assert runner._opencode_model == "opencode/gpt-5"
    assert runner._agent_name == "assetops"
    assert runner._run_dir == _REPO_ROOT


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


async def test_runner_run_parses_fake_opencode_subprocess(tmp_path, monkeypatch):
    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    workspace = tmp_path / "workspace"
    fake_opencode = tmp_path / "fake-opencode"
    fake_opencode.write_text(
        """#!/bin/sh
printf '%s\n' "$*" > "$FAKE_CAPTURE_DIR/argv.txt"
printf '%s\n' "$PWD" > "$FAKE_CAPTURE_DIR/cwd.txt"
printf '%s\n' "$OPENCODE_CONFIG_CONTENT" > "$FAKE_CAPTURE_DIR/config.json"
printf '%s\n' "$AGENT_TRAJECTORY_DIR" > "$FAKE_CAPTURE_DIR/trajectory-env.txt"
printf '%s\n' "$SCENARIOS_DATA_DIR" > "$FAKE_CAPTURE_DIR/scenarios-env.txt"
printf '%s\n' '{"type":"message.part.updated","properties":{"part":{"id":"t1","type":"text","text":"done"}}}'
printf '%s\n' '{"type":"step_finish","part":{"type":"step-finish","tokens":{"input":3,"output":2}}}'
printf '%s\n' 'diagnostic stderr' >&2
""",
        encoding="utf-8",
    )
    fake_opencode.chmod(0o755)
    monkeypatch.setenv("FAKE_CAPTURE_DIR", str(capture_dir))
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path / "parent-trajectories"))
    monkeypatch.setenv("SCENARIOS_DATA_DIR", str(tmp_path / "parent-scenarios"))

    runner = OpenCodeAgentRunner(
        server_paths={"iot": "iot-mcp-server"},
        model="opencode/gpt-5",
        opencode_bin=str(fake_opencode),
        allow_files=True,
        workspace_dir=workspace,
    )

    result = await runner.run("question with private details")
    argv = (capture_dir / "argv.txt").read_text(encoding="utf-8")
    config = (capture_dir / "config.json").read_text(encoding="utf-8")

    assert result.answer == "done"
    assert result.trajectory.started_at is not None
    assert result.trajectory.stderr == "diagnostic stderr\n"
    assert result.trajectory.total_input_tokens == 3
    assert result.trajectory.total_output_tokens == 2
    assert "question with private details" in argv
    assert (capture_dir / "cwd.txt").read_text(encoding="utf-8").strip() == str(
        workspace.resolve()
    )
    assert (capture_dir / "trajectory-env.txt").read_text(encoding="utf-8") == "\n"
    assert (capture_dir / "scenarios-env.txt").read_text(encoding="utf-8") == "\n"
    assert json.loads(config)["agent"]["assetops"]["permission"]["*"] == "deny"


def _text_part(part_id, text, *, message_id=None, part_type="text"):
    part = {"id": part_id, "type": part_type, "text": text}
    if message_id is not None:
        part["messageID"] = message_id
    return {"type": "message.part.updated", "properties": {"part": part}}


def _tool_part(part_id, tool, *, input=None, output=None):
    return {
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": part_id,
                "type": "tool",
                "tool": tool,
                "input": input or {},
                "output": output,
            }
        },
    }


def _step_finish(input_tokens, output_tokens):
    return {
        "type": "step_finish",
        "part": {
            "type": "step-finish",
            "tokens": {"input": input_tokens, "output": output_tokens},
        },
    }


def test_answer_excludes_intermediate_narration():
    """Only the final assistant message is the answer, not scratch talk."""
    events = [
        _text_part("t1", "Let me check the sensors. ", message_id="m1"),
        _tool_part("tool1", "iot_sensors", input={"a": "CH6"}, output=["s1", "s2"]),
        _step_finish(50, 5),
        _text_part("t3", "Chiller 6 has 2 sensors.", message_id="m2"),
        _step_finish(60, 8),
    ]
    answer, trajectory = _build_trajectory_from_events(events, [])
    assert answer == "Chiller 6 has 2 sensors."
    assert len(trajectory.turns) == 2  # two reasoning steps
    assert len(trajectory.all_tool_calls) == 1
    assert trajectory.total_input_tokens == 110
    assert trajectory.total_output_tokens == 13


def test_answer_excludes_reasoning_parts():
    events = [
        _text_part("r1", "(internal) suspect bearing wear", part_type="reasoning"),
        _text_part("t1", "The pump is healthy."),
    ]
    answer, _ = _build_trajectory_from_events(events, [])
    assert answer == "The pump is healthy."


def test_parallel_tool_calls_keep_their_own_outputs():
    """Two tool calls emitted before their outputs must not be mismatched."""
    events = [
        _tool_part("call_a", "get_temp", input={"a": "CH6"}, output="temp=42"),
        _tool_part("call_b", "get_vibration", input={"a": "CH6"}, output="vib=0.3"),
        _text_part("t1", "done"),
    ]
    _, trajectory = _build_trajectory_from_events(events, [])
    by_name = {tc.name: tc.output for tc in trajectory.all_tool_calls}
    assert by_name["get_temp"] == "temp=42"
    assert by_name["get_vibration"] == "vib=0.3"


def test_text_deltas_are_reconstructed():
    snapshot = [
        _text_part("t1", "Chiller 6 "),
        _text_part("t1", "Chiller 6 has 5 sensors."),
    ]
    delta = [
        _text_part("t1", "Chiller 6 "),
        _text_part("t1", "has 5 sensors."),
    ]
    assert _build_trajectory_from_events(snapshot, [])[0] == "Chiller 6 has 5 sensors."
    assert _build_trajectory_from_events(delta, [])[0] == "Chiller 6 has 5 sensors."
