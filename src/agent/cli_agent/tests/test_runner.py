"""Unit tests for the CLI coding-agent runners.

These tests never launch a real agent CLI: ``run`` is exercised against a fake
subprocess (a short ``python -c`` that emits canned NDJSON), and the config /
provider / parser helpers are tested directly. Mirrors the mocking style of the
SDK runners' ``tests/test_runner.py``.
"""

from __future__ import annotations

import tomllib

import pytest

from agent.cli_agent._providers import provider_name, resolve_model, resolve_provider
from agent.cli_agent.codex.runner import CodexCliRunner
from agent.models import AgentResult, Trajectory


# ---------------------------------------------------------------------------
# _providers
# ---------------------------------------------------------------------------


def test_provider_name_and_model_stripping():
    assert provider_name("litellm_proxy/azure/gpt-5.4") == "litellm"
    assert provider_name("openrouter/openai/gpt-5.4") == "openrouter"
    assert provider_name("tokenrouter/gpt-5") == "tokenrouter"
    assert provider_name("gpt-5") == "direct"
    assert resolve_model("tokenrouter/gpt-5") == "gpt-5"
    assert resolve_model("openrouter/openai/gpt-5.4") == "openai/gpt-5.4"


def test_resolve_provider_tokenrouter(monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://tok/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tk")
    p = resolve_provider("tokenrouter/gpt-5")
    assert p.name == "tokenrouter"
    assert p.model == "gpt-5"
    assert p.base_url == "https://tok/v1"
    assert p.api_key == "tk"
    assert p.api_key_env == "TOKENROUTER_API_KEY"


def test_resolve_provider_openrouter_default_base(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or")
    p = resolve_provider("openrouter/openai/gpt-5.4")
    assert p.base_url == "https://openrouter.ai/api/v1"


def test_resolve_provider_missing_env_raises(monkeypatch):
    monkeypatch.delenv("TOKENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("TOKENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TOKENROUTER"):
        resolve_provider("tokenrouter/gpt-5")


# ---------------------------------------------------------------------------
# Codex _write_config
# ---------------------------------------------------------------------------


def test_codex_write_config_proxy_block(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://tok/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "tk")
    runner = CodexCliRunner(
        model="tokenrouter/gpt-5",
        server_paths={"iot": "iot-mcp-server", "wo": "wo-mcp-server"},
    )
    provider = resolve_provider(runner._model_id)
    env = runner._write_config(tmp_path, provider)
    assert env["CODEX_HOME"] == str(tmp_path)

    cfg = tomllib.loads((tmp_path / "config.toml").read_text())
    # Model is passed on the CLI (ALE parity), not in config.toml.
    assert "model" not in cfg
    assert cfg["model_provider"] == "tokenrouter"
    assert cfg["model_providers"]["tokenrouter"]["base_url"] == "https://tok/v1"
    assert cfg["model_providers"]["tokenrouter"]["env_key"] == "TOKENROUTER_API_KEY"
    assert set(cfg["mcp_servers"]) == {"iot", "wo"}
    assert cfg["mcp_servers"]["iot"]["args"] == ["run", "iot-mcp-server"]
    # Model + stdin prompt are on the launch side.
    assert runner._build_command(tmp_path, "sys", "q")[:3] == ["codex", "exec", "--model"]
    assert "gpt-5" in runner._build_command(tmp_path, "sys", "q")
    assert "User question: q" in runner._stdin_text("sys", "q")


def test_codex_write_config_direct_omits_provider_block(tmp_path):
    runner = CodexCliRunner(model="gpt-5", server_paths={"iot": "iot-mcp-server"})
    provider = resolve_provider("gpt-5")  # direct
    runner._write_config(tmp_path, provider)
    cfg = tomllib.loads((tmp_path / "config.toml").read_text())
    assert "model_providers" not in cfg
    assert "model" not in cfg
    assert cfg["model_reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Codex _handle_event
# ---------------------------------------------------------------------------


def test_codex_handle_event_tool_message_usage():
    runner = CodexCliRunner(model="gpt-5", server_paths={})
    traj = Trajectory()
    runner._handle_event(
        {"type": "item.completed", "item": {
            "type": "mcp_tool_call", "server": "iot", "tool": "list_sensors",
            "arguments": {"asset": "CH-6"}, "result": "S1,S2"}},
        traj,
    )
    answer = runner._handle_event(
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": "Chiller 6 has S1, S2."}},
        traj,
    )
    runner._handle_event({"type": "turn.completed",
                          "usage": {"input_tokens": 100, "output_tokens": 20}}, traj)

    assert answer == "Chiller 6 has S1, S2."
    assert len(traj.all_tool_calls) == 1
    assert traj.all_tool_calls[0].name == "list_sensors"
    assert traj.all_tool_calls[0].input == {"asset": "CH-6"}
    assert (traj.total_input_tokens, traj.total_output_tokens) == (100, 20)


# ---------------------------------------------------------------------------
# Gemini _handle_event (stream-json NDJSON)
# ---------------------------------------------------------------------------


def test_gemini_handle_event_streamjson():
    from agent.cli_agent.gemini.runner import GeminiCliRunner

    runner = GeminiCliRunner(model="gemini-2.5-pro", server_paths={})
    traj = Trajectory()
    runner._handle_event(
        {"type": "tool_use", "tool_name": "list_sensors", "tool_id": "t1",
         "parameters": {"asset": "CH-6"}}, traj)
    runner._handle_event(
        {"type": "tool_result", "tool_id": "t1", "output": "S1,S2"}, traj)
    answer = runner._handle_event(
        {"type": "result", "response": "Chiller 6 has S1, S2.",
         "stats": {"input_tokens": 12, "output_tokens": 4}}, traj)

    assert answer == "Chiller 6 has S1, S2."
    assert traj.all_tool_calls[0].name == "list_sensors"
    assert traj.all_tool_calls[0].output == "S1,S2"
    assert (traj.total_input_tokens, traj.total_output_tokens) == (12, 4)


# ---------------------------------------------------------------------------
# run() against a fake subprocess
# ---------------------------------------------------------------------------


class _FakeCodex(CodexCliRunner):
    """Codex runner that fakes the subprocess + skips real config."""

    def _write_config(self, home, provider):
        return {}

    def _build_command(self, home, system_prompt, question):
        events = [
            {"type": "item.completed", "item": {
                "type": "mcp_tool_call", "server": "iot", "tool": "list_sensors",
                "arguments": {"asset": "CH-6"}, "result": "S1,S2"}},
            {"type": "item.completed", "item": {
                "type": "agent_message", "text": "Chiller 6 has sensors S1 and S2."}},
            {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}},
        ]
        script = "import json\n" + "".join(
            f"print(json.dumps({e!r}))\n" for e in events
        )
        return ["python3", "-c", script]


@pytest.mark.anyio
async def test_run_returns_agent_result_and_trajectory():
    runner = _FakeCodex(model="gpt-5", server_paths={"iot": "iot-mcp-server"})
    result = await runner.run("What sensors are on Chiller 6?")

    assert isinstance(result, AgentResult)
    assert result.answer == "Chiller 6 has sensors S1 and S2."
    traj = result.trajectory
    assert len(traj.all_tool_calls) == 1
    assert traj.all_tool_calls[0].name == "list_sensors"
    assert (traj.total_input_tokens, traj.total_output_tokens) == (100, 20)


@pytest.mark.anyio
async def test_run_unsupported_provider_raises(monkeypatch):
    # Claude Code excludes openrouter; resolve must raise before launch.
    from agent.cli_agent.claude_code.runner import ClaudeCodeRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "or")
    runner = ClaudeCodeRunner(model="openrouter/anthropic/claude", server_paths={})
    with pytest.raises(ValueError, match="does not support"):
        await runner.run("hi")
