"""Claude Code response handling and subprocess boundaries without live calls."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from llm.claude_code import ClaudeCodeBackend


def result_payload(**overrides):
    return {
        "type": "result", "subtype": "success", "is_error": False,
        "result": '{"ready": true}',
        "usage": {"input_tokens": 5, "cache_creation_input_tokens": 20,
                  "cache_read_input_tokens": 10, "output_tokens": 7},
        **overrides,
    }


def install_response(monkeypatch, payload, returncode=0):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps(payload), stderr="", returncode=returncode,
    ))


def test_prompt_is_piped_with_isolated_context_and_output_budget(monkeypatch):
    captured = {}
    monkeypatch.delenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", raising=False)

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(stdout=json.dumps(result_payload()), stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    backend = ClaudeCodeBackend("opus")
    result = backend.generate_with_usage("private prompt", max_tokens=4096)

    assert json.loads(result.text) == {"ready": True}
    assert (result.input_tokens, result.output_tokens) == (35, 7)
    assert "private prompt" not in captured["command"]
    assert captured["input"] == "private prompt"
    assert "--safe-mode" in captured["command"]
    assert "--no-session-persistence" in captured["command"]
    assert captured["command"][captured["command"].index("--tools") + 1] == ""
    assert "--dangerously-skip-permissions" not in captured["command"]
    assert captured["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in os.environ
    assert not Path(captured["cwd"]).exists()


def test_generate_preserves_text_interface_without_usage(monkeypatch):
    install_response(monkeypatch, result_payload(usage=None))
    backend = ClaudeCodeBackend()
    assert backend.generate("prompt") == '{"ready": true}'
    assert backend.generate_with_usage("prompt").input_tokens == 0


@pytest.mark.parametrize("payload,returncode,message", [
    (result_payload(is_error=True, result="Please log in"), 0, "Please log in"),
    (result_payload(subtype="error_max_turns", errors=["turn limit"]), 0, "turn limit"),
    (result_payload(result="process failed"), 1, "process failed"),
    (result_payload(result=""), 0, "empty text"),
    (result_payload(result=None), 0, "empty text"),
    ([], 0, "result envelope"),
    ({"type": "assistant", "result": "partial"}, 0, "result envelope"),
])
def test_failures_never_become_generation_text(monkeypatch, payload, returncode, message):
    install_response(monkeypatch, payload, returncode)
    with pytest.raises(RuntimeError, match=message):
        ClaudeCodeBackend().generate("prompt")


def test_invalid_cli_output_has_an_actionable_error(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="not json", stderr="unknown option", returncode=1,
    ))
    with pytest.raises(RuntimeError, match="Check 'claude auth status'"):
        ClaudeCodeBackend().generate("prompt")


@pytest.mark.parametrize("failure,message", [
    (FileNotFoundError(), "CLI not found"),
    (subprocess.TimeoutExpired("claude", 30), "timed out"),
])
def test_process_failures(monkeypatch, failure, message):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match=message):
        ClaudeCodeBackend().generate("prompt")


@pytest.mark.parametrize("kwargs,message", [
    ({"temperature": 0.7}, "does not expose a temperature setting"),
    ({"max_tokens": 0}, "max_tokens must be positive"),
])
def test_unsupported_parameters_fail_before_launch(monkeypatch, kwargs, message):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Invalid generation parameters must not launch Claude Code")
    monkeypatch.setattr(subprocess, "run", unexpected_launch)
    with pytest.raises(ValueError, match=message):
        ClaudeCodeBackend().generate("prompt", **kwargs)
