"""Codex CLI protocol and subprocess boundaries without live calls."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from llm.codex import CodexBackend


def completed_event(**overrides):
    return {
        "type": "turn.completed",
        "usage": {"input_tokens": 35, "cached_input_tokens": 10, "output_tokens": 7},
        **overrides,
    }


def install_response(monkeypatch, events, *, returncode=0, response='{"ready": true}'):
    def run(command, **kwargs):
        if response is not None:
            Path(command[command.index("--output-last-message") + 1]).write_text(response)
        return SimpleNamespace(
            stdout="\n".join(json.dumps(event) for event in events),
            stderr="", returncode=returncode,
        )
    monkeypatch.setattr(subprocess, "run", run)


def test_prompt_is_piped_in_isolated_workspace_and_budget_is_advisory(monkeypatch):
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        root = Path(kwargs["cwd"])
        captured["instructions"] = (root / "instructions.txt").read_text()
        Path(command[command.index("--output-last-message") + 1]).write_text('{"ready": true}')
        return SimpleNamespace(
            stdout=json.dumps(completed_event()), stderr="", returncode=0,
        )

    monkeypatch.setattr(subprocess, "run", run)
    backend = CodexBackend("example-model")
    result = backend.generate_with_usage("private prompt", max_tokens=4096)

    assert json.loads(result.text) == {"ready": True}
    # Cached input is already included in Codex's input_tokens count.
    assert (result.input_tokens, result.output_tokens) == (35, 7)
    command = captured["command"]
    assert command[-1] == "-"
    assert command[command.index("--model") + 1] == "example-model"
    assert "private prompt" not in command
    assert captured["input"] == "private prompt"
    assert "--ignore-user-config" in command
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "approval_policy=\"never\"" in command
    assert "web_search=\"disabled\"" in command
    option_pairs = list(zip(command, command[1:]))
    assert ("--enable", "skip_host_skill_discovery") in option_pairs
    assert ("--disable", "shell_tool") in option_pairs
    assert ("--disable", "unified_exec") in option_pairs
    assert "Aim to keep the final response within 4096 tokens" in captured["instructions"]
    assert not Path(captured["cwd"]).exists()


def test_generate_preserves_text_interface_without_usage(monkeypatch):
    install_response(monkeypatch, [completed_event(usage=None)])
    backend = CodexBackend()
    assert backend.generate("prompt") == '{"ready": true}'
    assert backend.generate_with_usage("prompt").input_tokens == 0


def test_no_model_override_uses_cli_default(monkeypatch):
    def run(command, **kwargs):
        assert "--model" not in command
        Path(command[command.index("--output-last-message") + 1]).write_text("done")
        return SimpleNamespace(stdout=json.dumps(completed_event()), stderr="", returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    assert CodexBackend().generate("prompt") == "done"


@pytest.mark.parametrize("events,returncode,response,message", [
    ([{"type": "error", "message": "Please log in"}], 1, None, "Please log in"),
    ([{"type": "turn.failed", "error": {"message": "turn limit"}}], 0, "partial", "turn limit"),
    ([completed_event()], 1, "partial", "exit 1"),
    ([{"type": "thread.started"}], 0, "partial", "no completed turn"),
    ([completed_event()], 0, None, "without a final response"),
    ([completed_event()], 0, " \n", "empty text"),
    ([[]], 0, "partial", "invalid event envelope"),
])
def test_failures_never_become_generation_text(monkeypatch, events, returncode, response, message):
    install_response(monkeypatch, events, returncode=returncode, response=response)
    with pytest.raises(RuntimeError, match=message):
        CodexBackend().generate("prompt")


def test_invalid_cli_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="not json", stderr="unknown option", returncode=1,
    ))
    with pytest.raises(RuntimeError, match="invalid JSON events"):
        CodexBackend().generate("prompt")


@pytest.mark.parametrize("failure,message", [
    (FileNotFoundError(), "CLI not found"),
    (subprocess.TimeoutExpired("codex", 30), "timed out"),
])
def test_process_failures(monkeypatch, failure, message):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match=message):
        CodexBackend().generate("prompt")


@pytest.mark.parametrize("kwargs,message", [
    ({"temperature": 0.7}, "does not expose a temperature setting"),
    ({"max_tokens": 0}, "max_tokens must be positive"),
])
def test_unsupported_parameters_fail_before_launch(monkeypatch, kwargs, message):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Invalid generation parameters must not launch Codex")
    monkeypatch.setattr(subprocess, "run", unexpected_launch)
    with pytest.raises(ValueError, match=message):
        CodexBackend().generate("prompt", **kwargs)
