"""Unit tests for :mod:`observability.persistence`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from observability import persist_trajectory, set_run_context
from observability import runspan as _runspan


@dataclass
class _FakeToolCall:
    name: str
    input: dict
    output: str = ""


@dataclass
class _FakeTurn:
    index: int
    text: str
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeTrajectory:
    turns: list = field(default_factory=list)


@pytest.fixture(autouse=True)
def _reset_ctx():
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)
    yield
    _runspan._run_id_var.set(None)
    _runspan._scenario_id_var.set(None)


def test_persist_disabled_without_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("AGENT_TRAJECTORY_DIR", raising=False)
    set_run_context(run_id="r1")
    result = persist_trajectory(
        runner_name="deep-agent",
        model="anthropic/claude",
        question="q",
        answer="a",
        trajectory=_FakeTrajectory(),
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_persist_writes_file(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    set_run_context(run_id="run-42", scenario_id="scn-9")
    traj = _FakeTrajectory(
        turns=[
            _FakeTurn(
                index=0,
                text="hello",
                tool_calls=[_FakeToolCall(name="sensors", input={"id": "CH-6"}, output="ok")],
                input_tokens=100,
                output_tokens=20,
            ),
        ],
    )

    out = persist_trajectory(
        runner_name="claude-agent",
        model="litellm_proxy/aws/claude-opus-4-6",
        question="what sensors?",
        answer="CH-6 has sensors",
        trajectory=traj,
    )

    assert out == tmp_path / "run-42.json"
    record = json.loads(out.read_text())
    assert record["run_id"] == "run-42"
    assert record["scenario_id"] == "scn-9"
    assert record["runner"] == "claude-agent"
    assert record["model"] == "litellm_proxy/aws/claude-opus-4-6"
    assert record["question"] == "what sensors?"
    assert record["answer"] == "CH-6 has sensors"
    assert record["trajectory"]["turns"][0]["text"] == "hello"
    assert record["trajectory"]["turns"][0]["tool_calls"][0]["name"] == "sensors"


def test_persist_serializes_list_trajectory(monkeypatch, tmp_path: Path):
    """plan-execute trajectories are list[StepResult] (dataclasses)."""
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    set_run_context(run_id="r2")

    @dataclass
    class _FakeStep:
        step_number: int
        task: str
        success: bool

    out = persist_trajectory(
        runner_name="plan-execute",
        model="watsonx/model",
        question="q",
        answer="a",
        trajectory=[_FakeStep(1, "do thing", True)],
    )

    record = json.loads(out.read_text())
    assert record["trajectory"] == [{"step_number": 1, "task": "do thing", "success": True}]


def test_persist_skips_when_no_run_id(monkeypatch, tmp_path: Path, caplog):
    """Env set but run_id missing → warn + skip (don't lose data silently)."""
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    # No set_run_context call → run_id still None.
    with caplog.at_level("WARNING"):
        result = persist_trajectory(
            runner_name="deep-agent",
            model="m",
            question="q",
            answer="a",
            trajectory=_FakeTrajectory(),
        )
    assert result is None
    assert any("no run_id" in r.message for r in caplog.records)


def test_persist_creates_nested_dir(monkeypatch, tmp_path: Path):
    target = tmp_path / "nested" / "trajectories"
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(target))
    set_run_context(run_id="r3")
    out = persist_trajectory(
        runner_name="deep-agent",
        model="m",
        question="q",
        answer="a",
        trajectory=_FakeTrajectory(),
    )
    assert out == target / "r3.json"
    assert target.is_dir()
