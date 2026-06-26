from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from benchmark import scenario_suite_runner as mr


def test_load_scenario_ids_ignores_blank_lines_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "scenarios.txt"
    p.write_text(
        """
        # scenario_suite scenarios

        11
        12

        # more
        14
        15
        """,
        encoding="utf-8",
    )

    assert mr.load_scenario_ids(p) == ["11", "12", "14", "15"]


def test_load_scenario_ids_raises_for_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        mr.load_scenario_ids(p)


def test_scenario_dir_for_id() -> None:
    root = Path("/tmp/scenarios_data")
    assert mr.scenario_dir_for_id(root, "11") == root / "scenario_11"


def test_read_question_reads_question_txt(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "scenario_11"
    scenario_dir.mkdir()
    (scenario_dir / "question.txt").write_text("What is the count?", encoding="utf-8")

    assert mr.read_question(tmp_path, "11") == "What is the count?"


def test_read_question_raises_when_missing(tmp_path: Path) -> None:
    (tmp_path / "scenario_11").mkdir()

    with pytest.raises(FileNotFoundError):
        mr.read_question(tmp_path, "11")


def test_build_methods_uses_cli_defaults() -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=False,
        opencode_allow_bash=False,
        opencode_allow_edit=False,
        opencode_workspace_root=None,
    )

    methods = mr.build_methods(args)

    assert methods["direct_llm"].command == "direct-llm-agent"
    assert methods["direct_llm"].model_id == "tokenrouter/MiniMax-M3"
    assert methods["stirrup_agent"].command == "stirrup-agent"
    assert methods["stirrup_agent"].model_id == "tokenrouter/MiniMax-M3"
    assert methods["opencode_agent"].command == "opencode-agent"
    assert methods["opencode_agent"].extra_args == ()
    assert methods["opencode_agent"].workspace_root is None


def test_build_methods_opencode_workspace_options(tmp_path: Path) -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=True,
        opencode_allow_bash=True,
        opencode_allow_edit=False,
        opencode_workspace_root=tmp_path / "workspaces",
    )

    methods = mr.build_methods(args)
    opencode = methods["opencode_agent"]

    assert opencode.extra_args == ("--allow-files", "--allow-bash")
    assert opencode.workspace_root == tmp_path / "workspaces"


def test_selected_methods_direct_llm_only() -> None:
    methods = {
        "direct_llm": mr.MethodConfig(
            agent_name="direct_llm",
            command="direct-llm-agent",
            model_id="tokenrouter/MiniMax-M3",
        ),
        "stirrup_agent": mr.MethodConfig(
            agent_name="stirrup_agent",
            command="stirrup-agent",
            model_id="tokenrouter/MiniMax-M3",
        ),
    }

    selected = mr.selected_methods(method_name="direct_llm", methods=methods)

    assert len(selected) == 1
    assert selected[0].agent_name == "direct_llm"


def test_selected_methods_all_returns_both() -> None:
    methods = {
        "direct_llm": mr.MethodConfig(
            agent_name="direct_llm",
            command="direct-llm-agent",
            model_id="tokenrouter/MiniMax-M3",
        ),
        "stirrup_agent": mr.MethodConfig(
            agent_name="stirrup_agent",
            command="stirrup-agent",
            model_id="tokenrouter/MiniMax-M3",
        ),
    }

    selected = mr.selected_methods(method_name="all", methods=methods)

    assert [m.agent_name for m in selected] == ["direct_llm", "stirrup_agent"]


def test_run_agent_for_scenario_dry_run_does_not_call_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess.run should not be called in dry_run")

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    method = mr.MethodConfig(
        agent_name="direct_llm",
        command="direct-llm-agent",
        model_id="tokenrouter/MiniMax-M3",
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="11",
        question="What is the count?",
        trajectory_dir=tmp_path / "traj",
        dry_run=True,
    )

    assert called is False


def test_run_agent_for_scenario_adds_opencode_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    method = mr.MethodConfig(
        agent_name="opencode_agent",
        command="opencode-agent",
        model_id="tokenrouter/MiniMax-M3",
        extra_args=("--allow-files", "--allow-bash"),
        workspace_root=tmp_path / "workspaces",
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="401",
        question="Which excavator costs the most?",
        trajectory_dir=tmp_path / "traj",
        dry_run=False,
    )

    expected_workspace = tmp_path / "workspaces" / "opencode_agent_401"
    assert expected_workspace.exists()
    assert captured["cmd"] == [
        "uv",
        "run",
        "opencode-agent",
        "--model-id",
        "tokenrouter/MiniMax-M3",
        "--allow-files",
        "--allow-bash",
        "--workspace-dir",
        str(expected_workspace),
        "--scenario-id",
        "401",
        "--run-id",
        "opencode_agent_401",
        "Which excavator costs the most?",
    ]


def test_run_evaluation_dry_run_does_not_call_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess.run should not be called in dry_run")

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    mr.run_evaluation(
        trajectory_dir=tmp_path / "traj",
        scenario_root=tmp_path / "scenarios",
        report_dir=tmp_path / "reports",
        dry_run=True,
    )

    assert called is False
