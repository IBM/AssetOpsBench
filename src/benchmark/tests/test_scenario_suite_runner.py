from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

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


def test_scenario_mappings_cover_expected_categories() -> None:
    expected = {"car", "fcc", "fmsr", "health", "tsfm", "wosr"}

    assert set(mr.SCENARIO_IDS_ALL) == expected
    assert set(mr.SCENARIO_IDS_LITE) == expected
    assert all(len(ids) == 10 for ids in mr.SCENARIO_IDS_ALL.values())
    assert all(len(ids) == 1 for ids in mr.SCENARIO_IDS_LITE.values())


def test_scenario_profiles_are_loaded_from_yaml() -> None:
    assert mr.SCENARIO_IDS_ALL == mr.load_scenario_profile(
        mr.SCENARIO_PROFILE_PATHS["all"]
    )
    assert mr.SCENARIO_IDS_LITE == mr.load_scenario_profile(
        mr.SCENARIO_PROFILE_PATHS["lite"]
    )


def test_scenario_profile_yaml_uses_integer_ids() -> None:
    for path in mr.SCENARIO_PROFILE_PATHS.values():
        raw_profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert all(
            isinstance(scenario_id, int)
            for scenario_ids in raw_profile.values()
            for scenario_id in scenario_ids
        )


def test_scenario_ids_for_selector_resolves_combined_all_categories() -> None:
    assert mr.scenario_ids_for_selector("fcc+fmsr_all") == [
        *mr.SCENARIO_IDS_ALL["fcc"],
        *mr.SCENARIO_IDS_ALL["fmsr"],
    ]


def test_scenario_ids_for_selector_resolves_lite_category() -> None:
    assert mr.scenario_ids_for_selector("fcc_lite") == ["301"]


def test_scenario_ids_for_selector_resolves_profile_shorthands() -> None:
    assert mr.scenario_ids_for_selector("lite") == [
        "151",
        "301",
        "902",
        "401",
        "1001",
        "1",
    ]
    assert len(mr.scenario_ids_for_selector("all")) == 60


@pytest.mark.parametrize(
    "selector",
    ["fcc", "fcc_fast", "unknown_lite", "fcc++fmsr_all", "_lite"],
)
def test_scenario_ids_for_selector_rejects_invalid_selector(selector: str) -> None:
    with pytest.raises(ValueError, match="Invalid scenario selector"):
        mr.scenario_ids_for_selector(selector)


def test_resolve_scenario_ids_keeps_file_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "custom.txt"
    path.write_text("301\n# skip\n902\n", encoding="utf-8")

    assert mr.resolve_scenario_ids(path) == ["301", "902"]


def test_resolve_scenario_ids_accepts_yaml_profile(tmp_path: Path) -> None:
    path = tmp_path / "profile.yaml"
    path.write_text(
        """
car: [151]
fcc: [301]
fmsr: [902]
health: [401]
tsfm: [1001]
wosr: [1]
""".strip(),
        encoding="utf-8",
    )

    assert mr.resolve_scenario_ids(path) == ["151", "301", "902", "401", "1001", "1"]


def test_load_scenario_profile_rejects_missing_category(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("fcc: [301]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing categories"):
        mr.load_scenario_profile(path)


def test_resolve_scenario_ids_raises_for_missing_file_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Scenario id file not found"):
        mr.resolve_scenario_ids(tmp_path / "missing.txt")


def test_parser_accepts_named_scenario_selector() -> None:
    args = mr._build_parser().parse_args(
        [
            "--scenario-ids",
            "fcc+fmsr_all",
            "--scenario-root",
            "/tmp/scenarios_data",
        ]
    )

    assert args.scenario_ids == "fcc+fmsr_all"


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


def test_run_agent_for_scenario_starts_with_empty_opencode_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    workspace_root = tmp_path / "workspaces"
    expected_workspace = workspace_root / "opencode_agent_1001"
    expected_workspace.mkdir(parents=True)
    (expected_workspace / "stale.txt").write_text("old data", encoding="utf-8")

    method = mr.MethodConfig(
        agent_name="opencode_agent",
        command="opencode-agent",
        model_id="tokenrouter/MiniMax-M3",
        extra_args=("--allow-files",),
        workspace_root=workspace_root,
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="1001",
        question="Find anomaly.",
        trajectory_dir=tmp_path / "traj",
        dry_run=False,
    )

    assert expected_workspace.exists()
    assert list(expected_workspace.iterdir()) == []
    assert captured["cmd"] == [
        "uv",
        "run",
        "opencode-agent",
        "--model-id",
        "tokenrouter/MiniMax-M3",
        "--allow-files",
        "--workspace-dir",
        str(expected_workspace),
        "--scenario-id",
        "1001",
        "--run-id",
        "opencode_agent_1001",
        "Find anomaly.",
    ]


def test_validate_workspace_root_rejects_repo_paths() -> None:
    with pytest.raises(ValueError):
        mr.validate_workspace_root_outside_repo(
            mr.REPO_ROOT / "traces" / "opencode_workspaces",
            "--opencode-workspace-root",
        )


def test_model_dir_name_normalizes_router_model_ids() -> None:
    assert mr.model_dir_name("tokenrouter/MiniMax-M3") == "tokenrouter-MiniMax-M3"
    assert (
        mr.model_dir_name("tokenrouter/openai/gpt-5.4") == "tokenrouter-openai-gpt-5.4"
    )
    assert mr.model_dir_name(" rits/qwen3:30b ") == "rits-qwen3-30b"


def test_method_output_paths_nest_by_agent_and_model(tmp_path: Path) -> None:
    method = mr.MethodConfig(
        agent_name="opencode_agent",
        command="opencode-agent",
        model_id="tokenrouter/MiniMax-M3",
        workspace_root=tmp_path / "workspaces",
    )

    trajectory_dir, report_dir, workspace_root = mr.method_output_paths(
        trajectory_root=tmp_path / "trajectories",
        reports_root=tmp_path / "reports",
        method=method,
    )

    assert (
        trajectory_dir
        == tmp_path / "trajectories" / "opencode_agent" / "tokenrouter-MiniMax-M3"
    )
    assert report_dir == (
        tmp_path / "reports" / "opencode_agent" / "tokenrouter-MiniMax-M3"
    )
    assert workspace_root == (
        tmp_path / "workspaces" / "opencode_agent" / "tokenrouter-MiniMax-M3"
    )


def test_build_methods_uses_cli_defaults() -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        gemini_model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        openclaw_model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=False,
        opencode_allow_bash=False,
        opencode_allow_edit=False,
        opencode_workspace_root=None,
        gemini_allow_files=False,
        gemini_allow_bash=False,
        gemini_allow_edit=False,
        gemini_allow_web=False,
        gemini_sandbox=False,
        gemini_workspace_root=None,
        openclaw_allow_files=False,
        openclaw_allow_bash=False,
        openclaw_allow_edit=False,
        openclaw_allow_web=False,
        openclaw_thinking="off",
        openclaw_workspace_root=None,
        stirrup_max_tokens=4096,
        temperature=None,
    )

    methods = mr.build_methods(args)

    assert methods["direct_llm"].command == "direct-llm-agent"
    assert methods["direct_llm"].model_id == "tokenrouter/MiniMax-M3"
    assert methods["stirrup_agent"].command == "stirrup-agent"
    assert methods["stirrup_agent"].model_id == "tokenrouter/MiniMax-M3"
    assert methods["stirrup_agent"].extra_args == ("--max-tokens", "4096")
    assert methods["stirrup_agent"].workspace_root is None
    assert methods["opencode_agent"].command == "opencode-agent"
    assert methods["opencode_agent"].extra_args == ()
    assert methods["opencode_agent"].workspace_root is None
    assert methods["openai_agent"].command == "openai-agent"
    assert methods["openai_agent"].model_id == "tokenrouter/MiniMax-M3"
    assert methods["openai_agent"].extra_args == ()
    assert methods["openai_agent"].workspace_root is None
    assert methods["gemini_cli_agent"].command == "gemini-cli-agent"
    assert (
        methods["gemini_cli_agent"].model_id
        == "tokenrouter_gemini/google/gemma-4-26b-a4b-it"
    )
    assert methods["gemini_cli_agent"].extra_args == ()
    assert methods["gemini_cli_agent"].workspace_root is None
    assert methods["openclaw_cli_agent"].command == "openclaw-cli-agent"
    assert methods["openclaw_cli_agent"].model_id == "tokenrouter/MiniMax-M3"
    assert methods["openclaw_cli_agent"].extra_args == ("--thinking", "off")
    assert methods["openclaw_cli_agent"].workspace_root is None


def test_build_methods_stirrup_workspace_options(tmp_path: Path) -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        stirrup_workspace_root=tmp_path / "stirrup-workspaces",
        preserve_workspaces=True,
        gemini_model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        openclaw_model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=False,
        opencode_allow_bash=False,
        opencode_allow_edit=False,
        opencode_workspace_root=None,
        gemini_allow_files=False,
        gemini_allow_bash=False,
        gemini_allow_edit=False,
        gemini_allow_web=False,
        gemini_sandbox=False,
        gemini_workspace_root=None,
        openclaw_allow_files=False,
        openclaw_allow_bash=False,
        openclaw_allow_edit=False,
        openclaw_allow_web=False,
        openclaw_thinking="off",
        openclaw_workspace_root=None,
        stirrup_max_tokens=4096,
        temperature=0.2,
    )

    methods = mr.build_methods(args)
    stirrup = methods["stirrup_agent"]

    assert stirrup.extra_args == (
        "--max-tokens",
        "4096",
        "--temperature",
        "0.2",
        "--preserve-workspace",
    )
    assert stirrup.workspace_root == tmp_path / "stirrup-workspaces"


def test_build_methods_opencode_workspace_options(tmp_path: Path) -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        gemini_model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        openclaw_model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=True,
        opencode_allow_bash=True,
        opencode_allow_edit=False,
        opencode_workspace_root=tmp_path / "workspaces",
        gemini_allow_files=False,
        gemini_allow_bash=False,
        gemini_allow_edit=False,
        gemini_allow_web=False,
        gemini_sandbox=False,
        gemini_workspace_root=None,
        openclaw_allow_files=False,
        openclaw_allow_bash=False,
        openclaw_allow_edit=False,
        openclaw_allow_web=False,
        openclaw_thinking="off",
        openclaw_workspace_root=None,
        stirrup_max_tokens=4096,
        temperature=None,
    )

    methods = mr.build_methods(args)
    opencode = methods["opencode_agent"]

    assert opencode.extra_args == ("--allow-files", "--allow-bash")
    assert opencode.workspace_root == tmp_path / "workspaces"


def test_build_methods_opencode_thinking_and_variant() -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        gemini_model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        openclaw_model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=False,
        opencode_allow_bash=False,
        opencode_allow_edit=False,
        opencode_thinking=True,
        opencode_variant="high",
        opencode_temperature=0.0,
        opencode_workspace_root=None,
        gemini_allow_files=False,
        gemini_allow_bash=False,
        gemini_allow_edit=False,
        gemini_allow_web=False,
        gemini_sandbox=False,
        gemini_workspace_root=None,
        openclaw_allow_files=False,
        openclaw_allow_bash=False,
        openclaw_allow_edit=False,
        openclaw_allow_web=False,
        openclaw_thinking="off",
        openclaw_workspace_root=None,
        stirrup_max_tokens=4096,
        temperature=None,
    )

    methods = mr.build_methods(args)

    assert methods["opencode_agent"].extra_args == (
        "--thinking",
        "--variant",
        "high",
        "--temperature",
        "0.0",
    )


def test_build_methods_openai_workspace_options(tmp_path: Path) -> None:
    args = Namespace(
        model_id="tokenrouter/anthropic/claude-opus-4.8",
        gemini_model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        openclaw_model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=False,
        opencode_allow_bash=False,
        opencode_allow_edit=False,
        opencode_workspace_root=None,
        openai_allow_files=True,
        openai_allow_bash=True,
        openai_allow_edit=False,
        openai_allow_web=True,
        openai_reasoning_summary="detailed",
        openai_workspace_root=tmp_path / "openai-workspaces",
        gemini_allow_files=False,
        gemini_allow_bash=False,
        gemini_allow_edit=False,
        gemini_allow_web=False,
        gemini_sandbox=False,
        gemini_workspace_root=None,
        openclaw_allow_files=False,
        openclaw_allow_bash=False,
        openclaw_allow_edit=False,
        openclaw_allow_web=False,
        openclaw_thinking="off",
        openclaw_workspace_root=None,
        stirrup_max_tokens=4096,
        temperature=None,
    )

    methods = mr.build_methods(args)
    openai = methods["openai_agent"]

    assert openai.extra_args == (
        "--allow-files",
        "--allow-bash",
        "--allow-web",
        "--reasoning-summary",
        "detailed",
    )
    assert openai.workspace_root == tmp_path / "openai-workspaces"


def test_build_methods_gemini_workspace_options(tmp_path: Path) -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        gemini_model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        openclaw_model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=False,
        opencode_allow_bash=False,
        opencode_allow_edit=False,
        opencode_workspace_root=None,
        gemini_allow_files=True,
        gemini_allow_bash=True,
        gemini_allow_edit=False,
        gemini_allow_web=True,
        gemini_sandbox=True,
        gemini_workspace_root=tmp_path / "gemini-workspaces",
        openclaw_allow_files=False,
        openclaw_allow_bash=False,
        openclaw_allow_edit=False,
        openclaw_allow_web=False,
        openclaw_thinking="off",
        openclaw_workspace_root=None,
        stirrup_max_tokens=4096,
        temperature=None,
    )

    methods = mr.build_methods(args)
    gemini = methods["gemini_cli_agent"]

    assert gemini.extra_args == (
        "--allow-files",
        "--allow-bash",
        "--allow-web",
        "--sandbox",
    )
    assert gemini.workspace_root == tmp_path / "gemini-workspaces"


def test_build_methods_openclaw_workspace_options(tmp_path: Path) -> None:
    args = Namespace(
        model_id="tokenrouter/MiniMax-M3",
        gemini_model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        openclaw_model_id="tokenrouter/MiniMax-M3",
        opencode_allow_files=False,
        opencode_allow_bash=False,
        opencode_allow_edit=False,
        opencode_workspace_root=None,
        gemini_allow_files=False,
        gemini_allow_bash=False,
        gemini_allow_edit=False,
        gemini_allow_web=False,
        gemini_sandbox=False,
        gemini_workspace_root=None,
        openclaw_allow_files=True,
        openclaw_allow_bash=True,
        openclaw_allow_edit=False,
        openclaw_allow_web=True,
        openclaw_thinking="medium",
        openclaw_workspace_root=tmp_path / "openclaw-workspaces",
        stirrup_max_tokens=4096,
        temperature=None,
    )

    methods = mr.build_methods(args)
    openclaw = methods["openclaw_cli_agent"]

    assert openclaw.extra_args == (
        "--allow-files",
        "--allow-bash",
        "--allow-web",
        "--thinking",
        "medium",
    )
    assert openclaw.workspace_root == tmp_path / "openclaw-workspaces"


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


def test_run_agent_for_scenario_adds_openai_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    method = mr.MethodConfig(
        agent_name="openai_agent",
        command="openai-agent",
        model_id="tokenrouter/anthropic/claude-opus-4.8",
        extra_args=("--allow-files", "--allow-bash", "--allow-web"),
        workspace_root=tmp_path / "openai-workspaces",
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="401",
        question="Which excavator costs the most?",
        trajectory_dir=tmp_path / "traj",
        dry_run=False,
    )

    expected_workspace = tmp_path / "openai-workspaces" / "openai_agent_401"
    assert expected_workspace.exists()
    assert captured["cmd"] == [
        "uv",
        "run",
        "openai-agent",
        "--model-id",
        "tokenrouter/anthropic/claude-opus-4.8",
        "--allow-files",
        "--allow-bash",
        "--allow-web",
        "--workspace-dir",
        str(expected_workspace),
        "--scenario-id",
        "401",
        "--run-id",
        "openai_agent_401",
        "Which excavator costs the most?",
    ]


def test_run_agent_for_scenario_recreates_empty_opencode_workspace(
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
        extra_args=("--allow-files",),
        workspace_root=tmp_path / "workspaces",
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="1001",
        question="Find anomaly.",
        trajectory_dir=tmp_path / "traj",
        dry_run=False,
    )

    expected_workspace = tmp_path / "workspaces" / "opencode_agent_1001"
    assert expected_workspace.exists()
    assert list(expected_workspace.iterdir()) == []
    assert "--workspace-dir" in captured["cmd"]


def test_run_agent_for_scenario_can_keep_existing_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    workspace_root = tmp_path / "workspaces"
    expected_workspace = workspace_root / "stirrup_agent_1001"
    expected_workspace.mkdir(parents=True)
    stale_file = expected_workspace / "keep-me.txt"
    stale_file.write_text("debug artifact", encoding="utf-8")

    method = mr.MethodConfig(
        agent_name="stirrup_agent",
        command="stirrup-agent",
        model_id="tokenrouter/MiniMax-M3",
        extra_args=("--preserve-workspace",),
        workspace_root=workspace_root,
        reset_workspace=False,
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="1001",
        question="Find anomaly.",
        trajectory_dir=tmp_path / "traj",
        dry_run=False,
    )

    assert stale_file.exists()
    assert captured["cmd"] == [
        "uv",
        "run",
        "stirrup-agent",
        "--model-id",
        "tokenrouter/MiniMax-M3",
        "--preserve-workspace",
        "--workspace-dir",
        str(expected_workspace),
        "--scenario-id",
        "1001",
        "--run-id",
        "stirrup_agent_1001",
        "Find anomaly.",
    ]


def test_run_agent_for_scenario_adds_gemini_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    method = mr.MethodConfig(
        agent_name="gemini_cli_agent",
        command="gemini-cli-agent",
        model_id="tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        extra_args=("--allow-files", "--allow-bash"),
        workspace_root=tmp_path / "gemini-workspaces",
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="401",
        question="Which excavator costs the most?",
        trajectory_dir=tmp_path / "traj",
        dry_run=False,
    )

    expected_workspace = tmp_path / "gemini-workspaces" / "gemini_cli_agent_401"
    assert expected_workspace.exists()
    assert captured["cmd"] == [
        "uv",
        "run",
        "gemini-cli-agent",
        "--model-id",
        "tokenrouter_gemini/google/gemma-4-26b-a4b-it",
        "--allow-files",
        "--allow-bash",
        "--workspace-dir",
        str(expected_workspace),
        "--scenario-id",
        "401",
        "--run-id",
        "gemini_cli_agent_401",
        "Which excavator costs the most?",
    ]


def test_run_agent_for_scenario_adds_openclaw_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(mr.subprocess, "run", fake_run)

    method = mr.MethodConfig(
        agent_name="openclaw_cli_agent",
        command="openclaw-cli-agent",
        model_id="tokenrouter/MiniMax-M3",
        extra_args=("--allow-files", "--allow-bash", "--thinking", "high"),
        workspace_root=tmp_path / "openclaw-workspaces",
    )

    mr.run_agent_for_scenario(
        method=method,
        scenario_id="401",
        question="Which excavator costs the most?",
        trajectory_dir=tmp_path / "traj",
        dry_run=False,
    )

    expected_workspace = tmp_path / "openclaw-workspaces" / "openclaw_cli_agent_401"
    assert expected_workspace.exists()
    assert captured["cmd"] == [
        "uv",
        "run",
        "openclaw-cli-agent",
        "--model-id",
        "tokenrouter/MiniMax-M3",
        "--allow-files",
        "--allow-bash",
        "--thinking",
        "high",
        "--workspace-dir",
        str(expected_workspace),
        "--scenario-id",
        "401",
        "--run-id",
        "openclaw_cli_agent_401",
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
