"""Sequential runner for the benchmark scenarios.

This runner reads a simple scenario-id file, runs each scenario with the
selected agent method, saves trajectories through AGENT_TRAJECTORY_DIR, and
optionally invokes the existing evaluator to generate reports.

Example:

    uv run python -m benchmark.scenario_suite_runner \
      --scenario-ids benchmarks/scenario_suite/scenarios.txt \
      --scenario-root /path/to/scenarios_data \
      --agent_name direct_llm \
      --model-id tokenrouter/MiniMax-M3

The scenario root is expected to contain folders such as:

    scenarios_data/
      scenario_11/
        question.txt
        groundtruth.txt
      scenario_12/
        question.txt
        groundtruth.txt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_MODEL_ID = "tokenrouter/MiniMax-M3"
_DEFAULT_GEMINI_MODEL_ID = "tokenrouter_gemini/google/gemma-4-26b-a4b-it"


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for one benchmark method."""

    agent_name: str
    command: str
    model_id: str
    extra_args: tuple[str, ...] = ()
    workspace_root: Path | None = None
    reset_workspace: bool = True


def model_dir_name(model_id: str) -> str:
    """Return a filesystem-safe directory name for a model id."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id.strip().replace("/", "-"))
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "unknown-model"


def method_output_paths(
    *,
    trajectory_root: Path,
    reports_root: Path,
    method: MethodConfig,
) -> tuple[Path, Path, Path | None]:
    """Return trajectory, report, and workspace roots nested by agent/model."""
    model_dir = model_dir_name(method.model_id)
    trajectory_dir = trajectory_root / method.agent_name / model_dir
    report_dir = reports_root / method.agent_name / model_dir
    workspace_root = (
        method.workspace_root / method.agent_name / model_dir
        if method.workspace_root is not None
        else None
    )
    return trajectory_dir, report_dir, workspace_root


def load_scenario_ids(path: Path) -> list[str]:
    """Load scenario ids from a plain text file.

    The file format is intentionally simple:

        11
        12
        14

    Blank lines and lines beginning with '#' are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scenario id file not found: {path}")

    scenario_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        scenario_ids.append(line)

    if not scenario_ids:
        raise ValueError(f"No scenario ids found in {path}")

    return scenario_ids


def scenario_dir_for_id(scenario_root: Path, scenario_id: str) -> Path:
    """Return the expected scenario folder path for a scenario id."""
    return scenario_root / f"scenario_{scenario_id}"


def read_question(scenario_root: Path, scenario_id: str) -> str:
    """Read question.txt for a scenario."""
    scenario_dir = scenario_dir_for_id(scenario_root, scenario_id)
    question_path = scenario_dir / "question.txt"

    if not question_path.exists():
        raise FileNotFoundError(
            f"Missing question file for scenario {scenario_id}: {question_path}"
        )

    question = question_path.read_text(encoding="utf-8").strip()
    if not question:
        raise ValueError(
            f"Question file is empty for scenario {scenario_id}: {question_path}"
        )

    return question


def validate_groundtruth_exists(scenario_root: Path, scenario_id: str) -> None:
    """Warn if groundtruth.txt is missing.

    The agent run itself only needs question.txt, but evaluation needs
    groundtruth.txt.
    """
    scenario_dir = scenario_dir_for_id(scenario_root, scenario_id)
    groundtruth_path = scenario_dir / "groundtruth.txt"

    if not groundtruth_path.exists():
        print(
            f"warning: missing groundtruth for scenario {scenario_id}: {groundtruth_path}",
            file=sys.stderr,
        )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def validate_workspace_root_outside_repo(workspace_root: Path, label: str) -> None:
    """Reject CLI workspaces inside the repo to avoid report/trace leakage."""
    workspace_root = workspace_root.expanduser().resolve()
    if _is_relative_to(workspace_root, REPO_ROOT):
        raise ValueError(
            f"{label} must be outside the repository when file/bash/edit "
            f"access is enabled: {workspace_root}"
        )


def reset_and_load_couchdb(scenario_id: str, scenario_root: Path, dry_run: bool) -> None:
    """Reset CouchDB and load the scenario-specific data from scenario_root."""
    env = os.environ.copy()
    env["SCENARIOS_DATA_DIR"] = str(scenario_root)

    reset_cmd = [sys.executable, "src/couchdb/init_data.py", "--reset-only"]
    load_cmd = [sys.executable, "src/couchdb/init_data.py", scenario_id]

    print("\n" + "-" * 80)
    print(f"Preparing CouchDB for scenario {scenario_id}")
    print("Reset command:")
    print(" ".join(reset_cmd))
    print("Load command:")
    print(" ".join(load_cmd))
    print("-" * 80)

    if dry_run:
        return

    subprocess.run(reset_cmd, check=True, cwd=str(REPO_ROOT), env=env)
    subprocess.run(load_cmd, check=True, cwd=str(REPO_ROOT), env=env)


def run_agent_for_scenario(
    *,
    method: MethodConfig,
    scenario_id: str,
    question: str,
    trajectory_dir: Path,
    dry_run: bool,
    scenario_root: Path | None = None,
) -> None:
    """Run one scenario with one method."""
    run_id = f"{method.agent_name}_{scenario_id}"

    env = os.environ.copy()
    env["AGENT_TRAJECTORY_DIR"] = str(trajectory_dir)

    extra_args = list(method.extra_args)
    if method.workspace_root is not None:
        workspace_dir = method.workspace_root / run_id
        extra_args.extend(["--workspace-dir", str(workspace_dir)])
        if not dry_run:
            if method.reset_workspace and workspace_dir.exists():
                shutil.rmtree(workspace_dir)
            workspace_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        method.command,
        "--model-id",
        method.model_id,
        *extra_args,
        "--scenario-id",
        scenario_id,
        "--run-id",
        run_id,
        question,
    ]

    print("\n" + "=" * 80)
    print(f"Method:      {method.agent_name}")
    print(f"Scenario ID: {scenario_id}")
    print(f"Run ID:      {run_id}")
    print(f"Trajectories: {trajectory_dir}")
    print("Command:")
    print(" ".join(cmd[:-1]) + " <question>")
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(cmd, check=True, env=env)


def run_evaluation(
    *,
    trajectory_dir: Path,
    scenario_root: Path,
    report_dir: Path,
    dry_run: bool,
) -> None:
    """Run the existing AssetOpsBench evaluator for one method."""
    cmd = [
        "uv",
        "run",
        "evaluate",
        "--trajectories",
        str(trajectory_dir),
        "--scenarios",
        str(scenario_root),
        "--scorer-default",
        "static_json",
        "--reports-dir",
        str(report_dir),
    ]

    print("\n" + "=" * 80)
    print("Running evaluation")
    print(f"Trajectories: {trajectory_dir}")
    print(f"Scenarios:    {scenario_root}")
    print(f"Reports:      {report_dir}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)

    if dry_run:
        return

    subprocess.run(cmd, check=True)


def build_methods(args: argparse.Namespace) -> dict[str, MethodConfig]:
    """Build available method configs from CLI args."""
    stirrup_extra_args: list[str] = []
    stirrup_extra_args.extend(
        ["--max-tokens", str(getattr(args, "stirrup_max_tokens", 4096))]
    )
    temperature = getattr(args, "temperature", None)
    if temperature is not None:
        stirrup_extra_args.extend(["--temperature", str(temperature)])
    if getattr(args, "preserve_workspaces", False) and getattr(
        args, "stirrup_workspace_root", None
    ) is not None:
        stirrup_extra_args.append("--preserve-workspace")

    opencode_extra_args: list[str] = []
    if args.opencode_allow_files:
        opencode_extra_args.append("--allow-files")
    if args.opencode_allow_bash:
        opencode_extra_args.append("--allow-bash")
    if args.opencode_allow_edit:
        opencode_extra_args.append("--allow-edit")
    if getattr(args, "opencode_thinking", False):
        opencode_extra_args.append("--thinking")
    opencode_variant = getattr(args, "opencode_variant", None)
    if opencode_variant:
        opencode_extra_args.extend(["--variant", opencode_variant])

    gemini_extra_args: list[str] = []
    if args.gemini_allow_files:
        gemini_extra_args.append("--allow-files")
    if args.gemini_allow_bash:
        gemini_extra_args.append("--allow-bash")
    if args.gemini_allow_edit:
        gemini_extra_args.append("--allow-edit")
    if args.gemini_allow_web:
        gemini_extra_args.append("--allow-web")
    if args.gemini_sandbox:
        gemini_extra_args.append("--sandbox")

    openclaw_extra_args: list[str] = []
    if args.openclaw_allow_files:
        openclaw_extra_args.append("--allow-files")
    if args.openclaw_allow_bash:
        openclaw_extra_args.append("--allow-bash")
    if args.openclaw_allow_edit:
        openclaw_extra_args.append("--allow-edit")
    if args.openclaw_allow_web:
        openclaw_extra_args.append("--allow-web")
    if args.openclaw_thinking:
        openclaw_extra_args.extend(["--thinking", args.openclaw_thinking])

    return {
        "direct_llm": MethodConfig(
            agent_name="direct_llm",
            command="direct-llm-agent",
            model_id=args.model_id,
        ),
        "stirrup_agent": MethodConfig(
            agent_name="stirrup_agent",
            command="stirrup-agent",
            model_id=args.model_id,
            extra_args=tuple(stirrup_extra_args),
            workspace_root=getattr(args, "stirrup_workspace_root", None),
        ),
        "opencode_agent": MethodConfig(
            agent_name="opencode_agent",
            command="opencode-agent",
            model_id=args.model_id,
            extra_args=tuple(opencode_extra_args),
            workspace_root=args.opencode_workspace_root,
        ),
        "gemini_cli_agent": MethodConfig(
            agent_name="gemini_cli_agent",
            command="gemini-cli-agent",
            model_id=args.gemini_model_id,
            extra_args=tuple(gemini_extra_args),
            workspace_root=args.gemini_workspace_root,
        ),
        "openclaw_cli_agent": MethodConfig(
            agent_name="openclaw_cli_agent",
            command="openclaw-cli-agent",
            model_id=args.openclaw_model_id,
            extra_args=tuple(openclaw_extra_args),
            workspace_root=args.openclaw_workspace_root,
        ),
    }


def selected_methods(
    *,
    method_name: str,
    methods: dict[str, MethodConfig],
) -> list[MethodConfig]:
    """Resolve the requested method selection."""
    if method_name == "all":
        return list(methods.values())

    if method_name not in methods:
        valid = ", ".join(sorted([*methods.keys(), "all"]))
        raise ValueError(f"Unknown method '{method_name}'. Valid choices: {valid}")

    return [methods[method_name]]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scenario_suite_runner",
        description="Run benchmark scenarios sequentially.",
    )
    parser.add_argument(
        "--stirrup-max-tokens",
        type=int,
        default=4096,
        help="Max output tokens per Stirrup model call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature for stirrup_agent model calls. Omitted by "
            "default, so the provider/client default is used."
        ),
    )
    parser.add_argument(
        "--scenario-ids",
        type=Path,
        default=Path("benchmarks/scenario_suite/scenarios.txt"),
        help="Plain text file containing one scenario id per line.",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        required=True,
        help="Directory containing scenario_<id>/question.txt and groundtruth.txt folders.",
    )
    parser.add_argument(
        "--agent_name",
        choices=[
            "direct_llm",
            "stirrup_agent",
            "opencode_agent",
            "gemini_cli_agent",
            "openclaw_cli_agent",
            "all",
        ],
        default="direct_llm",
        help="Which agent to run.",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path("traces/trajectories/scenario_suite"),
        help="Root directory for saved trajectories; outputs are nested by agent/model.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports/scenario_suite"),
        help="Root directory for evaluation reports; outputs are nested by agent/model.",
    )
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL_ID,
        help="Model id used by direct_llm, stirrup_agent, and opencode_agent.",
    )
    parser.add_argument(
        "--stirrup-workspace-root",
        type=Path,
        default=None,
        help=(
            "Root directory for per-run Stirrup code-execution workspaces. "
            "Workspaces are nested by agent/model/run_id."
        ),
    )
    parser.add_argument(
        "--gemini-model-id",
        default=_DEFAULT_GEMINI_MODEL_ID,
        help=(
            "Model id used by gemini_cli_agent "
            f"(default: {_DEFAULT_GEMINI_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--openclaw-model-id",
        default=_DEFAULT_MODEL_ID,
        help=(
            "Model id used by openclaw_cli_agent "
            f"(default: {_DEFAULT_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--opencode-workspace-root",
        type=Path,
        default=None,
        help=(
            "Root directory for per-run OpenCode workspaces. Required when "
            "using --opencode-allow-files, --opencode-allow-bash, or "
            "--opencode-allow-edit. Workspaces are nested by agent/model/run_id."
        ),
    )
    parser.add_argument(
        "--opencode-allow-files",
        action="store_true",
        help="Allow opencode-agent read/glob/grep/lsp/list tools inside its per-run workspace.",
    )
    parser.add_argument(
        "--opencode-allow-bash",
        action="store_true",
        help="Allow opencode-agent bash in its per-run workspace.",
    )
    parser.add_argument(
        "--opencode-allow-edit",
        action="store_true",
        help="Allow opencode-agent file edits inside its per-run workspace.",
    )
    parser.add_argument(
        "--opencode-thinking",
        action="store_true",
        help="Pass --thinking to opencode-agent. Disabled by default.",
    )
    parser.add_argument(
        "--opencode-variant",
        default=None,
        help=(
            "OpenCode model variant / reasoning effort for opencode_agent, "
            "e.g. minimal, low, medium, high, or max. Omitted by default."
        ),
    )
    parser.add_argument(
        "--gemini-workspace-root",
        type=Path,
        default=None,
        help=(
            "Root directory for per-run Gemini CLI workspaces. Required when "
            "using --gemini-allow-files, --gemini-allow-bash, or "
            "--gemini-allow-edit. Workspaces are nested by agent/model/run_id."
        ),
    )
    parser.add_argument(
        "--gemini-allow-files",
        action="store_true",
        help="Allow gemini-cli-agent file inspection tools inside its per-run workspace.",
    )
    parser.add_argument(
        "--gemini-allow-bash",
        action="store_true",
        help="Allow gemini-cli-agent shell commands in its per-run workspace.",
    )
    parser.add_argument(
        "--gemini-allow-edit",
        action="store_true",
        help="Allow gemini-cli-agent file edits inside its per-run workspace.",
    )
    parser.add_argument(
        "--gemini-allow-web",
        action="store_true",
        help="Allow gemini-cli-agent web search/fetch. Disabled by default.",
    )
    parser.add_argument(
        "--gemini-sandbox",
        action="store_true",
        help="Pass --sandbox to Gemini CLI for supported local sandbox setups.",
    )
    parser.add_argument(
        "--openclaw-workspace-root",
        type=Path,
        default=None,
        help=(
            "Root directory for per-run OpenClaw workspaces. Required when "
            "using --openclaw-allow-files, --openclaw-allow-bash, or "
            "--openclaw-allow-edit. Workspaces are nested by agent/model/run_id."
        ),
    )
    parser.add_argument(
        "--openclaw-allow-files",
        action="store_true",
        help="Allow openclaw-cli-agent file inspection tools inside its per-run workspace.",
    )
    parser.add_argument(
        "--openclaw-allow-bash",
        action="store_true",
        help="Allow openclaw-cli-agent shell commands in its per-run workspace.",
    )
    parser.add_argument(
        "--openclaw-allow-edit",
        action="store_true",
        help="Allow openclaw-cli-agent file edits inside its per-run workspace.",
    )
    parser.add_argument(
        "--openclaw-allow-web",
        action="store_true",
        help="Allow openclaw-cli-agent web search/fetch. Disabled by default.",
    )
    parser.add_argument(
        "--openclaw-thinking",
        default="off",
        help="OpenClaw thinking level for openclaw_cli_agent (default: off).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a scenario if its expected trajectory file already exists.",
    )
    parser.add_argument(
        "--preserve-workspaces",
        action="store_true",
        help=(
            "Keep existing per-run workspaces instead of deleting them before a run. "
            "For Stirrup, also preserve final code-execution files in --stirrup-workspace-root."
        ),
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Run agents only; do not invoke evaluator after the run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later scenarios if one scenario fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    opencode_workspace_required = (
        args.opencode_allow_files
        or args.opencode_allow_bash
        or args.opencode_allow_edit
    )
    if opencode_workspace_required and args.opencode_workspace_root is None:
        parser.error(
            "--opencode-workspace-root is required when enabling OpenCode "
            "files, bash, or edits"
        )
    if opencode_workspace_required:
        try:
            validate_workspace_root_outside_repo(
                args.opencode_workspace_root, "--opencode-workspace-root"
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.stirrup_workspace_root is not None:
        try:
            validate_workspace_root_outside_repo(
                args.stirrup_workspace_root, "--stirrup-workspace-root"
            )
        except ValueError as exc:
            parser.error(str(exc))
    gemini_workspace_required = (
        args.gemini_allow_files or args.gemini_allow_bash or args.gemini_allow_edit
    )
    if gemini_workspace_required and args.gemini_workspace_root is None:
        parser.error(
            "--gemini-workspace-root is required when enabling Gemini CLI "
            "files, bash, or edits"
        )
    openclaw_workspace_required = (
        args.openclaw_allow_files
        or args.openclaw_allow_bash
        or args.openclaw_allow_edit
    )
    if openclaw_workspace_required and args.openclaw_workspace_root is None:
        parser.error(
            "--openclaw-workspace-root is required when enabling OpenClaw "
            "files, bash, or edits"
        )

    scenario_ids = load_scenario_ids(args.scenario_ids)
    methods = selected_methods(
        method_name=args.agent_name,
        methods=build_methods(args),
    )

    print(f"Loaded {len(scenario_ids)} scenario ids from {args.scenario_ids}")
    print(f"Selected methods: {', '.join(method.agent_name for method in methods)}")

    for method in methods:
        trajectory_dir, report_dir, method_workspace_root = method_output_paths(
            trajectory_root=args.trajectory_root,
            reports_root=args.reports_root,
            method=method,
        )
        method_for_run = MethodConfig(
            agent_name=method.agent_name,
            command=method.command,
            model_id=method.model_id,
            extra_args=method.extra_args,
            workspace_root=method_workspace_root,
            reset_workspace=method.reset_workspace and not args.preserve_workspaces,
        )

        if not args.dry_run:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)

        for scenario_id in scenario_ids:
            expected_trajectory = trajectory_dir / f"{method.agent_name}_{scenario_id}.json"

            if args.skip_existing and expected_trajectory.exists():
                print(
                    f"Skipping scenario {scenario_id}; trajectory exists: {expected_trajectory}"
                )
                continue

            try:
                validate_groundtruth_exists(args.scenario_root, scenario_id)
                question = read_question(args.scenario_root, scenario_id)

                # Uniform CouchDB preparation for every agent and every scenario.
                reset_and_load_couchdb(
                    scenario_id=scenario_id,
                    scenario_root=args.scenario_root,
                    dry_run=args.dry_run,
                )

                run_agent_for_scenario(
                    method=method_for_run,
                    scenario_id=scenario_id,
                    question=question,
                    trajectory_dir=trajectory_dir,
                    scenario_root=args.scenario_root,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                print(
                    f"error: scenario {scenario_id} failed for method {method.agent_name}: {exc}",
                    file=sys.stderr,
                )
                if not args.continue_on_error:
                    raise

        if not args.no_evaluate:
            run_evaluation(
                trajectory_dir=trajectory_dir,
                scenario_root=args.scenario_root,
                report_dir=report_dir,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
