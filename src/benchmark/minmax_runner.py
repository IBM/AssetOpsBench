"""Sequential runner for the MinMax benchmark scenarios.

This runner reads a simple scenario-id file, runs each scenario with the
selected agent method, saves trajectories through AGENT_TRAJECTORY_DIR, and
optionally invokes the existing evaluator to generate reports.

Example:

    uv run python -m benchmark.minmax_runner \
      --scenario-ids benchmarks/minmax/scenarios.txt \
      --scenario-root /path/to/scenarios_data \
      --method direct_llm

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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_DIRECT_LLM_MODEL = "tokenrouter/MiniMax-M3"
_DEFAULT_STIRRUP_MODEL = "litellm_proxy/aws/claude-opus-4-8"

@dataclass(frozen=True)
class MethodConfig:
    """Configuration for one benchmark method."""

    name: str
    command: str
    model_id: str


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
        raise ValueError(f"Question file is empty for scenario {scenario_id}: {question_path}")

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


def run_agent_for_scenario(
    *,
    method: MethodConfig,
    scenario_id: str,
    question: str,
    trajectory_dir: Path,
    dry_run: bool,
) -> None:
    """Run one scenario with one method."""
    run_id = f"{method.name}_{scenario_id}"

    env = os.environ.copy()
    env["AGENT_TRAJECTORY_DIR"] = str(trajectory_dir)

    cmd = [
        "uv",
        "run",
        method.command,
        "--model-id",
        method.model_id,
        "--scenario-id",
        scenario_id,
        "--run-id",
        run_id,
        question,
    ]

    print("\n" + "=" * 80)
    print(f"Method:      {method.name}")
    print(f"Scenario ID: {scenario_id}")
    print(f"Run ID:      {run_id}")
    print(f"Trajectories:{trajectory_dir}")
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
    return {
        "direct_llm": MethodConfig(
            name="direct_llm",
            command="direct-llm-agent",
            model_id=args.direct_model_id,
        ),
        "stirrup_agent": MethodConfig(
            name="stirrup_agent",
            command="stirrup-agent",
            model_id=args.stirrup_model_id,
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
        prog="minmax-runner",
        description="Run MinMax benchmark scenarios sequentially.",
    )

    parser.add_argument(
        "--scenario-ids",
        type=Path,
        default=Path("benchmarks/minmax/scenarios.txt"),
        help="Plain text file containing one scenario id per line.",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        required=True,
        help="Directory containing scenario_<id>/question.txt and groundtruth.txt folders.",
    )
    parser.add_argument(
        "--method",
        choices=["direct_llm", "stirrup_agent", "all"],
        default="direct_llm",
        help="Which method to run.",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path("traces/trajectories/minmax"),
        help="Root directory for saved trajectories.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports/minmax"),
        help="Root directory for evaluation reports.",
    )
    parser.add_argument(
        "--direct-model-id",
        default=_DEFAULT_DIRECT_LLM_MODEL,
        help="Model id for direct_llm.",
    )
    parser.add_argument(
        "--stirrup-model-id",
        default=_DEFAULT_STIRRUP_MODEL,
        help="Model id for stirrup_agent.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a scenario if its expected trajectory file already exists.",
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

    scenario_ids = load_scenario_ids(args.scenario_ids)
    methods = selected_methods(
        method_name=args.method,
        methods=build_methods(args),
    )

    print(f"Loaded {len(scenario_ids)} scenario ids from {args.scenario_ids}")
    print(f"Selected methods: {', '.join(method.name for method in methods)}")

    for method in methods:
        trajectory_dir = args.trajectory_root / method.name
        report_dir = args.reports_root / method.name

        if not args.dry_run:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)

        for scenario_id in scenario_ids:
            expected_trajectory = trajectory_dir / f"{method.name}_{scenario_id}.json"

            if args.skip_existing and expected_trajectory.exists():
                print(f"Skipping scenario {scenario_id}; trajectory exists: {expected_trajectory}")
                continue

            try:
                validate_groundtruth_exists(args.scenario_root, scenario_id)
                question = read_question(args.scenario_root, scenario_id)

                run_agent_for_scenario(
                    method=method,
                    scenario_id=scenario_id,
                    question=question,
                    trajectory_dir=trajectory_dir,
                    dry_run=args.dry_run,
                )
            except Exception as exc:
                print(
                    f"error: scenario {scenario_id} failed for method {method.name}: {exc}",
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