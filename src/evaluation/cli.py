"""CLI entry point for the skills evaluation benchmark.

Usage::

    uv run eval-skills --config configs/eval_skills.yaml
    uv run eval-skills --config configs/eval_skills.yaml --model claude-sonnet-4-6 --condition mcp_skills --runs 1
    uv run eval-skills --leaderboard results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval-skills",
        description="Run the AssetOpsBench skills evaluation benchmark.",
    )
    p.add_argument("--config", type=Path, help="Path to YAML config file.")
    p.add_argument("--model", type=str, help="Override: evaluate only this model.")
    p.add_argument(
        "--condition",
        type=str,
        choices=["no_skills", "mcp_only", "mcp_skills"],
        help="Override: evaluate only this condition.",
    )
    p.add_argument("--runs", type=int, help="Override: runs per scenario.")
    p.add_argument(
        "--ablation",
        type=str,
        choices=["procedure_only", "tool_list_only"],
        help="Run a single ablation variant instead of the full ablation list.",
    )
    p.add_argument(
        "--leaderboard",
        type=Path,
        help="Generate leaderboard from existing checkpoint JSON files in this dir.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return p.parse_args(argv)


def _generate_leaderboard_from_dir(results_dir: Path) -> None:
    """Re-generate leaderboard tables from checkpoint files."""
    from .runner import EvalConfig, ScenarioResult, aggregate_results
    from .leaderboard import generate_leaderboard

    all_results: list[ScenarioResult] = []
    for f in sorted(results_dir.glob("checkpoint_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for row in data:
            all_results.append(
                ScenarioResult(
                    scenario_id=row["scenario_id"],
                    model=row["model"],
                    condition=row["condition"],
                    run=row["run"],
                    passed=row["passed"],
                    method=row["method"],
                    category=row["category"],
                )
            )
    if not all_results:
        print(f"No checkpoint files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    config = EvalConfig(output_dir=results_dir)
    metrics = aggregate_results(all_results, config)
    paths = generate_leaderboard(metrics, results_dir)
    for p in paths:
        print(f"Generated: {p}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    # Leaderboard-only mode
    if args.leaderboard:
        _generate_leaderboard_from_dir(args.leaderboard)
        return

    if not args.config:
        print(
            "Error: --config is required (unless using --leaderboard).", file=sys.stderr
        )
        sys.exit(1)

    from .conditions import EvalCondition
    from .runner import (
        AblationConfig,
        EvalConfig,
        aggregate_results,
        run_full_evaluation,
    )
    from .leaderboard import generate_leaderboard

    config = EvalConfig.from_yaml(args.config)

    # Apply overrides
    if args.model:
        config.models = [args.model]
    if args.condition:
        config.conditions = [EvalCondition(args.condition)]
    if args.runs:
        config.runs_per_scenario = args.runs
    if args.ablation:
        from .ablation import SkillStructure

        config.ablations = [
            AblationConfig(
                label=args.ablation,
                structure=SkillStructure(args.ablation),
            )
        ]

    # Load scenarios
    scenarios = _load_scenarios()

    # Run evaluation
    results = asyncio.run(run_full_evaluation(scenarios, config))

    # Aggregate + leaderboard
    metrics = aggregate_results(results, config)
    paths = generate_leaderboard(metrics, config.output_dir)
    for p in paths:
        print(f"Generated: {p}")

    print(
        f"\nEvaluation complete: {len(results)} results across "
        f"{len(config.models)} model(s) × {len(config.conditions)} condition(s)."
    )


def _load_scenarios() -> list:
    """Load scenarios from HuggingFace dataset."""
    try:
        from tmp.assetopsbench.core.scenarios import Scenario

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="ibm-research/AssetOpsBench",
            filename="data/scenarios/all_utterance.jsonl",
            repo_type="dataset",
        )
        scenarios = []
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            data = json.loads(line)
            scenarios.append(Scenario(**data))
        return scenarios
    except Exception as exc:
        print(f"Warning: Could not load HuggingFace scenarios: {exc}", file=sys.stderr)
        print(
            "Provide scenarios via --scenarios or fix HuggingFace access.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
