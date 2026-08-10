"""``uv run evaluate`` — offline scoring + report generation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import scorers as scorer_registry
from .evaluator import Evaluator
from .report import render_summary, write_reports_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Score saved agent trajectories against scenario files and "
            "emit a JSON report."
        ),
    )
    p.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help=(
            "Directory recursively containing trajectory JSON files "
            "(or a single JSON file)."
        ),
    )
    p.add_argument(
        "--scenarios",
        type=Path,
        nargs="+",
        required=True,
        help="One or more scenario JSON / JSONL files or scenario directories.",
    )
    p.add_argument(
        "--scenario-ids",
        default=None,
        metavar="SELECTOR",
        help=(
            "Optional benchmark YAML selector such as fcc_lite or "
            "fcc+fmsr_all. Only matching scenario IDs are evaluated."
        ),
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help=(
            "Directory to write the combined '_aggregate.json' report. "
            "Default: reports/."
        ),
    )
    p.add_argument(
        "--scorer-default",
        dest="scorer_default",
        default="llm_judge",
        help="Scorer name when scenario.scoring_method is unset. Default: llm_judge.",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Model id for the LLM-As-Judge scorer (e.g. "
        "litellm_proxy/anthropic/claude-opus-4-5). "
        "Required when any scenario routes to llm_judge.",
    )
    p.add_argument(
        "--charts",
        action="store_true",
        help=(
            "Generate report-derived SVG and PNG leaderboards under "
            "<reports-dir>/charts. Requires the visualization dependency group."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return p


def _maybe_install_judge(judge_model: str | None) -> None:
    if not judge_model:
        return
    # Imported lazily so the CLI works for code-based-only runs even if
    # the LiteLLM dep happens to be flaky in the dev environment.
    from llm import make_backend  # type: ignore[import-not-found]

    from .scorers.llm_judge import install

    install(make_backend(judge_model))


def _validate_scorer_default(name: str) -> None:
    try:
        scorer_registry.get(name)
    except KeyError as exc:
        raise SystemExit(str(exc))


def _resolve_scenario_ids(selector: str | None) -> set[str] | None:
    """Resolve an optional benchmark scenario selector from all/lite YAML."""
    if selector is None:
        return None

    from benchmark.scenario_suite_runner import scenario_ids_for_selector

    return set(scenario_ids_for_selector(selector))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.charts:
        from .visualization import (
            VisualizationDependencyError,
            require_visualization_dependency,
        )

        try:
            require_visualization_dependency()
        except VisualizationDependencyError as exc:
            parser.error(str(exc))

    try:
        scenario_ids = _resolve_scenario_ids(args.scenario_ids)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    _maybe_install_judge(args.judge_model)
    _validate_scorer_default(args.scorer_default)

    report = Evaluator(
        default_scorer=args.scorer_default,
        judge_model=args.judge_model,
    ).evaluate(
        trajectories_path=args.trajectories,
        scenarios_paths=list(args.scenarios),
        scenario_ids=scenario_ids,
    )

    out_dir = write_reports_dir(report, args.reports_dir)
    chart_paths: tuple[Path, ...] = ()
    if args.charts:
        from .visualization import render_leaderboards

        chart_paths = render_leaderboards(report, out_dir / "charts")
    print(render_summary(report))
    print(f"\nAggregate report written: {out_dir}/_aggregate.json")
    if args.charts:
        if chart_paths:
            print(f"Leaderboard charts written: {out_dir}/charts")
        else:
            print(
                "No leaderboard charts generated: the report has no applicable "
                "Boolean llm_judge criterion results."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
