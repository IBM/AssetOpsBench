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
        help="Directory of {run_id}.json trajectory files (or a single file).",
    )
    p.add_argument(
        "--scenarios",
        type=Path,
        nargs="+",
        required=True,
        help="One or more scenario JSON / JSONL files.",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help=(
            "Directory to write per-run JSON reports (one file per run, "
            "named '<run_id>.json'), plus '_aggregate.json' for the rollup. "
            "Default: reports/."
        ),
    )
    p.add_argument(
        "--scorer-default",
        dest="scorer_default",
        default="llm_judge",
        help="Scorer name when scenario.scoring_method is unset. "
        "Default: llm_judge.",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Model id for the LLM-As-Judge scorer (e.g. "
        "litellm_proxy/anthropic/claude-opus-4-5). "
        "Required when any scenario routes to llm_judge.",
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
    from llm import LiteLLMBackend  # type: ignore[import-not-found]

    from .scorers.llm_judge import install

    install(LiteLLMBackend(model_id=judge_model))


def _validate_scorer_default(name: str) -> None:
    try:
        scorer_registry.get(name)
    except KeyError as exc:
        raise SystemExit(str(exc))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _maybe_install_judge(args.judge_model)
    _validate_scorer_default(args.scorer_default)

    report = Evaluator(default_scorer=args.scorer_default).evaluate(
        trajectories_path=args.trajectories,
        scenarios_paths=list(args.scenarios),
    )

    out_dir = write_reports_dir(report, args.reports_dir)
    print(render_summary(report))
    print(f"\nReports written: {out_dir}/<run_id>.json ({len(report.results)} files)")
    print(f"Aggregate:       {out_dir}/_aggregate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
