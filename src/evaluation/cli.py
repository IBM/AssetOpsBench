"""``uv run evaluate`` — offline grading + report generation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import graders as grader_registry
from .report import render_summary, write_report
from .runner import evaluate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Grade saved agent trajectories against scenario files and "
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
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON report.",
    )
    p.add_argument(
        "--grader-default",
        default="llm_judge",
        help="Grader name when scenario.grading_method is unset. "
        "Default: llm_judge.",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        help="Model id for the LLM judge (e.g. "
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
    # Imported lazily so the CLI works for deterministic-only runs even
    # if the LiteLLM dep happens to be flaky in the dev environment.
    from llm import LiteLLMBackend  # type: ignore[import-not-found]

    from .graders.llm_judge import install

    install(LiteLLMBackend(model=judge_model))


def _validate_grader_default(name: str) -> None:
    try:
        grader_registry.get(name)
    except KeyError as exc:
        raise SystemExit(str(exc))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _maybe_install_judge(args.judge_model)
    _validate_grader_default(args.grader_default)

    report = evaluate(
        trajectories_path=args.trajectories,
        scenarios_paths=list(args.scenarios),
        default_grading_method=args.grader_default,
    )

    out = write_report(report, args.output)
    print(render_summary(report))
    print(f"\nReport written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
