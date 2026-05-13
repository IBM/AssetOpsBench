"""Backwards-friendly functional entry point delegating to :class:`Evaluator`."""

from __future__ import annotations

from pathlib import Path

from .evaluator import Evaluator
from .models import EvalReport


def evaluate(
    *,
    trajectories_path: Path,
    scenarios_paths: list[Path],
    default_grading_method: str = "llm_judge",
) -> EvalReport:
    """Load, score, and aggregate.

    Per-scenario scorer is picked from ``scenario.grading_method`` when
    set, falling back to ``default_grading_method``.
    """
    return Evaluator(default_scorer=default_grading_method).evaluate(
        trajectories_path=trajectories_path,
        scenarios_paths=scenarios_paths,
    )
