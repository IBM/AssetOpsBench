"""Glue: load → grade → assemble report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import graders as grader_registry
from .loader import join_records, load_scenarios, load_trajectories
from .metrics import metrics_from_trajectory
from .models import EvalReport, PersistedTrajectory, Scenario, ScenarioResult
from .report import build_report

_log = logging.getLogger(__name__)


def evaluate(
    *,
    trajectories_path: Path,
    scenarios_paths: list[Path],
    default_grading_method: str = "llm_judge",
) -> EvalReport:
    """Load, grade, and aggregate.

    Per-scenario grader is picked from ``scenario.grading_method`` when
    set, falling back to ``default_grading_method``.
    """
    scenarios = load_scenarios(scenarios_paths)
    trajectories = load_trajectories(trajectories_path)

    results: list[ScenarioResult] = []
    for scenario, traj in join_records(scenarios, trajectories):
        results.append(_grade_one(scenario, traj, default_grading_method))

    return build_report(results)


def _grade_one(
    scenario: Scenario,
    traj: PersistedTrajectory,
    default_grading_method: str,
) -> ScenarioResult:
    method = scenario.grading_method or default_grading_method
    grader = grader_registry.get(method)
    trajectory_text = _trajectory_to_text(traj)
    grade = grader(scenario, traj.answer, trajectory_text)

    return ScenarioResult(
        scenario_id=scenario.id,
        scenario_type=scenario.type,
        runner=traj.runner,
        model=traj.model,
        question=traj.question,
        answer=traj.answer,
        grade=grade,
        ops=metrics_from_trajectory(traj),
    )


def _trajectory_to_text(traj: PersistedTrajectory) -> str:
    """Flatten a trajectory to a text blob for the LLM judge prompt."""
    if traj.trajectory is None:
        return ""
    try:
        return json.dumps(traj.trajectory, indent=2, default=str)
    except (TypeError, ValueError):
        return str(traj.trajectory)
