"""Evaluator — orchestrates a set of scorers over a batch of records.

Mirrors MLflow's evaluator/scorer split: the :class:`Evaluator` owns
the loading + per-record dispatch, while each :data:`Scorer` is a small
callable that produces a single :class:`ScorerResult`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import scorers as scorer_registry
from .loader import join_records, load_scenarios, load_trajectories
from .metrics import metrics_from_trajectory
from .models import (
    EvalReport,
    PersistedTrajectory,
    Scenario,
    ScenarioResult,
)
from .report import build_report
from .scorers import Scorer

_log = logging.getLogger(__name__)


class Evaluator:
    """Run a batch of scenarios against their saved trajectories.

    ``default_scorer`` names the registered scorer to use when a
    scenario does not set ``scoring_method``.  Per-scenario overrides
    take precedence.
    """

    def __init__(
        self,
        default_scorer: str = "llm_judge",
        judge_model: str | None = None,
    ) -> None:
        self.default_scorer = default_scorer
        self.judge_model = judge_model

    def evaluate(
        self,
        *,
        trajectories_path: Path,
        scenarios_paths: list[Path],
    ) -> EvalReport:
        scenarios = load_scenarios(scenarios_paths)
        trajectories = load_trajectories(trajectories_path)

        results: list[ScenarioResult] = []
        for scenario, traj in join_records(scenarios, trajectories):
            results.append(self._score_one(scenario, traj))

        return build_report(results)

    def _score_one(
        self, scenario: Scenario, traj: PersistedTrajectory
    ) -> ScenarioResult:
        name = scenario.scoring_method or self.default_scorer
        scorer = self._resolve(name)
        self._validate_judge_model(name, traj)
        trajectory_text = _trajectory_to_text(traj)
        score = scorer(scenario, traj.answer, trajectory_text)

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_type=scenario.type,
            run_id=traj.run_id,
            runner=traj.runner,
            model=traj.model,
            question=traj.question,
            answer=traj.answer,
            score=score,
            ops=metrics_from_trajectory(traj),
        )

    @staticmethod
    def _resolve(name: str) -> Scorer:
        return scorer_registry.get(name)

    def _validate_judge_model(self, scorer_name: str, traj: PersistedTrajectory) -> None:
        if scorer_name != "llm_judge" or not self.judge_model:
            return

        trajectory_model = _normalize_model_id(traj.model)
        judge_model = _normalize_model_id(self.judge_model)
        if not trajectory_model or not judge_model:
            return

        if trajectory_model == judge_model:
            raise ValueError(
                "self-judging is not allowed for llm_judge: "
                f"trajectory model '{traj.model}' matches judge model '{self.judge_model}'"
            )


def _trajectory_to_text(traj: PersistedTrajectory) -> str:
    """Flatten a trajectory to a text blob for the LLM-As-Judge prompt."""
    if traj.trajectory is None:
        return ""
    try:
        return json.dumps(traj.trajectory, indent=2, default=str)
    except (TypeError, ValueError):
        return str(traj.trajectory)


def _normalize_model_id(model_id: str | None) -> str:
    if not model_id:
        return ""
    normalized = model_id.strip()
    if normalized.startswith("litellm_proxy/"):
        normalized = normalized[len("litellm_proxy/") :]
    return normalized
