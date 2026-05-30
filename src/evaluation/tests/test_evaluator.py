"""Tests for the Evaluator class — the orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation import scorers as registry
from evaluation.evaluator import Evaluator
from evaluation.models import Scenario, ScorerResult


def _stub_scorer(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    return ScorerResult(scorer="stub-evaluator", passed=True, score=1.0)


def test_evaluator_routes_to_default_scorer(tmp_path: Path, make_persisted_record):
    rec = make_persisted_record(run_id="run-1", scenario_id=1)
    (tmp_path / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "iot"}]),
        encoding="utf-8",
    )

    registry.register("stub-evaluator", _stub_scorer)

    report = Evaluator(default_scorer="stub-evaluator").evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
    )

    assert report.totals["passed"] == 1
    assert report.results[0].score.scorer == "stub-evaluator"


def _fail_scorer(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    return ScorerResult(scorer="fail-default", passed=False, score=0.0)


def test_evaluator_per_scenario_override_wins(tmp_path: Path, make_persisted_record):
    # The scenario-level scoring_method must route around the default
    # scorer, even when the default scorer would reject the answer.
    rec = make_persisted_record(run_id="run-1", scenario_id=1, answer="answer text")
    (tmp_path / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "text": "Q",
                    "type": "tsfm",
                    "scoring_method": "stub-evaluator",
                }
            ]
        ),
        encoding="utf-8",
    )

    registry.register("stub-evaluator", _stub_scorer)
    registry.register("fail-default", _fail_scorer)

    report = Evaluator(default_scorer="fail-default").evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
    )

    assert report.totals["passed"] == 1
    assert report.results[0].score.scorer == "stub-evaluator"
