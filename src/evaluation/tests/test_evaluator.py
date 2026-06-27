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


def test_evaluator_strips_think_blocks_before_scoring(
    tmp_path: Path, make_persisted_record
):
    seen: dict[str, str] = {}

    def capture_scorer(
        scenario: Scenario, answer: str, trajectory_text: str
    ) -> ScorerResult:
        seen["answer"] = answer
        return ScorerResult(scorer="capture-evaluator", passed=True, score=1.0)

    rec = make_persisted_record(
        run_id="run-1",
        scenario_id=1,
        answer=(
            "<think>I should inspect the work orders.</think>\n\n"
            "<think>There are no kit entries.</think>\n\n"
            "0"
        ),
    )
    (tmp_path / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "wo"}]),
        encoding="utf-8",
    )

    registry.register("capture-evaluator", capture_scorer)

    report = Evaluator(default_scorer="capture-evaluator").evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
    )

    assert seen["answer"] == "0"
    assert report.results[0].answer == "0"


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