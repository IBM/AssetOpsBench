"""Smoke test for the end-to-end evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import GradeResult, Scenario
from evaluation.runner import evaluate
from evaluation import graders as registry


def _always_pass_grader(scenario: Scenario, answer: str, trajectory_text: str) -> GradeResult:
    return GradeResult(grading_method="stub", passed=True, score=1.0)


def test_evaluate_end_to_end(tmp_path: Path, make_persisted_record):
    # Two trajectories, both joinable to scenarios.
    rec_a = make_persisted_record(run_id="run-a", scenario_id=1, answer="A")
    rec_b = make_persisted_record(run_id="run-b", scenario_id=2, answer="B")
    (tmp_path / "run-a.json").write_text(json.dumps(rec_a), encoding="utf-8")
    (tmp_path / "run-b.json").write_text(json.dumps(rec_b), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {"id": 1, "text": "Q1", "type": "iot"},
                {"id": 2, "text": "Q2", "type": "tsfm"},
            ]
        ),
        encoding="utf-8",
    )

    registry.register("stub", _always_pass_grader)

    report = evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
        default_grading_method="stub",
    )

    assert report.totals["scenarios"] == 2
    assert report.totals["passed"] == 2
    assert set(report.by_scenario_type.keys()) == {"iot", "tsfm"}
    assert report.ops.tokens_in_total > 0


def test_evaluate_uses_per_scenario_grading_method(tmp_path: Path, make_persisted_record):
    rec = make_persisted_record(run_id="run-x", scenario_id=1)
    (tmp_path / "run-x.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "text": "Q",
                    "type": "iot",
                    "expected_answer": "A.",
                    "grading_method": "exact_string_match",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
        default_grading_method="numeric_match",  # would fail; per-scenario override wins
    )

    assert report.totals["passed"] == 1
    assert report.results[0].grade.grading_method == "exact_string_match"
