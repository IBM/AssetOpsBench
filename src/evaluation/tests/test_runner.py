"""Smoke test for the end-to-end evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import Scenario, ScorerResult
from evaluation.runner import evaluate
from evaluation import scorers as registry


def _always_pass_scorer(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    return ScorerResult(scorer="stub", passed=True, score=1.0)


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

    registry.register("stub", _always_pass_scorer)

    report = evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
        default_scoring_method="stub",
    )

    assert report.totals["scenarios"] == 2
    assert report.totals["passed"] == 2
    assert set(report.by_scenario_type.keys()) == {"iot", "tsfm"}
    assert report.ops.tokens_in_total > 0


def _always_fail_scorer(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    return ScorerResult(scorer="stub-fail", passed=False, score=0.0)


def test_evaluate_uses_per_scenario_scoring_method(tmp_path: Path, make_persisted_record):
    rec = make_persisted_record(run_id="run-x", scenario_id=1, answer="A.")
    (tmp_path / "run-x.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "text": "Q",
                    "type": "iot",
                    "scoring_method": "stub-pass",
                }
            ]
        ),
        encoding="utf-8",
    )

    registry.register("stub-pass", _always_pass_scorer)
    registry.register("stub-fail", _always_fail_scorer)

    report = evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
        default_scoring_method="stub-fail",  # per-scenario override wins
    )

    # Override wins: scenario routed to the always-pass stub even though
    # the default scorer would have failed it.
    assert report.totals["passed"] == 1
