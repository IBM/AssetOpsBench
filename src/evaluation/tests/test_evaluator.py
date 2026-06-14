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


def test_evaluator_rejects_self_judging_model(tmp_path: Path, make_persisted_record):
    trajectories_dir = tmp_path / "trajectories"
    trajectories_dir.mkdir()

    rec = make_persisted_record(
        run_id="run-1",
        scenario_id=1,
        model="litellm_proxy/aws/claude-opus-4-6",
    )
    (trajectories_dir / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "iot"}]),
        encoding="utf-8",
    )

    registry.register("llm_judge", _stub_scorer)

    try:
        Evaluator(
            default_scorer="llm_judge",
            judge_model="litellm_proxy/aws/claude-opus-4-6",
        ).evaluate(
            trajectories_path=trajectories_dir,
            scenarios_paths=[scenarios_path],
        )
    except ValueError as exc:
        assert "self-judging is not allowed" in str(exc)
    else:
        raise AssertionError("expected ValueError for self-judging")


def test_evaluator_rejects_self_judging_with_normalized_model_ids(
    tmp_path: Path, make_persisted_record
):
    trajectories_dir = tmp_path / "trajectories"
    trajectories_dir.mkdir()

    rec = make_persisted_record(
        run_id="run-1",
        scenario_id=1,
        model="litellm_proxy/aws/claude-opus-4-6",
    )
    (trajectories_dir / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "iot"}]),
        encoding="utf-8",
    )

    registry.register("llm_judge", _stub_scorer)

    try:
        Evaluator(
            default_scorer="llm_judge",
            judge_model="aws/claude-opus-4-6",
        ).evaluate(
            trajectories_path=trajectories_dir,
            scenarios_paths=[scenarios_path],
        )
    except ValueError as exc:
        assert "self-judging is not allowed" in str(exc)
    else:
        raise AssertionError("expected ValueError for self-judging")


def test_evaluator_allows_non_llm_judge_even_with_matching_model(
    tmp_path: Path, make_persisted_record
):
    rec = make_persisted_record(
        run_id="run-1",
        scenario_id=1,
        model="litellm_proxy/aws/claude-opus-4-6",
    )
    (tmp_path / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "iot"}]),
        encoding="utf-8",
    )

    registry.register("stub-evaluator", _stub_scorer)

    report = Evaluator(
        default_scorer="stub-evaluator",
        judge_model="aws/claude-opus-4-6",
    ).evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
    )

    assert report.totals["passed"] == 1
