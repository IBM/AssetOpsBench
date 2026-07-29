"""Tests for the Evaluator class — the orchestration layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_evaluator_filters_to_selected_scenario_ids(
    tmp_path: Path, make_persisted_record
):
    for scenario_id in (301, 302):
        record = make_persisted_record(
            run_id=f"run-{scenario_id}",
            scenario_id=scenario_id,
        )
        (tmp_path / f"run-{scenario_id}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {"id": 301, "text": "Q301", "type": "fcc"},
                {"id": 302, "text": "Q302", "type": "fcc"},
            ]
        ),
        encoding="utf-8",
    )

    registry.register("stub-evaluator", _stub_scorer)
    report = Evaluator(default_scorer="stub-evaluator").evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
        scenario_ids={"302"},
    )

    assert report.totals["scenarios"] == 1
    assert [result.scenario_id for result in report.results] == ["302"]
    summary = report.score_summary["plan-execute_watsonx/ibm/granite"]
    assert summary["scored_results"] == 1


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


def test_evaluator_uses_repaired_answer_field(
    tmp_path: Path, make_persisted_record
):
    seen: dict[str, str] = {}

    def capture_scorer(
        scenario: Scenario, answer: str, trajectory_text: str
    ) -> ScorerResult:
        seen["answer"] = answer
        return ScorerResult(scorer="capture-repair", passed=True, score=1.0)

    rec = make_persisted_record(
        run_id="run-1",
        scenario_id=1,
        answer="original answer",
        answer_repair="repaired answer",
    )
    (tmp_path / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "wo"}]),
        encoding="utf-8",
    )

    registry.register("capture-repair", capture_scorer)
    report = Evaluator(
        default_scorer="capture-repair",
        answer_field="answer_repair",
    ).evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
    )

    assert seen["answer"] == "repaired answer"
    assert report.results[0].answer == "repaired answer"


def test_evaluator_defaults_to_original_answer_field(
    tmp_path: Path, make_persisted_record
):
    seen: dict[str, str] = {}

    def capture_scorer(
        scenario: Scenario, answer: str, trajectory_text: str
    ) -> ScorerResult:
        seen["answer"] = answer
        return ScorerResult(scorer="capture-default", passed=True, score=1.0)

    rec = make_persisted_record(
        run_id="run-1",
        scenario_id=1,
        answer="original answer",
        answer_repair="repaired answer",
    )
    (tmp_path / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "wo"}]),
        encoding="utf-8",
    )

    registry.register("capture-default", capture_scorer)
    report = Evaluator(default_scorer="capture-default").evaluate(
        trajectories_path=tmp_path,
        scenarios_paths=[scenarios_path],
    )

    assert seen["answer"] == "original answer"
    assert report.results[0].answer == "original answer"


def test_evaluator_rejects_missing_repaired_answer_field(
    tmp_path: Path, make_persisted_record
):
    rec = make_persisted_record(run_id="run-1", scenario_id=1)
    (tmp_path / "run-1.json").write_text(json.dumps(rec), encoding="utf-8")

    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps([{"id": 1, "text": "Q", "type": "wo"}]),
        encoding="utf-8",
    )

    registry.register("capture-repair", _stub_scorer)
    with pytest.raises(ValueError, match="answer_repair"):
        Evaluator(
            default_scorer="capture-repair",
            answer_field="answer_repair",
        ).evaluate(
            trajectories_path=tmp_path,
            scenarios_paths=[scenarios_path],
        )


def _fail_scorer(scenario: Scenario, answer: str, trajectory_text: str) -> ScorerResult:
    return ScorerResult(scorer="fail-default", passed=False, score=0.0)


def test_evaluator_per_scenario_override_wins(tmp_path: Path, make_persisted_record):
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
