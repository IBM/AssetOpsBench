"""Tests for EvalReport assembly and serialization."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import (
    OpsMetrics,
    ScenarioResult,
    ScorerResult,
)
from evaluation.report import (
    build_report,
    render_summary,
    write_report,
    write_reports_dir,
)


def _result(stype: str, passed: bool, run_id: str = "", **ops_kwargs) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="x",
        scenario_type=stype,
        run_id=run_id,
        runner="plan-execute",
        model="watsonx/ibm/granite",
        question="q",
        answer="a",
        score=ScorerResult(scorer="llm_judge", passed=passed, score=1.0 if passed else 0.0),
        ops=OpsMetrics(**ops_kwargs),
    )


def test_build_report_totals_and_breakdown():
    results = [
        _result("iot", True, tokens_in=10, tokens_out=5),
        _result("iot", False, tokens_in=8, tokens_out=4),
        _result("tsfm", True, tokens_in=20, tokens_out=10),
    ]
    report = build_report(results)

    assert report.totals == {
        "scenarios": 3,
        "scored": 3,
        "passed": 2,
        "pass_rate": round(2 / 3, 4),
    }
    assert report.by_scenario_type["iot"].total == 2
    assert report.by_scenario_type["iot"].passed == 1
    assert report.by_scenario_type["tsfm"].pass_rate == 1.0
    assert report.ops.tokens_in_total == 38
    summary_ops = report.score_summary["plan-execute_watsonx/ibm/granite"]["ops"]
    assert summary_ops == {
        "tokens_in_total": 38,
        "tokens_out_total": 19,
        "est_input_cost_usd_total": None,
        "est_output_cost_usd_total": None,
        "duration_ms_p50": None,
        "duration_ms_p95": None,
        "tool_calls_total": 0,
        "est_cost_usd_total": None,
    }


def test_build_report_handles_empty():
    report = build_report([])
    assert report.totals["scenarios"] == 0
    assert report.totals["pass_rate"] == 0.0
    assert report.by_scenario_type == {}


def test_write_report_round_trips(tmp_path: Path):
    results = [_result("iot", True)]
    report = build_report(results)
    out = write_report(report, tmp_path / "nested" / "report.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["totals"]["passed"] == 1
    assert data["by_scenario_type"]["iot"]["pass_rate"] == 1.0


def test_write_reports_dir_writes_only_aggregate(tmp_path: Path):
    results = [
        _result("iot", True, run_id="run-a"),
        _result("tsfm", False, run_id="run-b"),
    ]
    out_dir = write_reports_dir(build_report(results), tmp_path / "reports")

    assert (out_dir / "_aggregate.json").exists()
    assert sorted(path.name for path in out_dir.glob("*.json")) == ["_aggregate.json"]

    agg = json.loads((out_dir / "_aggregate.json").read_text())
    assert agg["totals"]["scenarios"] == 2
    assert set(agg["score_summary"]) == {"plan-execute_watsonx/ibm/granite"}
    score_summary = agg["score_summary"]["plan-execute_watsonx/ibm/granite"]
    assert score_summary["total"] == 2
    assert score_summary["passed"] == 1
    assert score_summary["pass_rate"] == 0.5
    assert score_summary["ops"]["tokens_in_total"] == 0
    assert len(agg["results"]) == 2


def test_render_summary_includes_headlines():
    results = [
        _result("iot", True, tokens_in=10, tokens_out=5, duration_ms=100.0, tool_call_count=1),
        _result("iot", False, tokens_in=8, tokens_out=4, duration_ms=200.0),
    ]
    text = render_summary(build_report(results))
    assert "Pass rate" in text
    assert "plan-execute_watsonx/ibm/granite" in text
    assert "passed: 1/2 (50.0%)" in text
    assert "iot" in text
    assert "tokens_in_total" in text

def test_build_report_includes_score_summary():
    from evaluation.models import ScenarioResult, ScorerResult, OpsMetrics

    results = [
        ScenarioResult(
            scenario_id="11",
            scenario_type="structured",
            run_id="direct_llm_11",
            runner="direct-llm-agent",
            model="tokenrouter/MiniMax-M3",
            question="Q",
            answer='{"energy":0,"material":0}',
            score=ScorerResult(
                scorer="static_json",
                passed=False,
                score=0.0,
                rationale="structured answer differs from ground truth",
                details={
                    "partial_exact_match_accuracy": 0.0,
                    "strict_exact_match_accuracy": 0.0,
                    "partial_similarity_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "total_gold_keys": 2,
                    "total_model_keys": 2,
                    "matched_keys": 2,
                    "exact_value_matches": 0,
                    "missing_keys": [],
                    "extra_keys": [],
                    "details": [],
                },
            ),
            ops=OpsMetrics(
                turn_count=1,
                tool_call_count=0,
                unique_tools=[],
                tokens_in=390,
                tokens_out=245,
                duration_ms=6224.3382,
                est_cost_usd=None,
            ),
        )
    ]

    report = build_report(results)

    assert report.score_summary is not None
    summary = report.score_summary["direct-llm-agent_tokenrouter/MiniMax-M3"]
    assert summary["total"] == 1
    assert summary["passed"] == 0
    assert summary["pass_rate"] == 0.0
    assert summary["partial_exact_match_accuracy_avg"] == 0.0
    assert summary["strict_exact_match_accuracy_avg"] == 0.0
    assert summary["missing_keys_total"] == 0


def test_build_report_groups_score_summary_by_runner_and_model():
    results = [
        _result("iot", True),
        ScenarioResult(
            scenario_id="y",
            scenario_type="iot",
            run_id="run-direct",
            runner="direct-llm-agent",
            model="tokenrouter/MiniMax-M3",
            question="q",
            answer="a",
            score=ScorerResult(
                scorer="static_json",
                passed=False,
                score=0.25,
            ),
            ops=OpsMetrics(),
        ),
    ]

    report = build_report(results)

    assert set(report.score_summary or {}) == {
        "direct-llm-agent_tokenrouter/MiniMax-M3",
        "plan-execute_watsonx/ibm/granite",
    }
    assert (
        report.score_summary["direct-llm-agent_tokenrouter/MiniMax-M3"][
            "score_avg"
        ]
        == 0.25
    )
