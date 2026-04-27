"""Tests for EvalReport assembly and serialization."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import (
    GradeResult,
    OpsMetrics,
    ScenarioResult,
)
from evaluation.report import build_report, render_summary, write_report


def _result(stype: str, passed: bool, **ops_kwargs) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="x",
        scenario_type=stype,
        runner="plan-execute",
        model="watsonx/ibm/granite",
        question="q",
        answer="a",
        grade=GradeResult(grading_method="llm_judge", passed=passed, score=1.0 if passed else 0.0),
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
        "graded": 3,
        "passed": 2,
        "pass_rate": round(2 / 3, 4),
    }
    assert report.by_scenario_type["iot"].total == 2
    assert report.by_scenario_type["iot"].passed == 1
    assert report.by_scenario_type["tsfm"].pass_rate == 1.0
    assert report.ops.tokens_in_total == 38


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


def test_render_summary_includes_headlines():
    results = [
        _result("iot", True, tokens_in=10, tokens_out=5, duration_ms=100.0, tool_call_count=1),
        _result("iot", False, tokens_in=8, tokens_out=4, duration_ms=200.0),
    ]
    text = render_summary(build_report(results))
    assert "Pass rate" in text
    assert "iot" in text
    assert "tokens_in_total" in text
