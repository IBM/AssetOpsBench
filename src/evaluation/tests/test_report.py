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


def test_write_reports_dir_per_run_files(tmp_path: Path):
    results = [
        _result("iot", True, run_id="run-a"),
        _result("tsfm", False, run_id="run-b"),
    ]
    out_dir = write_reports_dir(build_report(results), tmp_path / "reports")

    assert (out_dir / "run-a.json").exists()
    assert (out_dir / "run-b.json").exists()
    assert (out_dir / "_aggregate.json").exists()

    per_run = json.loads((out_dir / "run-a.json").read_text())
    assert per_run["run_id"] == "run-a"
    assert per_run["score"]["passed"] is True

    agg = json.loads((out_dir / "_aggregate.json").read_text())
    assert agg["totals"]["scenarios"] == 2


def test_write_reports_dir_falls_back_to_scenario_id(tmp_path: Path):
    # ScenarioResult.run_id is empty when the trajectory pre-dates the
    # run_id field; the writer must still produce a file.
    results = [_result("iot", True)]
    out_dir = write_reports_dir(build_report(results), tmp_path / "reports")
    assert (out_dir / "scenario-x.json").exists()


def test_render_summary_includes_headlines():
    results = [
        _result("iot", True, tokens_in=10, tokens_out=5, duration_ms=100.0, tool_call_count=1),
        _result("iot", False, tokens_in=8, tokens_out=4, duration_ms=200.0),
    ]
    text = render_summary(build_report(results))
    assert "Pass rate" in text
    assert "iot" in text
    assert "tokens_in_total" in text
