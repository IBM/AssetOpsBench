"""Build an :class:`EvalReport` from scored scenario results."""

from __future__ import annotations

import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .metrics import aggregate_ops
from .models import EvalReport, ScenarioResult, TypeBreakdown

_AGGREGATE_FILENAME = "_aggregate.json"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _aggregate_score_summary(results: list[ScenarioResult]) -> dict[str, Any]:
    """Aggregate static_json-style score.details across one result group.

    Per-scenario key-level details stay in each result. Here we summarize the
    numeric metrics and count totals across the runner/model group.
    """
    metric_names = [
        "partial_match_accuracy",
        "partial_exact_match_accuracy",
        "strict_exact_match_accuracy",
        "partial_similarity_score",
        "partial_numeric_match_accuracy",
        "range_match_accuracy",
        "delta_1_match_accuracy",
        "precision",
        "recall",
        "f1",
        "total_gold_keys",
        "total_model_keys",
        "matched_keys",
        "accepted_value_matches",
        "exact_value_matches",
        "numeric_gold_keys",
        "numeric_value_matches",
        "range_eligible_keys",
        "range_value_matches",
        "delta_1_eligible_keys",
        "delta_1_value_matches",
        "mode_key_match",
        "mode_exactly_one_key",
        "mode_term_coverage",
    ]

    score_values: dict[str, list[float]] = {name: [] for name in metric_names}
    score_values["score"] = []

    missing_keys_total = 0
    extra_keys_total = 0
    detail_entries_total = 0
    scored_results = 0

    for result in results:
        # Top-level score field, if present
        score_value = _safe_float(result.score.score)
        if score_value is not None:
            score_values["score"].append(score_value)

        details = result.score.details
        if not isinstance(details, dict):
            continue

        scored_results += 1

        for name in metric_names:
            value = _safe_float(details.get(name))
            if value is not None:
                score_values[name].append(value)

        missing = details.get("missing_keys")
        if isinstance(missing, list):
            missing_keys_total += len(missing)

        extra = details.get("extra_keys")
        if isinstance(extra, list):
            extra_keys_total += len(extra)

        per_key_details = details.get("details")
        if isinstance(per_key_details, list):
            detail_entries_total += len(per_key_details)

    return {
        "scored_results": scored_results,
        "score_avg": _avg(score_values["score"]),
        "score_min": round(min(score_values["score"]), 4) if score_values["score"] else None,
        "score_max": round(max(score_values["score"]), 4) if score_values["score"] else None,
        "partial_match_accuracy_avg": _avg(score_values["partial_match_accuracy"]),
        "partial_exact_match_accuracy_avg": _avg(score_values["partial_exact_match_accuracy"]),
        "strict_exact_match_accuracy_avg": _avg(score_values["strict_exact_match_accuracy"]),
        "partial_similarity_score_avg": _avg(score_values["partial_similarity_score"]),
        "partial_numeric_match_accuracy_avg": _avg(
            score_values["partial_numeric_match_accuracy"]
        ),
        "range_match_accuracy_avg": _avg(score_values["range_match_accuracy"]),
        "delta_1_match_accuracy_avg": _avg(score_values["delta_1_match_accuracy"]),
        "precision_avg": _avg(score_values["precision"]),
        "recall_avg": _avg(score_values["recall"]),
        "f1_avg": _avg(score_values["f1"]),
        "total_gold_keys_avg": _avg(score_values["total_gold_keys"]),
        "total_model_keys_avg": _avg(score_values["total_model_keys"]),
        "matched_keys_avg": _avg(score_values["matched_keys"]),
        "accepted_value_matches_avg": _avg(score_values["accepted_value_matches"]),
        "exact_value_matches_avg": _avg(score_values["exact_value_matches"]),
        "numeric_gold_keys_avg": _avg(score_values["numeric_gold_keys"]),
        "numeric_value_matches_avg": _avg(score_values["numeric_value_matches"]),
        "range_eligible_keys_avg": _avg(score_values["range_eligible_keys"]),
        "range_value_matches_avg": _avg(score_values["range_value_matches"]),
        "delta_1_eligible_keys_avg": _avg(score_values["delta_1_eligible_keys"]),
        "delta_1_value_matches_avg": _avg(score_values["delta_1_value_matches"]),
        "mode_key_match_avg": _avg(score_values["mode_key_match"]),
        "mode_exactly_one_key_avg": _avg(score_values["mode_exactly_one_key"]),
        "mode_term_coverage_avg": _avg(score_values["mode_term_coverage"]),
        "missing_keys_total": missing_keys_total,
        "extra_keys_total": extra_keys_total,
        "detail_entries_total": detail_entries_total,
    }


def _score_summaries_by_runner_model(
    results: list[ScenarioResult],
) -> dict[str, dict[str, Any]]:
    """Group score summaries under ``<runner>_<model>`` keys."""
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[f"{result.runner}_{result.model}"].append(result)

    return {
        key: _aggregate_score_summary(grouped[key])
        for key in sorted(grouped)
    }


def build_report(results: list[ScenarioResult]) -> EvalReport:
    total = len(results)
    passed = sum(1 for r in results if r.score.passed)

    by_type: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_type[r.scenario_type or "unknown"].append(r)

    breakdown: dict[str, TypeBreakdown] = {}
    for stype, items in by_type.items():
        n = len(items)
        p = sum(1 for r in items if r.score.passed)
        breakdown[stype] = TypeBreakdown(
            total=n,
            passed=p,
            pass_rate=round(p / n, 4) if n else 0.0,
        )

    return EvalReport(
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        runners=sorted({r.runner for r in results}),
        models=sorted({r.model for r in results}),
        totals={
            "scenarios": total,
            "scored": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        },
        by_scenario_type=breakdown,
        ops=aggregate_ops(results),
        score_summary=_score_summaries_by_runner_model(results),
        results=results,
    )


def write_report(report: EvalReport, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output


def write_reports_dir(report: EvalReport, reports_dir: Path) -> Path:
    """Write only the aggregate report and return its directory path."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / _AGGREGATE_FILENAME).write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return reports_dir


def render_summary(report: EvalReport) -> str:
    lines: list[str] = []
    t = report.totals
    lines.append(
        f"Scenarios: {t.get('scenarios', 0)}  "
        f"Passed: {t.get('passed', 0)}  "
        f"Pass rate: {t.get('pass_rate', 0):.1%}"
    )

    if report.score_summary:
        lines.append("")
        lines.append("Score summary by runner/model:")
        for key, summary in sorted(report.score_summary.items()):
            lines.append(f"  {key}:")
            _append_score_summary(lines, summary, indent="    ")

    if report.by_scenario_type:
        lines.append("")
        lines.append("By scenario type:")
        for stype, b in sorted(report.by_scenario_type.items()):
            lines.append(
                f"  {stype:<16} {b.passed:>4}/{b.total:<4}  ({b.pass_rate:.1%})"
            )

    o = report.ops
    lines.append("")
    lines.append("Operational metrics:")
    lines.append(f"  tokens_in_total:   {o.tokens_in_total}")
    lines.append(f"  tokens_out_total:  {o.tokens_out_total}")
    lines.append(f"  tool_calls_total:  {o.tool_calls_total}")
    if o.duration_ms_p50 is not None:
        lines.append(f"  duration_ms_p50:   {o.duration_ms_p50:.1f}")
    if o.duration_ms_p95 is not None:
        lines.append(f"  duration_ms_p95:   {o.duration_ms_p95:.1f}")
    if o.est_cost_usd_total is not None:
        lines.append(f"  est_cost_usd:      ${o.est_cost_usd_total:.4f}")
    return "\n".join(lines)


def _append_score_summary(
    lines: list[str], summary: dict[str, Any], *, indent: str
) -> None:
    """Append one runner/model score summary to rendered CLI output."""
    metric_labels = {
        "score_avg": "score_avg",
        "score_min": "score_min",
        "score_max": "score_max",
        "partial_match_accuracy_avg": "partial_match_avg",
        "partial_exact_match_accuracy_avg": "partial_exact_match_avg",
        "strict_exact_match_accuracy_avg": "strict_exact_match_avg",
        "partial_similarity_score_avg": "partial_similarity_avg",
        "partial_numeric_match_accuracy_avg": "partial_numeric_match_avg",
        "range_match_accuracy_avg": "range_match_avg",
        "delta_1_match_accuracy_avg": "delta_1_match_avg",
        "precision_avg": "precision_avg",
        "recall_avg": "recall_avg",
        "f1_avg": "f1_avg",
        "total_gold_keys_avg": "total_gold_keys_avg",
        "total_model_keys_avg": "total_model_keys_avg",
        "matched_keys_avg": "matched_keys_avg",
        "exact_value_matches_avg": "exact_value_matches_avg",
    }
    for metric, label in metric_labels.items():
        value = summary.get(metric)
        if value is not None:
            lines.append(f"{indent}{label}: {value:.4f}")

    lines.append(f"{indent}missing_keys_total: {summary.get('missing_keys_total', 0)}")
    lines.append(f"{indent}extra_keys_total: {summary.get('extra_keys_total', 0)}")
    lines.append(
        f"{indent}detail_entries_total: {summary.get('detail_entries_total', 0)}"
    )


def report_to_json(report: EvalReport) -> str:
    """Convenience JSON dump that round-trips through pydantic."""
    return json.dumps(json.loads(report.model_dump_json()), indent=2)
