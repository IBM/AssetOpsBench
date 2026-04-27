"""Build an :class:`EvalReport` from graded scenario results."""

from __future__ import annotations

import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path

from .metrics import aggregate_ops
from .models import EvalReport, ScenarioResult, TypeBreakdown


def build_report(results: list[ScenarioResult]) -> EvalReport:
    total = len(results)
    passed = sum(1 for r in results if r.grade.passed)

    by_type: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_type[r.scenario_type or "unknown"].append(r)

    breakdown: dict[str, TypeBreakdown] = {}
    for stype, items in by_type.items():
        n = len(items)
        p = sum(1 for r in items if r.grade.passed)
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
            "graded": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        },
        by_scenario_type=breakdown,
        ops=aggregate_ops(results),
        results=results,
    )


def write_report(report: EvalReport, output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output


def render_summary(report: EvalReport) -> str:
    lines: list[str] = []
    t = report.totals
    lines.append(
        f"Scenarios: {t.get('scenarios', 0)}  "
        f"Passed: {t.get('passed', 0)}  "
        f"Pass rate: {t.get('pass_rate', 0):.1%}"
    )
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


def report_to_json(report: EvalReport) -> str:
    """Convenience JSON dump that round-trips through pydantic."""
    return json.dumps(json.loads(report.model_dump_json()), indent=2)
