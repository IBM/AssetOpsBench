#!/usr/bin/env python
"""One model-level results table from a sweep's `_aggregate.json` files.

    uv run python benchmarks/consolidate_results.py "$LEADERBOARD_DIR"
    uv run python benchmarks/consolidate_results.py path/to/_aggregate.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_runs(target: Path) -> list[dict[str, Any]]:
    """Flatten every scored scenario from every report under `target`."""
    paths = [target] if target.is_file() else sorted(target.rglob("_aggregate.json"))
    runs: list[dict[str, Any]] = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            continue
        for result in report.get("results") or []:
            if not isinstance(result, dict):
                continue
            score = result.get("score") or {}
            ops = result.get("ops") or {}
            runs.append(
                {
                    "model": result.get("model") or "unknown",
                    "scenario_id": str(result.get("scenario_id") or ""),
                    "passed": bool(score.get("passed")),
                    "score": score.get("score"),
                    "turns": ops.get("turn_count") or 0,
                    "tool_calls": ops.get("tool_call_count") or 0,
                    "tokens_in": ops.get("tokens_in") or 0,
                    "tokens_out": ops.get("tokens_out") or 0,
                    "duration_ms": ops.get("duration_ms"),
                    "cost_usd": ops.get("est_cost_usd"),
                }
            )
    return runs


def _id_sort_key(scenario_id: str) -> tuple[int, Any]:
    """Sort numeric scenario ids numerically, everything else lexically after."""
    return (0, int(scenario_id)) if scenario_id.isdigit() else (1, scenario_id)


def stdev(values: list[float]) -> float | None:
    """Sample standard deviation, or None when fewer than two values."""
    return statistics.stdev(values) if len(values) > 1 else None


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)

    rows = []
    for model, items in by_model.items():
        n = len(items)
        passed = sum(1 for r in items if r["passed"])
        scores = [r["score"] for r in items if isinstance(r["score"], (int, float))]
        costs = [r["cost_usd"] for r in items if isinstance(r["cost_usd"], (int, float))]
        # duration_ms is optional per scenario, so average only what was recorded.
        secs = [
            r["duration_ms"] / 1000
            for r in items
            if isinstance(r["duration_ms"], (int, float))
        ]
        turns = [r["turns"] for r in items]
        calls = [r["tool_calls"] for r in items]
        tokens_in = [r["tokens_in"] for r in items]
        tokens_out = [r["tokens_out"] for r in items]
        rows.append(
            {
                "model": model,
                "n": n,
                "passed": passed,
                "pass_rate": passed / n if n else 0.0,
                "score_avg": statistics.fmean(scores) if scores else None,
                "score_std": stdev(scores),
                "turns_avg": statistics.fmean(turns),
                "turns_std": stdev(turns),
                "calls_avg": statistics.fmean(calls),
                "calls_std": stdev(calls),
                "tokens_in_avg": statistics.fmean(tokens_in),
                "tokens_in_std": stdev(tokens_in),
                "tokens_out_avg": statistics.fmean(tokens_out),
                "tokens_out_std": stdev(tokens_out),
                "secs_avg": statistics.fmean(secs) if secs else None,
                "secs_std": stdev(secs),
                "secs_median": statistics.median(secs) if secs else None,
                "secs_total": sum(secs) if secs else None,
                "cost_total": sum(costs) if costs else None,
                "cost_per_pass": (sum(costs) / passed) if costs and passed else None,
            }
        )
    return sorted(rows, key=lambda r: -r["pass_rate"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="LEADERBOARD_DIR or an _aggregate.json")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include models that did not complete the full scenario count.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    root = args.target
    if root.is_dir() and (root / "assetopsbench-reports").is_dir():
        root = root / "assetopsbench-reports"

    runs = load_runs(root)
    if not runs:
        print(f"error: no scored results under {root}", file=sys.stderr)
        return 2

    counts = Counter(r["model"] for r in runs)
    full = max(counts.values())
    incomplete = {m: c for m, c in counts.items() if c < full}

    # Scenario ids a complete model covered, so a dropped model can be resumed.
    expected_ids = {
        r["scenario_id"] for r in runs if counts[r["model"]] == full and r["scenario_id"]
    }
    ids_by_model: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        if run["scenario_id"]:
            ids_by_model[run["model"]].add(run["scenario_id"])

    if incomplete and not args.all_models:
        runs = [r for r in runs if counts[r["model"]] == full]

    rows = summarize(runs)

    def mean_sd(avg: float | None, sd: float | None, spec: str) -> str:
        """Render 'mean±sd', dropping the spread when a single run gives none."""
        if avg is None:
            return "-"
        if sd is None:
            return format(avg, spec)
        return f"{format(avg, spec)}±{format(sd, spec)}"

    header = (
        f"{'model':<34}{'n':>4}{'pass':>5}{'rate':>7}{'score':>13}"
        f"{'turns':>12}{'calls':>12}{'secs':>15}{'tok_in':>17}{'cost$':>8}{'$/pass':>8}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for r in rows:
        cost = f"{r['cost_total']:,.2f}" if r["cost_total"] is not None else "-"
        per_pass = f"{r['cost_per_pass']:,.2f}" if r["cost_per_pass"] is not None else "-"
        print(
            f"{r['model'][:33]:<34}{r['n']:>4}{r['passed']:>5}{r['pass_rate']:>6.1%}"
            f"{mean_sd(r['score_avg'], r['score_std'], '.3f'):>13}"
            f"{mean_sd(r['turns_avg'], r['turns_std'], '.1f'):>12}"
            f"{mean_sd(r['calls_avg'], r['calls_std'], '.1f'):>12}"
            f"{mean_sd(r['secs_avg'], r['secs_std'], ',.1f'):>15}"
            f"{mean_sd(r['tokens_in_avg'], r['tokens_in_std'], ',.0f'):>17}"
            f"{cost:>8}{per_pass:>8}"
        )

    total = sum(r["n"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    print(f"\n{passed}/{total} passed ({passed / total:.1%}) across {len(rows)} models")

    wall = [r["secs_total"] for r in rows if r["secs_total"] is not None]
    if wall:
        slowest = max(rows, key=lambda r: r["secs_total"] or 0)
        print(
            f"agent time {sum(wall) / 60:,.1f} min total, "
            f"slowest model {slowest['model']} at {slowest['secs_total'] / 60:,.1f} min"
        )
    print(
        "mean±sd; sd is the sample standard deviation across scenarios, so it "
        "measures task-to-task spread, not run-to-run variance."
    )

    if incomplete:
        verb = "incomplete" if args.all_models else "dropped"
        print(f"\n{verb}: {len(incomplete)} model(s) short of the full {full} scenarios")
        width = max(len(m) for m in incomplete)
        for model, count in sorted(incomplete.items(), key=lambda kv: (-kv[1], kv[0])):
            missing = sorted(expected_ids - ids_by_model[model], key=_id_sort_key)
            shown = ", ".join(missing[:8])
            if len(missing) > 8:
                shown += f", +{len(missing) - 8} more"
            detail = f"  missing {shown}" if shown else ""
            print(f"  {model:<{width}}  {count:>4}/{full}{detail}")
        if not args.all_models:
            print("  pass --all-models to include them in the table above")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())