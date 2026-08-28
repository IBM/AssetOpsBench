#!/usr/bin/env python
"""One model-level results table from a sweep's `_aggregate.json` files.

    uv run python scripts/consolidate_results.py "$LEADERBOARD_DIR"
    uv run python scripts/consolidate_results.py path/to/_aggregate.json --csv out.csv
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
                    "passed": bool(score.get("passed")),
                    "score": score.get("score"),
                    "turns": ops.get("turn_count") or 0,
                    "tool_calls": ops.get("tool_call_count") or 0,
                    "tokens_in": ops.get("tokens_in") or 0,
                    "tokens_out": ops.get("tokens_out") or 0,
                    "cost_usd": ops.get("est_cost_usd"),
                }
            )
    return runs


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
        rows.append(
            {
                "model": model,
                "n": n,
                "passed": passed,
                "pass_rate": passed / n if n else 0.0,
                "score_avg": statistics.fmean(scores) if scores else None,
                "turns_avg": statistics.fmean(r["turns"] for r in items),
                "calls_avg": statistics.fmean(r["tool_calls"] for r in items),
                "tokens_in_avg": statistics.fmean(r["tokens_in"] for r in items),
                "tokens_out_avg": statistics.fmean(r["tokens_out"] for r in items),
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

    if not args.all_models:
        counts = Counter(r["model"] for r in runs)
        full = max(counts.values())
        incomplete = {m: c for m, c in counts.items() if c < full}
        if incomplete:
            print(
                f"showing models with all {full} scenarios; dropped "
                + ", ".join(f"{m} (n={c})" for m, c in sorted(incomplete.items())),
                file=sys.stderr,
            )
            runs = [r for r in runs if counts[r["model"]] == full]

    rows = summarize(runs)

    header = (
        f"{'model':<38}{'n':>5}{'pass':>6}{'rate':>8}{'score':>8}"
        f"{'turns':>7}{'calls':>7}{'tok_in':>11}{'cost$':>9}{'$/pass':>9}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for r in rows:
        score = f"{r['score_avg']:.3f}" if r["score_avg"] is not None else "-"
        cost = f"{r['cost_total']:,.2f}" if r["cost_total"] is not None else "-"
        per_pass = f"{r['cost_per_pass']:,.2f}" if r["cost_per_pass"] is not None else "-"
        print(
            f"{r['model'][:37]:<38}{r['n']:>5}{r['passed']:>6}{r['pass_rate']:>7.1%}"
            f"{score:>8}{r['turns_avg']:>7.1f}{r['calls_avg']:>7.1f}"
            f"{r['tokens_in_avg']:>11,.0f}{cost:>9}{per_pass:>9}"
        )

    total = sum(r["n"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    print(f"\n{passed}/{total} passed ({passed / total:.1%}) across {len(rows)} models")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())