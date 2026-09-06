#!/usr/bin/env python
"""One model-level results table from a sweep's `_aggregate.json` files.

    uv run python benchmarks/consolidate_results.py "$LEADERBOARD_DIR"
    uv run python benchmarks/consolidate_results.py path/to/_aggregate.json --csv out.csv

By default only models that completed the full scenario count are shown, which
drops a model that ran 91 of 97. Three ways to include it instead:

    --denominator max     score every model out of the largest scenario count;
                          a scenario a model never ran counts as not passed
    --denominator 97      the same, with the count stated explicitly
    --common              compare only on the scenarios every model ran
    --all-models          show everyone as-is, each on its own denominator

`--denominator` is the right choice for a fixed benchmark: the suite is N
scenarios and a missing answer is not a pass. `--common` is the right choice
when a sweep was cut short for reasons that have nothing to do with the models.
Pair either with `--min-runs` to exclude runs too partial to be meaningful.
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
                    # kept so runs can be intersected across models
                    "scenario": str(result.get("scenario_id") or ""),
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


def summarize(runs: list[dict[str, Any]],
              denominator: int | None = None) -> list[dict[str, Any]]:
    """One row per model.

    `denominator` scores every model out of the same scenario count, so a model
    that ran 91 of 97 is scored out of 97 and the 6 it never ran count as not
    passed. Averages stay over the runs that exist, because averaging
    consumption over scenarios a model never attempted would understate it.
    """
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)

    rows = []
    for model, items in by_model.items():
        n_runs = len(items)
        denom = denominator or n_runs
        passed = sum(1 for r in items if r["passed"])
        scores = [r["score"] for r in items if isinstance(r["score"], (int, float))]
        costs = [r["cost_usd"] for r in items if isinstance(r["cost_usd"], (int, float))]
        rows.append(
            {
                "model": model,
                # `n` is the denominator the rates use; `n_runs` is what was
                # actually scored. make_charts.py scales token totals by n_runs.
                "n": denom,
                "n_runs": n_runs,
                "missing": max(denom - n_runs, 0),
                "passed": passed,
                "pass_rate": passed / denom if denom else 0.0,
                # a missing scenario scores 0, matching the pass-rate treatment
                "score_avg": (sum(scores) / denom) if scores and denom else None,
                "score_avg_scored": statistics.fmean(scores) if scores else None,
                "turns_avg": statistics.fmean(r["turns"] for r in items),
                "calls_avg": statistics.fmean(r["tool_calls"] for r in items),
                "tokens_in_avg": statistics.fmean(r["tokens_in"] for r in items),
                "tokens_out_avg": statistics.fmean(r["tokens_out"] for r in items),
                "cost_total": sum(costs) if costs else None,
                "cost_per_pass": (sum(costs) / passed) if costs and passed else None,
            }
        )
    return sorted(rows, key=lambda r: -r["pass_rate"])


def print_table(rows: list[dict[str, Any]], *, show_missing: bool) -> None:
    miss = f"{'miss':>6}" if show_missing else ""
    header = (
        f"{'model':<38}{'n':>5}{miss}{'pass':>6}{'rate':>8}{'score':>8}"
        f"{'turns':>7}{'calls':>7}{'tok_in':>11}{'cost$':>9}{'$/pass':>9}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for r in rows:
        score = f"{r['score_avg']:.3f}" if r["score_avg"] is not None else "-"
        cost = f"{r['cost_total']:,.2f}" if r["cost_total"] is not None else "-"
        per_pass = f"{r['cost_per_pass']:,.2f}" if r["cost_per_pass"] is not None else "-"
        gap = f"{r['missing']:>6}" if show_missing else ""
        print(
            f"{r['model'][:37]:<38}{r['n']:>5}{gap}{r['passed']:>6}{r['pass_rate']:>7.1%}"
            f"{score:>8}{r['turns_avg']:>7.1f}{r['calls_avg']:>7.1f}"
            f"{r['tokens_in_avg']:>11,.0f}{cost:>9}{per_pass:>9}"
        )
    total = sum(r["n"] for r in rows)
    passed = sum(r["passed"] for r in rows)
    print(f"\n{passed}/{total} passed ({passed / total:.1%}) across {len(rows)} models")


def write_csv(rows: list[dict[str, Any]], path: Path | None) -> None:
    if not path:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path,
                        help="LEADERBOARD_DIR or an _aggregate.json")
    parser.add_argument(
        "--denominator",
        metavar="N|max",
        default=None,
        help=(
            "Score every model out of the same scenario count. 'max' uses the "
            "largest count in the sweep. Scenarios a model never ran count as "
            "not passed. Implies --all-models."
        ),
    )
    parser.add_argument(
        "--common",
        action="store_true",
        help=(
            "Compare only on the scenarios every model ran. Equal counts alone "
            "do not guarantee the same scenarios, so this is the strict "
            "apples-to-apples comparison."
        ),
    )
    parser.add_argument(
        "--min-runs", type=int, default=0, metavar="N",
        help="Drop models with fewer than N scored scenarios before comparing.",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Include models that did not complete the full scenario count.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if args.denominator and args.common:
        parser.error("--denominator and --common are different comparisons; pick one")

    root = args.target
    if root.is_dir() and (root / "assetopsbench-reports").is_dir():
        root = root / "assetopsbench-reports"

    runs = load_runs(root)
    if not runs:
        print(f"error: no scored results under {root}", file=sys.stderr)
        return 2

    counts = Counter(r["model"] for r in runs)

    if args.min_runs:
        thin = {m: c for m, c in counts.items() if c < args.min_runs}
        if thin:
            print("below --min-runs, dropped "
                  + ", ".join(f"{m} (n={c})" for m, c in sorted(thin.items())),
                  file=sys.stderr)
            runs = [r for r in runs if counts[r["model"]] >= args.min_runs]
            counts = Counter(r["model"] for r in runs)
        if not runs:
            print(f"error: no model has {args.min_runs}+ scenarios", file=sys.stderr)
            return 2

    denominator = None
    if args.denominator:
        if args.denominator == "max":
            denominator = max(counts.values())
        else:
            try:
                denominator = int(args.denominator)
            except ValueError:
                parser.error("--denominator takes an integer or 'max'")
        short = {m: denominator - c for m, c in counts.items() if c < denominator}
        over = {m: c for m, c in counts.items() if c > denominator}
        print(f"scoring every model out of {denominator} scenarios"
              + ("; unrun scenarios count as not passed for "
                 + ", ".join(f"{m} (-{d})" for m, d in sorted(short.items()))
                 if short else ""),
              file=sys.stderr)
        if over:
            print("warning: more runs than the denominator for "
                  + ", ".join(f"{m} (n={c})" for m, c in sorted(over.items()))
                  + " — rates will exceed 100%", file=sys.stderr)

    elif args.common:
        by_model: dict[str, set[str]] = defaultdict(set)
        for r in runs:
            by_model[r["model"]].add(r["scenario"])
        shared = set.intersection(*by_model.values()) if by_model else set()
        if not shared:
            print("error: the models share no scenarios", file=sys.stderr)
            return 2
        trimmed = {m: len(s) - len(shared) for m, s in by_model.items()
                   if len(s) > len(shared)}
        print(f"comparing on the {len(shared)} scenarios every model ran"
              + ("; trimmed " + ", ".join(f"{m} (-{d})"
                                          for m, d in sorted(trimmed.items()))
                 if trimmed else ""),
              file=sys.stderr)
        runs = [r for r in runs if r["scenario"] in shared]

    elif not args.all_models:
        full = max(counts.values())
        incomplete = {m: c for m, c in counts.items() if c < full}
        if incomplete:
            print(
                f"showing models with all {full} scenarios; dropped "
                + ", ".join(f"{m} (n={c})" for m, c in sorted(incomplete.items()))
                + "\nuse --denominator max to score them out of "
                f"{full} instead, or --common to compare on shared scenarios",
                file=sys.stderr,
            )
            runs = [r for r in runs if counts[r["model"]] == full]

    rows = summarize(runs, denominator)
    print_table(rows, show_missing=bool(denominator))
    write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())