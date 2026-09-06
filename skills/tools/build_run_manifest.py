#!/usr/bin/env python3
"""Build a Gate 5 run manifest from AssetOpsBench evaluation output.

`gate5_counterfactual.py` wants one JSONL line per recorded run. The benchmark
already writes everything it needs, in two places, so nothing has to be
instrumented: `_aggregate.json` under the reports root carries the score and the
operational metrics, and the trajectory JSON carries the record of what the
agent actually opened. This joins them.

    # one arm at a time, appending to the same manifest
    python skills/tools/build_run_manifest.py --k-level k0 \\
        --reports-root  runs/k0/assetopsbench-reports \\
        --trajectory-root runs/k0/assetopsbench-trajectories \\
        --out runs/manifest.jsonl

    python skills/tools/build_run_manifest.py --k-level k1 \\
        --reports-root  runs/k1/assetopsbench-reports \\
        --trajectory-root runs/k1/assetopsbench-trajectories \\
        --out runs/manifest.jsonl --append

    python skills/tools/gate5_counterfactual.py --runs runs/manifest.jsonl --per-graph

Both roots are the ones passed to `benchmark.scenario_suite_runner` as
`--reports-root` and `--trajectory-root`, and both nest as
`<root>/<agent_name>/<model-slug>/`. Every arm must be written to a **separate**
pair of roots, because the file names carry the scenario id and nothing else: run
k0 and k1 into the same directory and the second overwrites the first.

The K level is supplied here rather than read from the output, because nothing
in the benchmark's own records it. That is the one place this join can go wrong
and it is worth an explicit check: pass `--expect-skills` on a k1 arm and
`--expect-no-skills` on k0, and the builder will fail if the trajectories
disagree with the label. A mislabelled arm produces a clean-looking manifest and
a meaningless result, so it is worth ten seconds.

Exit codes: 0 written, 1 a check failed or nothing was found, 2 bad invocation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

AGGREGATE = "_aggregate.json"
# A path into the mounted collection, as it appears in a code-exec argument.
CONSULT_RE = re.compile(r"repo-skills(?:-router)?/")


def find_aggregates(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(root.rglob(AGGREGATE))


def load_results(path: pathlib.Path) -> list[dict]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: {type(exc).__name__}: {exc}")
    results = doc.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"{path}: no `results` list; is this an EvalReport?")
    return results


def trajectory_for(traj_root: pathlib.Path, runner: str,
                   scenario_id: str) -> pathlib.Path | None:
    """The suite runner writes `<agent>_<scenario>.json` under `<agent>/<model>/`.

    Matched on the file name rather than on the model directory, so a manifest
    can still be built when the reports and the trajectories were written under
    slightly different model slugs.
    """
    exact = list(traj_root.rglob(f"{runner}_{scenario_id}.json"))
    if exact:
        return exact[0]
    loose = list(traj_root.rglob(f"*_{scenario_id}.json"))
    return loose[0] if len(loose) == 1 else None


def consulted(path: pathlib.Path | None) -> bool:
    if path is None or not path.exists():
        return False
    return bool(CONSULT_RE.search(path.read_text(encoding="utf-8", errors="replace")))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports-root", type=pathlib.Path, required=True)
    ap.add_argument("--trajectory-root", type=pathlib.Path, required=True)
    ap.add_argument("--k-level", required=True, choices=("k0", "k1", "k1-recovery"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of replacing it")
    ap.add_argument("--repetition", type=int, default=0,
                    help="repetition index, when the same arm is run more than once")
    ap.add_argument("--asset-class-map", type=pathlib.Path,
                    help="optional JSON mapping scenario_id to asset class, which "
                         "enables the per-asset-class breakdown in gate 5")
    ap.add_argument("--expect-skills", action="store_true",
                    help="fail if any trajectory does NOT reference the collection")
    ap.add_argument("--expect-no-skills", action="store_true",
                    help="fail if ANY trajectory references the collection; use on k0")
    a = ap.parse_args()

    if a.expect_skills and a.expect_no_skills:
        ap.error("--expect-skills and --expect-no-skills are mutually exclusive")
    for p in (a.reports_root, a.trajectory_root):
        if not p.is_dir():
            print(f"not a directory: {p}", file=sys.stderr)
            return 2

    classes = {}
    if a.asset_class_map:
        classes = json.loads(a.asset_class_map.read_text(encoding="utf-8"))

    aggregates = find_aggregates(a.reports_root)
    if not aggregates:
        print(f"no {AGGREGATE} under {a.reports_root}", file=sys.stderr)
        return 1

    lines, missing_traj, with_skills, without_skills = [], [], [], []
    for agg in aggregates:
        for r in load_results(agg):
            sid = str(r.get("scenario_id", "")).strip()
            if not sid:
                continue
            runner = str(r.get("runner", "")).strip() or "stirrup_agent"
            score = r.get("score") or {}
            ops = r.get("ops") or {}
            traj = trajectory_for(a.trajectory_root, runner, sid)
            if traj is None:
                missing_traj.append(sid)
            (with_skills if consulted(traj) else without_skills).append(sid)

            rec = {
                "task_id": sid,
                "k_level": a.k_level,
                "score": float(score.get("score", 0.0)),
                "repetition": a.repetition,
                "passed": bool(score.get("passed", False)),
                "model": r.get("model", ""),
                "tokens": int(ops.get("tokens_in", 0)) + int(ops.get("tokens_out", 0)),
                "steps": int(ops.get("turn_count", 0)),
                "tool_calls": int(ops.get("tool_call_count", 0)),
            }
            if sid in classes:
                rec["asset_class"] = classes[sid]
            if traj is not None:
                rec["trajectory"] = str(traj.resolve())
            lines.append(json.dumps(rec))

    mode = "a" if a.append else "w"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open(mode, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"{len(lines)} run(s) written to {a.out} "
          f"({'appended' if a.append else 'replaced'})")
    print(f"  arm                {a.k_level}, repetition {a.repetition}")
    print(f"  aggregates read    {len(aggregates)}")
    print(f"  consulted skills   {len(with_skills)}")
    print(f"  consulted nothing  {len(without_skills)}")
    if missing_traj:
        print(f"  WARN no trajectory for {len(missing_traj)} run(s), so those rows "
              f"cannot carry per-graph attribution: {missing_traj[:8]}")

    failed = False
    if a.expect_no_skills and with_skills:
        print(f"\nFAIL labelled {a.k_level} but {len(with_skills)} trajectory(ies) "
              f"reference the skill collection: {with_skills[:8]}")
        print("     the baseline is contaminated, or the arms were mislabelled")
        failed = True
    if a.expect_skills and without_skills:
        print(f"\nFAIL labelled {a.k_level} but {len(without_skills)} trajectory(ies) "
              f"reference no skill at all: {without_skills[:8]}")
        print("     the mount did not reach the agent, or it chose never to look. "
              "Check one workspace for a skills/ directory before trusting the arm")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
