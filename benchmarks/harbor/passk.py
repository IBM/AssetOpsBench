#!/usr/bin/env python3
"""Consistency metrics for a Harbor job run with -k repeats.

Harbor's own pass_at_k is unusable for this suite: harbor/utils/pass_at_k.py
returns {} unless reward.json holds exactly one key whose value is 0 or 1, and
its eligible k values are 2, 4, 8, ... and 5, 10, ..., so k=3 is never computed.
This reads the trial results directly instead.

    python3 benchmarks/harbor/passk.py <harbor-jobs>[/<job>]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

KEYS = (("passed", "mode"),)


def is_job(path: Path) -> bool:
    """A job holds trial directories; a trial holds result.json beside verifier/."""
    return any(
        (child / "result.json").exists() and (child / "verifier").is_dir()
        for child in path.iterdir()
        if child.is_dir()
    )


def iter_jobs(root: Path):
    if is_job(root):
        yield root
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and is_job(child):
            yield child


def collect(job: Path) -> dict[str, list[dict]]:
    """task name -> one entry per attempt."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for trial in sorted(job.glob("*/result.json")):
        result = json.loads(trial.read_text())
        task = result.get("task_name") or trial.parent.name.split("__")[0]
        rewards = ((result.get("verifier_result") or {}).get("rewards")) or {}
        by_task[task].append(
            {
                "reward": rewards.get("reward"),
                "passed": bool(rewards.get("passed")),
                "errored": bool(result.get("exception_info")),
            }
        )
    return by_task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--detail", action="store_true", help="per-scenario pass pattern")
    args = ap.parse_args()

    for job in iter_jobs(args.path):
        by_task = collect(job)
        if not by_task:
            continue
        counts = {len(v) for v in by_task.values()}
        k = min(counts)
        print(f"\n=== {job.name}")
        print(f"    tasks {len(by_task)}   attempts per task {sorted(counts)}"
              f"{'   RAGGED, using k=' + str(k) if len(counts) > 1 else ''}")

        errored = sum(a["errored"] for v in by_task.values() for a in v)
        if errored:
            print(f"    {errored} attempt(s) errored and count as a failure")

        rewards = [a["reward"] for v in by_task.values() for a in v if isinstance(a["reward"], (int, float))]
        if rewards:
            print(f"    mean reward {st.mean(rewards):.3f}")

        for field, label in KEYS:
            per_task = [[a[field] for a in v[:k]] for v in by_task.values()]
            any_pass = sum(any(p) for p in per_task) / len(per_task)
            all_pass = sum(all(p) for p in per_task) / len(per_task)
            per_attempt = [
                sum(p[i] for p in per_task) / len(per_task) for i in range(k)
            ]
            spread = (f"  per-run {', '.join(f'{x:.3f}' for x in per_attempt)}"
                      f"  mean {st.mean(per_attempt):.3f}"
                      + (f"  sd {st.stdev(per_attempt):.3f}" if k > 1 else ""))
            print(f"    {label:<7} pass@{k} {any_pass:.3f}   pass^{k} {all_pass:.3f}{spread}")

        if args.detail:
            print(f"    {'task':<22}{'pattern':<10}reward per attempt")
            for task, attempts in sorted(by_task.items()):
                pattern = "".join("Y" if a["passed"] else "." for a in attempts)
                vals = " ".join(
                    f"{a['reward']:.3f}" if isinstance(a["reward"], (int, float)) else "  -  "
                    for a in attempts
                )
                print(f"    {task.split('/')[-1]:<22}{pattern:<10}{vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
