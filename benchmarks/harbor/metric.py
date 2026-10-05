"""Dataset-level metric: AssetOpsBench category rollups.

Harbor runs this when metric.py is listed in the dataset's [[files]].

  -i/-o        Harbor's own path. Harbor passes rewards only, with no task
               identity, so this reports overall figures only.
  --job-dir    Reads a finished job's <trial>/result.json files, whose
               task_name carries the category (wosr-1, fmsr-12), and reports
               per-category means.

Do not put the category into the reward dict instead: Harbor averages each
reward key over all trials, counting it as zero where it is absent.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CATEGORIES = ("car", "fcc", "fmea", "fmsr", "health", "tsfm", "wosr")


def from_rewards(input_path: Path) -> dict[str, float | int]:
    scores: list[float] = []
    passes: list[int] = []

    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        reward = json.loads(line)
        if reward is None:
            # A trial that produced no reward counts as zero, matching Harbor's
            # own aggregate_reward_dicts behavior.
            scores.append(0.0)
            passes.append(0)
            continue
        scores.append(float(reward.get("reward") or 0.0))
        passes.append(int(reward.get("passed") or 0))

    return {
        "mean": sum(scores) / len(scores) if scores else 0.0,
        "pass_rate": sum(passes) / len(passes) if passes else 0.0,
        "n_trials": len(scores),
    }


def from_job_dir(job_dir: Path) -> dict[str, float | int]:
    by_category: dict[str, list[float]] = defaultdict(list)
    overall: list[float] = []

    for result_path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        score = float(rewards.get("reward") or 0.0)
        overall.append(score)

        task_name = (result.get("task_name") or "").split("/")[-1]
        category = task_name.split("-", 1)[0]
        if category in CATEGORIES:
            by_category[category].append(score)

    out: dict[str, float | int] = {
        "mean": sum(overall) / len(overall) if overall else 0.0,
        "n_trials": len(overall),
    }
    for category, values in sorted(by_category.items()):
        out[f"{category}_mean"] = sum(values) / len(values)
        out[f"{category}_n"] = len(values)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-path", type=Path)
    parser.add_argument("-o", "--output-path", type=Path)
    parser.add_argument("--job-dir", type=Path)
    args = parser.parse_args()

    if args.job_dir is not None:
        print(json.dumps(from_job_dir(args.job_dir), indent=2))
        return 0

    if args.input_path is None or args.output_path is None:
        parser.error("pass -i and -o, or --job-dir")

    args.output_path.write_text(
        json.dumps(from_rewards(args.input_path), indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
