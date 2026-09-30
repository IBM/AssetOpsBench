"""Map an AssetOpsBench EvalReport onto Harbor's reward.json.

Harbor averages each key across trials, so only scores belong here; token
counts and cost reach Harbor through the ATIF trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-status", type=int, default=0)
    args = parser.parse_args()

    rewards: dict[str, float | int] = {"reward": 0.0, "passed": 0}

    if not args.report.exists():
        print(
            f"evaluation produced no report at {args.report} "
            f"(exit status {args.eval_status}); scoring 0",
            file=sys.stderr,
        )
    else:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        results = report.get("results") or []
        if not results:
            print(
                "evaluation report contains no scored results; scoring 0",
                file=sys.stderr,
            )
        else:
            # One task is one scenario, so there is exactly one result.
            score = results[0].get("score") or {}
            rewards["reward"] = float(score.get("score") or 0.0)
            rewards["passed"] = int(bool(score.get("passed")))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rewards), encoding="utf-8")
    print(json.dumps(rewards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
