#!/usr/bin/env bash
# Oracle. Writes the ground-truth answer in the shape observability.persistence
# writes, so `--agent oracle` should score 1.0 on every scenario.
#
# Nothing in this design requires an oracle, but Harbor's adapter guidelines
# treat a 100% oracle pass as the gate that proves a task is scorable at all,
# and for an LLM-judge benchmark it doubles as a judge sanity check: a scenario
# where the oracle does not score 1.0 has a scorer or ground-truth problem, not
# an agent problem.
set -euo pipefail

mkdir -p /logs/agent
python3 - <<'PY'
import json
import pathlib

answer = pathlib.Path("/solution/groundtruth.txt").read_text(encoding="utf-8").strip()
question = pathlib.Path("/solution/question.txt").read_text(encoding="utf-8").strip()

record = {
    "run_id": "oracle_1",
    "scenario_id": "1",
    "runner": "oracle",
    "model": "oracle",
    "question": question,
    "answer": answer,
    "trajectory": None,
}
pathlib.Path("/logs/agent/oracle_1.json").write_text(
    json.dumps(record, indent=2), encoding="utf-8"
)
PY
