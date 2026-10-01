#!/usr/bin/env bash
# Harbor verifier for an AssetOpsBench scenario. Runs in `main`, where the repo
# and its uv environment already are; Harbor uploads tests/ to /tests and reads
# /logs/verifier/reward.json.
#
# No `set -e` and no `${VAR:?}`: a missing reward file is a non-retryable
# harness failure rather than a zero, so every exit path must reach
# to_reward.py.
set -uo pipefail

LOG_DIR=/logs/verifier
mkdir -p "$LOG_DIR" "$LOG_DIR/reports"

cd /opt/aob || {
    echo "FATAL: /opt/aob missing; the task image is not the AssetOpsBench runtime" \
        >"$LOG_DIR/test-stderr.txt"
    echo '{"reward": 0.0, "passed": 0}' >"$LOG_DIR/reward.json"
    exit 0
}

# The CLI's default scorer, llm_judge, exists only with --judge-model, so name
# static_json as the fallback. The scenario's own scoring_method still wins.
judge_args=(--scorer-default static_json)
if [ -n "${AOB_JUDGE_MODEL:-}" ]; then
    judge_args+=(--judge-model "${AOB_JUDGE_MODEL}")
fi

# /logs/agent also holds Harbor's trajectory.json and other agent files, which
# the evaluator would try to parse as run records and log a traceback for. Hand
# it only the top-level run records. None at all scores 0.
TRAJ_DIR="$LOG_DIR/trajectories"
mkdir -p "$TRAJ_DIR"
for candidate in /logs/agent/*.json; do
    [ -f "$candidate" ] || continue
    case "$(basename "$candidate")" in
        trajectory.json) continue ;;
    esac
    cp "$candidate" "$TRAJ_DIR/"
done

uv run evaluate \
  --trajectories "$TRAJ_DIR" \
  --scenarios /tests/scenarios \
  --reports-dir "$LOG_DIR/reports" \
  "${judge_args[@]}" \
  >"$LOG_DIR/test-stdout.txt" 2>"$LOG_DIR/test-stderr.txt"
status=$?

uv run python /tests/to_reward.py \
  --report "$LOG_DIR/reports/_aggregate.json" \
  --out "$LOG_DIR/reward.json" \
  --eval-status "$status"
