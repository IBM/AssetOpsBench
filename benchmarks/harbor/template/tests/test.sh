#!/usr/bin/env bash
# Harbor verifier for an AssetOpsBench scenario.
#
# Runs in the `main` container (verifier environment_mode defaults to
# "shared"), so the repo, the uv environment and the evaluation package are
# already present and this needs no image of its own.
#
# Harbor uploads tests/ to /tests and reads /logs/verifier/reward.json
# afterwards, preferring it over reward.txt.
#
# No `set -e` and no `${VAR:?}` anywhere in this file, deliberately. A missing
# reward file raises RewardFileNotFoundError, which Harbor classes as a harness
# failure rather than a score of zero AND puts on its non-retryable list. Every
# exit path below therefore has to reach to_reward.py.
set -uo pipefail

LOG_DIR=/logs/verifier
mkdir -p "$LOG_DIR" "$LOG_DIR/reports"

cd /opt/aob || {
    echo "FATAL: /opt/aob missing; the task image is not the AssetOpsBench runtime" \
        >"$LOG_DIR/test-stderr.txt"
    echo '{"reward": 0.0, "passed": 0}' >"$LOG_DIR/reward.json"
    exit 0
}

# --scorer-default must name a scorer that is actually registered. The CLI's
# own default is llm_judge, but evaluation.cli only registers that scorer inside
# _maybe_install_judge(), which no-ops without --judge-model. It then calls
# _validate_scorer_default() and exits:
#
#   unknown scorer 'llm_judge'; registered: ['fmea', 'static_json']
#
# So name static_json explicitly. This is only the fallback: the scenario's own
# scoring_method, which evaluation.loader sets from scenario_meta.json, still
# wins in evaluator._score_one. A scenario that asks for llm_judge needs
# AOB_JUDGE_MODEL set, which registers that scorer.
judge_args=(--scorer-default static_json)
if [ -n "${AOB_JUDGE_MODEL:-}" ]; then
    judge_args+=(--judge-model "${AOB_JUDGE_MODEL}")
fi

# /logs/agent is Harbor's agent log directory, not a trajectory store: by the
# time the verifier runs it also holds trajectory.json (the ATIF file the agent
# writes from populate_context_post_run, which Harbor itself reads back) and
# whatever else the agent left there, including subdirectories. Pointing the
# evaluator at it directly means evaluation.loader.load_trajectories rglobs
# every *.json underneath and logs a pydantic traceback for each one that is
# not a PersistedTrajectory. Scoring survives that - the loader catches and
# skips - but a 30-line ValidationError in test-stderr.txt on every passing
# trial is exactly the noise that sends someone debugging the wrong thing.
#
# So hand the evaluator only the run records: top level, *.json, minus the
# files Harbor and the agent own. An empty directory here scores 0 through
# to_reward.py, the same as no record at all.
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
