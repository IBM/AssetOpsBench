#!/usr/bin/env bash
# Run one scenario (or a file of ids) through the suite runner, taking paths from
# the same environment variables run.sh uses: SCENARIO_DIR, TRAJECTORY_DIR,
# LEADERBOARD_DIR. Output paths mirror run.sh so results land where you expect.
#
#   ./run_one.sh 3009
#   ./run_one.sh my_scenarios.txt
#   AGENT=stirrup_agent MODEL=litellm_proxy/aws/claude-opus-5 ./run_one.sh 3009
#   DRY_RUN=1 ./run_one.sh 3009
#
# Run from the repository root.
set -euo pipefail

target="${1:-}"
[ -n "$target" ] || { echo "usage: $0 <scenario-id | ids-file>" >&2; exit 2; }

: "${SCENARIO_DIR:?export SCENARIO_DIR to your scenarios_data directory}"
trajectory_dir="${TRAJECTORY_DIR:-traces/trajectories/scenario_suite}"
leaderboard_dir="${LEADERBOARD_DIR:-reports}"

agent="${AGENT:-direct_llm}"
model="${MODEL:-tokenrouter/MiniMax-M3}"

# A bare id is not a valid selector, so wrap it in a temporary ids file.
if [ -f "$target" ]; then
  ids_file="$target"
else
  ids_file="$(mktemp)"
  trap 'rm -f "$ids_file"' EXIT
  printf '%s\n' "$target" > "$ids_file"
fi

args=(
  --scenario-ids "$ids_file"
  --scenario-root "$SCENARIO_DIR"
  --agent_name "$agent"
  --model-id "$model"
  --trajectory-root "$trajectory_dir"
  --reports-root "$leaderboard_dir/assetopsbench-reports"
)
[ -n "${REASONING_EFFORT:-}" ] && args+=(--reasoning-effort "$REASONING_EFFORT")
[ "$agent" = "stirrup_agent" ] && args+=(--stirrup-workspace-root "$leaderboard_dir/assetopsbench-stirrup-workspaces" --preserve-workspaces)
[ -n "${DRY_RUN:-}" ] && args+=(--dry-run)

echo "scenario(s): $target   agent: $agent   model: $model"
uv run python -m benchmark.scenario_suite_runner "${args[@]}"
