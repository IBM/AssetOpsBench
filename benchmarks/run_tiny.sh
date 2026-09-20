#!/usr/bin/env bash

set -euo pipefail

scenario_dir="${1:-${SCENARIO_DIR:-}}"
leaderboard_dir="${2:-${LEADERBOARD_DIR:-}}"

if [[ -z "$scenario_dir" || -z "$leaderboard_dir" ]]; then
  printf 'Usage: %s SCENARIO_DIR LEADERBOARD_DIR\n' "$0" >&2
  printf '       or set SCENARIO_DIR and LEADERBOARD_DIR\n' >&2
  exit 2
fi

agent_name=stirrup_agent
scenario_ids=open

model_configs=(
  "litellm_proxy/aws/claude-opus-5 high"
)

for model_config in "${model_configs[@]}"; do
  read -r model_id reasoning_effort <<< "$model_config"

  echo "Running $model_id with reasoning effort $reasoning_effort"

  uv run python -m benchmark.scenario_suite_runner \
    --scenario-ids "$scenario_ids" \
    --scenario-root "$scenario_dir" \
    --agent_name "$agent_name" \
    --model-id "$model_id" \
    --reasoning-effort "$reasoning_effort" \
    --trajectory-root "$leaderboard_dir/assetopsbench-trajectories" \
    --reports-root "$leaderboard_dir/assetopsbench-reports" \
    --stirrup-workspace-root "$leaderboard_dir/assetopsbench-stirrup-workspaces" \
    --preserve-workspaces \
    --continue-on-error
done
