#!/usr/bin/env bash

set -euo pipefail

scenario_dir="${1:-${SCENARIO_DIR:-}}"
leaderboard_dir="${2:-${LEADERBOARD_DIR:-}}"

if [[ -z "$scenario_dir" || -z "$leaderboard_dir" ]]; then
  printf 'Usage: %s SCENARIO_DIR LEADERBOARD_DIR\n' "$0" >&2
  printf '       or set SCENARIO_DIR and LEADERBOARD_DIR\n' >&2
  printf '\n' >&2
  printf 'Environment:\n' >&2
  printf '  AGENTS  space-separated methods to run (default: stirrup_agent).\n' >&2
  printf '          AGENTS="stirrup_agent stirrup_agent_gateway" runs the flat\n' >&2
  printf '          and gateway tool surfaces back to back;\n' >&2
  printf '          stirrup_agent_gateway_search is the search-only variant.\n' >&2
  exit 2
fi

# Separate method names on purpose: trajectories, reports and workspaces are
# nested by agent name, so running two topologies under one name would have the
# second overwrite the first, which is precisely the comparison being made.
agent_names="${AGENTS:-stirrup_agent}"
scenario_ids=open

model_configs=(
  "litellm_proxy/aws/claude-opus-5 high"
)

for model_config in "${model_configs[@]}"; do
  read -r model_id reasoning_effort <<< "$model_config"

  for agent_name in $agent_names; do
    echo "Running $agent_name / $model_id with reasoning effort $reasoning_effort"

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
done
