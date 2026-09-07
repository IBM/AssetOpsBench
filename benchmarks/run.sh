#!/usr/bin/env bash

set -euo pipefail

usage() {
  printf 'Usage: %s -s SCENARIO_DIR -l LEADERBOARD_DIR -t TRAJECTORY_DIR\n' "$0" >&2
  printf '       %s SCENARIO_DIR LEADERBOARD_DIR TRAJECTORY_DIR\n' "$0" >&2
  printf '       or set SCENARIO_DIR, LEADERBOARD_DIR, and TRAJECTORY_DIR\n' >&2
}

scenario_dir="${SCENARIO_DIR:-}"
leaderboard_dir="${LEADERBOARD_DIR:-}"
trajectory_dir="${TRAJECTORY_DIR:-}"

while getopts ':s:l:t:' option; do
  case "$option" in
    s) scenario_dir="$OPTARG" ;;
    l) leaderboard_dir="$OPTARG" ;;
    t) trajectory_dir="$OPTARG" ;;
    :) printf 'Option -%s requires an argument.\n' "$OPTARG" >&2; usage; exit 2 ;;
    \?) printf 'Unknown option: -%s\n' "$OPTARG" >&2; usage; exit 2 ;;
  esac
done

option_arg_count=$((OPTIND - 1))
shift "$option_arg_count"

if (( option_arg_count > 0 && $# > 0 )); then
  printf 'Options and positional arguments cannot be combined.\n' >&2
  usage
  exit 2
fi

if (( option_arg_count == 0 )); then
  if (( $# > 3 )); then
    usage
    exit 2
  fi
  scenario_dir="${1:-$scenario_dir}"
  leaderboard_dir="${2:-$leaderboard_dir}"
  trajectory_dir="${3:-$trajectory_dir}"
fi

if [[ -z "$scenario_dir" || -z "$leaderboard_dir" || -z "$trajectory_dir" ]]; then
  usage
  exit 2
fi

agent_name=stirrup_agent
scenario_ids=tsfm_lite

model_configs=(
  "litellm_proxy/gcp/gemini-3.6-flash high"
  "litellm_proxy/azure/gpt-5.6-sol max"
  "litellm_proxy/aws/claude-opus-5 high"
  "litellm_proxy/aws/claude-sonnet-5 max"
  "tokenrouter/MiniMax-M3 high"
  "tokenrouter/moonshotai/kimi-k3 max"
  "tokenrouter/z-ai/glm-5.3 max"
  "tokenrouter/deepseek/deepseek-v4-flash max"
  ""

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
    --trajectory-root "$trajectory_dir" \
    --reports-root "$leaderboard_dir/assetopsbench-reports" \
    --stirrup-workspace-root "$leaderboard_dir/assetopsbench-stirrup-workspaces" \
    --skip-existing \
    --continue-on-error \
    --preserve-workspaces

done
