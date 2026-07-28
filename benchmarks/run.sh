#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: benchmarks/run.sh SCENARIO_ROOT [SCENARIO_RUNNER_ARGS...]" >&2
  echo "       SCENARIO_ROOT=/path/to/scenarios_data benchmarks/run.sh" >&2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/.." && pwd -P)"

if (( $# > 0 )); then
  scenario_root="$1"
  shift
else
  scenario_root="${SCENARIO_ROOT:-}"
fi

if [[ -z "$scenario_root" ]]; then
  usage
  exit 2
fi

if [[ ! -d "$scenario_root" ]]; then
  echo "error: scenario root does not exist: $scenario_root" >&2
  exit 2
fi

scenario_root="$(cd -- "$scenario_root" && pwd -P)"
reasoning_effort="${OPENAI_REASONING_EFFORT:-medium}"

models=(
  "tokenrouter/openai/gpt-5.6-sol"
  "tokenrouter/anthropic/claude-opus-4.8"
  "tokenrouter/MiniMax-M3"
  "tokenrouter/google/gemini-3.6-flash"
  "tokenrouter/z-ai/glm-5.2"
)

cd -- "$repo_root"

for model in "${models[@]}"; do
  printf '\nRunning Lite suite with %s\n' "$model"

  uv run python -m benchmark.scenario_suite_runner \
    --scenario-ids lite \
    --scenario-root "$scenario_root" \
    --agent_name openai_agent \
    --model-id "$model" \
    --trajectory-root /tmp/leaderboard/assetopsbench-trajectories \
    --reports-root /tmp/leaderboard/assetopsbench-reports \
    --openai-workspace-root /tmp/leaderboard/assetopsbench-workspaces \
    --openai-allow-files \
    --openai-allow-bash \
    --openai-allow-edit \
    --openai-reasoning-effort "$reasoning_effort" \
    --openai-reasoning-summary auto \
    --skip-existing \
    --continue-on-error \
    "$@"
done
