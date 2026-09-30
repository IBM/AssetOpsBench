#!/usr/bin/env bash
# Harbor counterpart of benchmarks/run.sh.
#
# Same scenarios, same stirrup-agent CLI and the same code-execution sandbox,
# but each scenario runs as a Harbor trial with its own Compose project and its
# own CouchDB, so scenarios run concurrently instead of one at a time against a
# shared database.
#
#   bash benchmarks/harbor/run.sh -s SCENARIO_DIR -l LEADERBOARD_DIR \
#     [-n N_CONCURRENT] [-p PROFILE] [-m "MODEL_ID REASONING_EFFORT"]...
#
# Prerequisites: Docker running, `uv sync --extra harbor`, and the runtime image
#
#   docker build -t assetopsbench/runtime:dev \
#     -f benchmarks/harbor/base-image/Dockerfile .
#
# Credentials are read from ENV_FILE (default .env) by `uv run --env-file`, into
# the Harbor process only; StirrupAgent forwards them to the agent phase. They
# never enter an image.
#
# One Harbor job per model, at LEADERBOARD_DIR/harbor-jobs/stirrup_agent__<model>.
# Re-running resumes that job, finishing only the trials it has not completed,
# which is the equivalent of run.sh's --skip-existing.

set -euo pipefail

usage() {
  printf 'Usage: %s -s SCENARIO_DIR -l LEADERBOARD_DIR [-n N_CONCURRENT] [-p PROFILE] [-m "MODEL_ID EFFORT"]...\n' "$0" >&2
}

scenario_dir="${SCENARIO_DIR:-}"
leaderboard_dir="${LEADERBOARD_DIR:-}"
n_concurrent="${N_CONCURRENT:-4}"
profile="${PROFILE:-benchmarks/scenario_suite/all.yaml}"
env_file="${ENV_FILE:-.env}"
model_configs=()

while getopts ':s:l:n:p:m:' option; do
  case "$option" in
    s) scenario_dir="$OPTARG" ;;
    l) leaderboard_dir="$OPTARG" ;;
    n) n_concurrent="$OPTARG" ;;
    p) profile="$OPTARG" ;;
    m) model_configs+=("$OPTARG") ;;
    :) printf 'Option -%s requires an argument.\n' "$OPTARG" >&2; usage; exit 2 ;;
    \?) printf 'Unknown option: -%s\n' "$OPTARG" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$scenario_dir" || -z "$leaderboard_dir" ]]; then
  usage
  exit 2
fi

if [[ -z "${model_configs[*]+set}" ]]; then
  model_configs=(
    "litellm_proxy/gcp/gemini-3.6-flash high"
    "litellm_proxy/azure/gpt-5.6-sol max"
    "litellm_proxy/aws/claude-opus-5 high"
    "litellm_proxy/aws/claude-sonnet-5 max"
    "tokenrouter/MiniMax-M3 high"
    "tokenrouter/moonshotai/kimi-k3 max"
    "tokenrouter/z-ai/glm-5.3 max"
    "tokenrouter/deepseek/deepseek-v4-flash max"
  )
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

scenario_dir="$(cd "$scenario_dir" && pwd)"
mkdir -p "$leaderboard_dir"
leaderboard_dir="$(cd "$leaderboard_dir" && pwd)"
jobs_dir="$leaderboard_dir/harbor-jobs"

if [[ ! -f "$env_file" ]]; then
  printf 'Credentials file not found: %s (set ENV_FILE)\n' "$env_file" >&2
  exit 2
fi

runtime_image=assetopsbench/runtime:dev
code_image=assetops-code:dev
code_tar="${AOB_CODE_TAR:-$HOME/assetops-code.tar}"
dataset_dir=benchmarks/harbor/datasets/assetopsbench-suite

if ! docker image inspect "$runtime_image" >/dev/null 2>&1; then
  printf 'Runtime image %s not found. Build it first:\n' "$runtime_image" >&2
  printf '  docker build -t %s -f benchmarks/harbor/base-image/Dockerfile .\n' "$runtime_image" >&2
  exit 1
fi

# The suite's shared/ data reaches each trial through
# overlays/private-data.yaml, a read-only bind mount of this directory's shared/.
# Compose reads the variable on `harbor run` and again on `harbor jobs resume`.
# A missing shared/ would not fail the mount: Docker creates the host directory,
# and every collection then loads empty without an error.
if [[ ! -d "$scenario_dir/shared" ]]; then
  printf 'No shared/ directory in %s; is -s the suite'"'"'s scenarios_data?\n' "$scenario_dir" >&2
  exit 2
fi
export AOB_PRIVATE_DIR="$scenario_dir"

# The code sandbox image, as a tar each trial's Docker-in-Docker daemon loads
# (benchmarks/harbor/overlays/code-sandbox.yaml).
if [[ ! -s "$code_tar" ]]; then
  echo "Building $code_image and saving it to $code_tar"
  docker build -q -t "$code_image" \
    -f src/agent/stirrup_agent/Dockerfile.code src/agent/stirrup_agent
  docker save "$code_image" -o "$code_tar"
  chmod 644 "$code_tar"
fi
export AOB_CODE_TAR="$code_tar" AOB_CODE_IMAGE="$code_image"

# Regenerate from scratch: the generator overwrites tasks but never removes
# them, so a folder left over from a larger profile would join this run.
rm -rf "$dataset_dir"
uv run python benchmarks/harbor/adapter/generate_tasks.py \
  --scenario-root "$scenario_dir" \
  --profile "$profile" \
  --output-dir "$dataset_dir" \
  --dataset-name assetopsbench/suite \
  --skip-missing \
  --overwrite >/dev/null

# Fail fast when a model's router is unreachable. Otherwise every trial builds
# its containers, loads its data, retries the model for minutes and exits 1,
# which is how a dropped VPN turns into a job of failed trials.
router_reachable() {
  local base_var
  case "$1" in
    litellm_proxy/*) base_var=LITELLM_BASE_URL ;;
    tokenrouter/*) base_var=TOKENROUTER_BASE_URL ;;
    *) return 0 ;;
  esac
  uv run --env-file "$env_file" python - "$base_var" <<'PY'
import os, sys, urllib.error, urllib.request
name = sys.argv[1]
url = os.environ.get(name, "")
if not url:
    sys.exit(f"{name} is not set in the env file")
try:
    urllib.request.urlopen(url, timeout=15)
except urllib.error.HTTPError:
    pass  # the router answered; any HTTP status proves it is reachable
except Exception as exc:
    sys.exit(f"cannot reach {name} ({exc}); check the VPN or network")
PY
}

for model_config in "${model_configs[@]}"; do
  read -r model_id reasoning_effort <<< "$model_config"
  [[ -z "${model_id:-}" ]] && continue

  model_slug="$(printf '%s' "$model_id" | tr -c 'A-Za-z0-9._-' '-' | tr -s '-')"
  job_name="stirrup_agent__${model_slug%-}"
  job_path="$jobs_dir/$job_name"

  if ! router_reachable "$model_id"; then
    echo "Skipping $model_id: its router is unreachable" >&2
    continue
  fi

  echo "Running $model_id with reasoning effort ${reasoning_effort:-default} -> $job_path"

  if [[ -f "$job_path/config.json" ]]; then
    # Drop trials whose agent crashed (e.g. the model was unreachable) so
    # they run again; scored trials are kept, as run.sh's --skip-existing
    # kept scenarios that already had a trajectory.
    uv run --env-file "$env_file" harbor jobs resume -p "$job_path" \
      --filter-error-type NonZeroAgentExitCodeError || true
    continue
  fi

  effort_args=()
  if [[ -n "${reasoning_effort:-}" ]]; then
    effort_args=(--ak "reasoning_effort=$reasoning_effort")
  fi

  # --continue-on-error equivalent: a failed trial is recorded in the job and
  # the loop moves on to the next model.
  uv run --env-file "$env_file" harbor run -y \
    -p "$dataset_dir" \
    --agent assetops_harbor.stirrup:StirrupAgent \
    --model "$model_id" \
    --ak code_enabled=true \
    --ak code_backend=docker \
    --ak allow_docker_backend=true \
    --ak workspace_dir=/workspace-share \
    ${effort_args[@]+"${effort_args[@]}"} \
    --extra-docker-compose benchmarks/harbor/overlays/private-data.yaml \
    --extra-docker-compose benchmarks/harbor/overlays/code-sandbox.yaml \
    --n-concurrent "$n_concurrent" \
    --job-name "$job_name" \
    -o "$jobs_dir" || true
done
