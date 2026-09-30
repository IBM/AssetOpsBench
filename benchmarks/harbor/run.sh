#!/usr/bin/env bash
# Harbor counterpart of benchmarks/run.sh.
#
# Same scenarios, same stirrup-agent CLI and the same code-execution sandbox,
# but each scenario runs as a Harbor trial with its own Compose project and its
# own CouchDB, so scenarios run concurrently instead of one at a time against a
# shared database.
#
#   bash benchmarks/harbor/run.sh -s SCENARIO_DIR -l LEADERBOARD_DIR \
#     [-n N_CONCURRENT] [-p PROFILE] [-r RUNTIME_IMAGE] \
#     [-m "MODEL_ID REASONING_EFFORT"]...
#
# Prerequisites: Docker running, `uv sync --extra harbor`, and the runtime image,
# either built locally
#
#   bash benchmarks/harbor/scripts/build-runtime-image.sh
#
# or published, passed as -r (or AOB_RUNTIME_IMAGE in the shell or ENV_FILE),
# e.g. -r quay.io/assetopsbench/runtime:dev.
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
  printf 'Usage: %s -s SCENARIO_DIR -l LEADERBOARD_DIR [-n N_CONCURRENT] [-p PROFILE] [-r RUNTIME_IMAGE] [-m "MODEL_ID EFFORT"]...\n' "$0" >&2
}

scenario_dir="${SCENARIO_DIR:-}"
leaderboard_dir="${LEADERBOARD_DIR:-}"
n_concurrent="${N_CONCURRENT:-4}"
profile="${PROFILE:-benchmarks/scenario_suite/all.yaml}"
env_file="${ENV_FILE:-.env}"
runtime_image="${AOB_RUNTIME_IMAGE:-}"
model_configs=()

while getopts ':s:l:n:p:r:m:' option; do
  case "$option" in
    s) scenario_dir="$OPTARG" ;;
    l) leaderboard_dir="$OPTARG" ;;
    n) n_concurrent="$OPTARG" ;;
    p) profile="$OPTARG" ;;
    r) runtime_image="$OPTARG" ;;
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

code_image=assetops-code:dev
code_tar="${AOB_CODE_TAR:-$HOME/assetops-code.tar}"
dataset_dir=benchmarks/harbor/datasets/assetopsbench-suite

# The suite's shared/ data reaches each trial through
# overlays/private-data.yaml, a read-only bind mount of this directory's shared/.
# Compose reads the variable on `harbor run` and again on `harbor jobs resume`.
# A missing shared/ would not fail the mount: Docker creates the host directory,
# and every collection then loads empty without an error.
if [[ ! -d "$scenario_dir/shared" ]]; then
  printf "No shared/ directory in %s; is -s the suite's scenarios_data?\n" "$scenario_dir" >&2
  exit 2
fi
export AOB_PRIVATE_DIR="$scenario_dir"

# Every task image builds FROM the runtime image, through the AOB_RUNTIME_IMAGE
# build arg in the task's docker-compose.yaml. -r wins, then the shell's
# AOB_RUNTIME_IMAGE, then ENV_FILE's, then the local default: the order
# `uv run --env-file` gives Harbor, where the environment beats the file.
if [[ -z "$runtime_image" ]]; then
  runtime_image="$(uv run --env-file "$env_file" python -c \
    'import os; print(os.environ.get("AOB_RUNTIME_IMAGE", ""))')"
fi
runtime_image="${runtime_image:-assetopsbench/runtime:dev}"

# The reference without its tag or digest, spelled as .RepoDigests spells it.
image_repo() {
  local ref="${1%@*}"
  if [[ "${ref##*/}" == *:* ]]; then ref="${ref%:*}"; fi
  ref="${ref#docker.io/}"
  printf '%s' "${ref#library/}"
}

# True when the local copy of $1 came from (or went to) that same repository,
# i.e. it is a published image rather than a local build.
from_registry() {
  local repo digest
  repo="$(image_repo "$1")"
  while read -r digest; do
    [[ "${digest%@*}" == "$repo" ]] && return 0
  done < <(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$1")
  return 1
}

# A local copy satisfies FROM, so the build never refreshes a published image;
# pull it here instead, on Docker Hub or any other registry. A local build that
# was never pushed under this name (the default assetopsbench/runtime:dev) is
# used as is, and a pull never replaces it.
if ! docker image inspect "$runtime_image" >/dev/null 2>&1; then
  if ! docker pull "$runtime_image"; then
    printf 'Runtime image %s is not local and could not be pulled. Build it with\n' "$runtime_image" >&2
    printf '  bash benchmarks/harbor/scripts/build-runtime-image.sh\n' >&2
    exit 1
  fi
elif from_registry "$runtime_image" && ! docker pull "$runtime_image"; then
  printf 'warning: could not pull %s; using the local copy, which may be stale\n' \
    "$runtime_image" >&2
fi

# Pin the base for the whole run. A tag such as :dev can move while the run is
# going (a rebuild, or another run's pull), and every trial resolves FROM when
# it builds, so later trials would silently switch base. FROM cannot name an
# image id, so tag this one under a name private to this process and remove it
# on exit. Compose reads the variable on `harbor run` and on `harbor jobs resume`.
runtime_id="$(docker image inspect --format '{{.Id}}' "$runtime_image")"
runtime_id="${runtime_id#sha256:}"
runtime_pin="aob-runtime-pin:${runtime_id:0:12}-$$"
docker tag "$runtime_image" "$runtime_pin"
trap 'docker rmi "$runtime_pin" >/dev/null 2>&1 || true' EXIT
export AOB_RUNTIME_IMAGE="$runtime_pin"
printf 'Runtime image: %s (%s)\n' "$runtime_image" "${runtime_id:0:12}"

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

# Non-zero when a model's job could not run or resume. A trial that fails
# inside a job does not count: Harbor records it and the loop moves on.
status=0

for model_config in "${model_configs[@]}"; do
  read -r model_id reasoning_effort <<< "$model_config"
  [[ -z "${model_id:-}" ]] && continue

  model_slug="$(printf '%s' "$model_id" | tr -c 'A-Za-z0-9._-' '-' | tr -s '-')"
  job_name="stirrup_agent__${model_slug%-}"
  job_path="$jobs_dir/$job_name"
  # The runtime image a job started on, beside the job rather than in it.
  # Harbor's resume lock covers the task files but not the base they build
  # FROM, so without this a resume with another -r would mix two images in
  # one job.
  image_record="$jobs_dir/$job_name.runtime-image"

  if ! router_reachable "$model_id"; then
    echo "Skipping $model_id: its router is unreachable" >&2
    continue
  fi

  echo "Running $model_id with reasoning effort ${reasoning_effort:-default} -> $job_path"

  if [[ -f "$job_path/config.json" ]]; then
    if [[ -f "$image_record" ]] && [[ "$(cut -f1 "$image_record")" != "$runtime_id" ]]; then
      printf '%s started on runtime image %s, not %s (%s).\n' \
        "$job_path" "$(cut -f2 "$image_record")" "$runtime_image" "${runtime_id:0:12}" >&2
      printf 'Pass that image as -r to finish it, or move the job aside to rerun %s.\n' \
        "$model_id" >&2
      status=1
      continue
    fi
    # Drop trials whose agent crashed (e.g. the model was unreachable) so
    # they run again; scored trials are kept, as run.sh's --skip-existing
    # kept scenarios that already had a trajectory.
    # Harbor refuses to resume once the tasks or overlays differ from the
    # job's lock, e.g. a job started before run.sh switched to the shared/
    # mount. Say so rather than skip the model silently.
    if ! uv run --env-file "$env_file" harbor jobs resume -p "$job_path" \
      --filter-error-type NonZeroAgentExitCodeError; then
      printf 'Could not resume %s. If its tasks or overlays changed since it\n' "$job_path" >&2
      printf 'started, move it aside to rerun %s from scratch.\n' "$model_id" >&2
      status=1
    fi
    continue
  fi

  mkdir -p "$jobs_dir"
  printf '%s\t%s\n' "$runtime_id" "$runtime_image" >"$image_record"

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

exit "$status"
