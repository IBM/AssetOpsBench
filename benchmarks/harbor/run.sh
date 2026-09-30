#!/usr/bin/env bash
# Harbor counterpart of benchmarks/run.sh: the same scenarios, stirrup-agent and
# code sandbox, but each scenario is a Harbor trial with its own CouchDB, so
# scenarios run concurrently. See benchmarks/harbor/README.md.
#
#   bash benchmarks/harbor/run.sh -s SCENARIO_DIR -l LEADERBOARD_DIR \
#     [-n N_CONCURRENT] [-p PROFILE] [-r RUNTIME_IMAGE] \
#     [-m "MODEL_ID REASONING_EFFORT"]...
#
# Needs Docker, `uv sync --extra harbor` and the runtime image, built with
# scripts/build-runtime-image.sh or published and passed as -r (or
# AOB_RUNTIME_IMAGE). Credentials come from ENV_FILE (default: the repo's .env),
# the only file read. Relative paths are relative to the caller's directory.
#
# One Harbor job per profile, model and effort, at
# LEADERBOARD_DIR/harbor-jobs/stirrup_agent__<profile>__<model>[__<effort>].
# Re-running resumes it with its original settings. Exits non-zero when a
# model's job could not start or resume, or the model was skipped.

set -euo pipefail

usage() {
  printf 'Usage: %s -s SCENARIO_DIR -l LEADERBOARD_DIR [-n N_CONCURRENT] [-p PROFILE] [-r RUNTIME_IMAGE] [-m "MODEL_ID EFFORT"]...\n' "$0" >&2
}

scenario_dir="${SCENARIO_DIR:-}"
leaderboard_dir="${LEADERBOARD_DIR:-}"
n_concurrent="${N_CONCURRENT:-4}"
profile="${PROFILE:-}"
env_file="${ENV_FILE:-}"
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

# Resolve caller paths before the cd below.
caller_dir="$PWD"
absolute() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$caller_dir" "$1" ;;
  esac
}
scenario_dir="$(absolute "$scenario_dir")"
leaderboard_dir="$(absolute "$leaderboard_dir")"
if [[ -n "$profile" ]]; then
  profile="$(absolute "$profile")"
else
  profile="$repo_root/benchmarks/scenario_suite/all.yaml"
fi
if [[ -n "$env_file" ]]; then
  env_file="$(absolute "$env_file")"
else
  env_file="$repo_root/.env"
fi
code_tar_dir="$(absolute "${AOB_CODE_TAR_DIR:-$HOME/.cache/assetopsbench}")"

cd "$repo_root"

if [[ ! -d "$scenario_dir" ]]; then
  printf 'Scenario directory not found: %s\n' "$scenario_dir" >&2
  exit 2
fi
scenario_dir="$(cd "$scenario_dir" && pwd)"
if [[ ! -f "$profile" ]]; then
  printf 'Profile not found: %s\n' "$profile" >&2
  exit 2
fi
mkdir -p "$leaderboard_dir"
leaderboard_dir="$(cd "$leaderboard_dir" && pwd)"
jobs_dir="$leaderboard_dir/harbor-jobs"
mkdir -p "$jobs_dir"

if [[ ! -f "$env_file" ]]; then
  printf 'Credentials file not found: %s (set ENV_FILE)\n' "$env_file" >&2
  exit 2
fi
# StirrupAgent reads this file instead of the nearest .env.
export AOB_ENV_FILE="$env_file"

# Each job gets its own copy of the tasks, generated once when it starts:
# Harbor refuses to resume a job whose tasks changed, and a shared folder could
# be rewritten under a running job. The copies hold answers, so they stay in the
# gitignored datasets/ rather than beside the results.
tasks_root="$repo_root/benchmarks/harbor/datasets/jobs"

# Removed on exit: the runtime image pin, a partial code tar, the job lock.
runtime_pin=""
code_tar_partial=""
job_lock=""
cleanup() {
  if [[ -n "$runtime_pin" ]]; then docker rmi "$runtime_pin" >/dev/null 2>&1 || true; fi
  if [[ -n "$code_tar_partial" ]]; then rm -f "$code_tar_partial"; fi
  if [[ -n "$job_lock" ]]; then rm -rf "$job_lock"; fi
}
trap cleanup EXIT

# overlays/private-data.yaml mounts $AOB_PRIVATE_DIR/shared into each trial.
if [[ ! -d "$scenario_dir/shared" ]]; then
  printf "No shared/ directory in %s; is -s the suite's scenarios_data?\n" "$scenario_dir" >&2
  exit 2
fi
export AOB_PRIVATE_DIR="$scenario_dir"

# -r wins, then the shell's AOB_RUNTIME_IMAGE, then ENV_FILE's, then the local
# default.
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

# A build never refreshes a published image it already has, so pull it here. A
# local build (e.g. the default assetopsbench/runtime:dev) is used as is.
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

# Pin the base for the whole run: a tag such as :dev can move mid-run, and each
# trial resolves FROM when it builds. FROM cannot name an image id, so tag it
# under a name private to this process.
runtime_id="$(docker image inspect --format '{{.Id}}' "$runtime_image")"
runtime_id="${runtime_id#sha256:}"
runtime_pin="aob-runtime-pin:${runtime_id:0:12}-$$"
docker tag "$runtime_image" "$runtime_pin"
export AOB_RUNTIME_IMAGE="$runtime_pin"
printf 'Runtime image: %s (%s)\n' "$runtime_image" "${runtime_id:0:12}"

# The code sandbox image, as a tar each trial's dind loads. Rebuilt every run
# (cheap when cached) and saved once per image id, so a running job's tar is
# never rewritten. Old tars stay in AOB_CODE_TAR_DIR until removed.
code_image=assetops-code:dev
docker build -q -t "$code_image" \
  -f src/agent/stirrup_agent/Dockerfile.code src/agent/stirrup_agent >/dev/null
code_id="$(docker image inspect --format '{{.Id}}' "$code_image")"
code_id="${code_id#sha256:}"
code_tar="$code_tar_dir/assetops-code-${code_id:0:12}.tar"
if [[ ! -s "$code_tar" ]]; then
  mkdir -p "$code_tar_dir"
  echo "Saving $code_image to $code_tar"
  code_tar_partial="$code_tar.partial.$$"
  docker save "$code_image" -o "$code_tar_partial"
  chmod 644 "$code_tar_partial"
  mv "$code_tar_partial" "$code_tar"
  code_tar_partial=""
fi
export AOB_CODE_IMAGE="$code_image"
printf 'Code image: %s (%s)\n' "$code_image" "${code_id:0:12}"

# A name safe for a directory: anything but letters, digits and ._- becomes a
# single dash, and a trailing dash is dropped.
slug() {
  local name
  name="$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '-' | tr -s '-')"
  printf '%s' "${name%-}"
}

profile_name="$(basename "$profile")"
profile_slug="$(slug "${profile_name%.*}")"

# Fail fast when a model cannot be served, rather than a job of failed trials.
# The model's and FMSR_MODEL_ID's routers must answer GET /models without a 401
# or 403. A model with no router prefix needs FMSR_MODEL_ID, since the FMSR
# server accepts only router models.
check_model() {
  uv run --env-file "$env_file" python - "$1" <<'PY'
import os
import sys
import urllib.error
import urllib.request

# src/llm/routers.py PROXY_ROUTERS; src/assetops_harbor/tests checks they match.
ROUTERS = {
    "litellm_proxy/": ("LITELLM_BASE_URL", "LITELLM_API_KEY"),
    "tokenrouter/": ("TOKENROUTER_BASE_URL", "TOKENROUTER_API_KEY"),
}


def router(model):
    return next((prefix for prefix in ROUTERS if model.startswith(prefix)), None)


model = sys.argv[1]
fmsr_model = os.environ.get("FMSR_MODEL_ID", "").strip()
if fmsr_model and not router(fmsr_model):
    sys.exit(f"FMSR_MODEL_ID={fmsr_model} needs a {' or '.join(ROUTERS)} prefix")
if not fmsr_model and not router(model):
    sys.exit(
        f"{model} has no {' or '.join(ROUTERS)} prefix, so the FMSR server would "
        "reject it; set FMSR_MODEL_ID to a router model"
    )

for prefix in dict.fromkeys(p for p in (router(model), router(fmsr_model)) if p):
    base_var, key_var = ROUTERS[prefix]
    base, key = os.environ.get(base_var, ""), os.environ.get(key_var, "")
    missing = [name for name, value in ((base_var, base), (key_var, key)) if not value]
    if missing:
        sys.exit(f"{' and '.join(missing)} not set for {prefix} models")
    request = urllib.request.Request(
        base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            sys.exit(f"{base_var} rejected {key_var} (HTTP {exc.code})")
    except Exception as exc:
        sys.exit(f"cannot reach {base_var} ({exc}); check the VPN or network")
PY
}

# One run.sh per job at a time. mkdir is atomic; the lock holds its owner's
# PID, so a lock left by a dead run.sh is taken over.
lock_job() {
  local lock="$1" owner
  if mkdir "$lock" 2>/dev/null; then
    printf '%s\n' "$$" >"$lock/pid"
    return 0
  fi
  owner="$(cat "$lock/pid" 2>/dev/null || true)"
  if [[ -z "$owner" ]] || kill -0 "$owner" 2>/dev/null; then
    return 1
  fi
  rm -rf "$lock"
  mkdir "$lock" 2>/dev/null || return 1
  printf '%s\n' "$$" >"$lock/pid"
}

# Trials a resume reruns: failures not caused by the model's own work (API,
# network, environment, verifier, Ctrl-C). Harbor matches exact class names;
# src/assetops_harbor/tests/test_run_sh.py checks them against Harbor. Timeouts,
# context/output overruns and safety refusals are kept as results.
retry_error_types=(
  CancelledError
  NonZeroAgentExitCodeError
  ApiError
  ApiRateLimitError
  ApiUsageLimitError
  ApiInternalServerError
  ApiOverloadedError
  ApiConnectionClosedError
  ApiResponseStalledError
  UnknownApiError
  ApiProviderResourceNotFoundError
  AgentAuthenticationError
  ModelNotFoundError
  NetworkConnectionError
  AgentSetupTimeoutError
  EnvironmentStartTimeoutError
  HealthcheckError
  VerifierTimeoutError
  RewardFileNotFoundError
  RewardFileEmptyError
  VerifierOutputParseError
  AddTestsDirError
  DownloadVerifierDirError
)
retry_filters=()
for error_type in "${retry_error_types[@]}"; do
  retry_filters+=(--filter-error-type "$error_type")
done

# Non-zero when a model's job could not start or resume, or was skipped.
status=0

for model_config in "${model_configs[@]}"; do
  read -r model_id reasoning_effort <<< "$model_config"
  [[ -z "${model_id:-}" ]] && continue

  # Profile and effort are in the name so each gets its own job.
  job_name="stirrup_agent__${profile_slug}__$(slug "$model_id")"
  if [[ -n "${reasoning_effort:-}" ]]; then
    job_name+="__$(slug "$reasoning_effort")"
  fi
  job_path="$jobs_dir/$job_name"
  # Keyed by the job's full path, so another LEADERBOARD_DIR gets its own copy.
  tasks_dir="$tasks_root/$job_name-$(printf '%s' "$job_path" | cksum | cut -d' ' -f1)"
  # What the job started on, which Harbor's own resume check does not cover: the
  # runtime image, the code tar, and the suite whose shared/ the mount supplies.
  image_record="$jobs_dir/$job_name.runtime-image"
  code_record="$jobs_dir/$job_name.code-tar"
  suite_record="$jobs_dir/$job_name.suite"

  if ! check_model "$model_id"; then
    echo "Skipping $model_id" >&2
    status=1
    continue
  fi

  if ! lock_job "$job_path.lock"; then
    printf 'Skipping %s: another run.sh (PID %s) is working on it.\n' \
      "$job_path" "$(cat "$job_path.lock/pid" 2>/dev/null || echo unknown)" >&2
    printf 'If none is, remove %s.\n' "$job_path.lock" >&2
    status=1
    continue
  fi
  job_lock="$job_path.lock"

  echo "Running $model_id with reasoning effort ${reasoning_effort:-default} -> $job_path"

  if [[ -f "$job_path/config.json" ]]; then
    job_code_tar="$code_tar"
    [[ -f "$code_record" ]] && job_code_tar="$(cat "$code_record")"
    if [[ -f "$image_record" ]] && [[ "$(cut -f1 "$image_record")" != "$runtime_id" ]]; then
      printf '%s started on runtime image %s, not %s (%s).\n' \
        "$job_path" "$(cut -f2 "$image_record")" "$runtime_image" "${runtime_id:0:12}" >&2
      printf 'Pass that image as -r to finish it, or move the job aside to rerun %s.\n' \
        "$model_id" >&2
      status=1
    elif [[ -f "$suite_record" ]] && [[ "$(cat "$suite_record")" != "$scenario_dir" ]]; then
      printf '%s started on the suite in %s, not %s.\n' \
        "$job_path" "$(cat "$suite_record")" "$scenario_dir" >&2
      printf 'Pass that directory as -s to finish it, or move the job aside to rerun %s.\n' \
        "$model_id" >&2
      status=1
    elif [[ ! -d "$tasks_dir" ]]; then
      printf 'The tasks %s started with are gone (%s).\n' "$job_path" "$tasks_dir" >&2
      printf 'Move the job aside to rerun %s from scratch.\n' "$model_id" >&2
      status=1
    elif [[ ! -s "$job_code_tar" ]]; then
      printf 'The code image %s started with is gone (%s).\n' "$job_path" "$job_code_tar" >&2
      printf 'Move the job aside to rerun %s from scratch.\n' "$model_id" >&2
      status=1
    # Rerun the trials in retry_error_types; scored trials are kept.
    elif ! AOB_CODE_TAR="$job_code_tar" uv run --env-file "$env_file" \
      harbor jobs resume -p "$job_path" "${retry_filters[@]}"; then
      printf 'Could not resume %s. If its overlays changed since it started,\n' "$job_path" >&2
      printf 'move it aside to rerun %s from scratch.\n' "$model_id" >&2
      status=1
    fi
    rm -rf "$job_lock"
    job_lock=""
    continue
  fi

  # The generator never removes tasks, so clear any left by a failed start.
  rm -rf "$tasks_dir"
  uv run python benchmarks/harbor/adapter/generate_tasks.py \
    --scenario-root "$scenario_dir" \
    --profile "$profile" \
    --output-dir "$tasks_dir" \
    --dataset-name assetopsbench/suite \
    --skip-missing \
    --overwrite >/dev/null

  printf '%s\t%s\n' "$runtime_id" "$runtime_image" >"$image_record"
  printf '%s\n' "$code_tar" >"$code_record"
  printf '%s\n' "$scenario_dir" >"$suite_record"

  effort_args=()
  if [[ -n "${reasoning_effort:-}" ]]; then
    effort_args=(--ak "reasoning_effort=$reasoning_effort")
  fi

  # harbor run exits 0 when trials fail; non-zero means the job itself could not
  # run, e.g. a rejected config or missing credentials.
  if ! AOB_CODE_TAR="$code_tar" uv run --env-file "$env_file" harbor run -y \
    -p "$tasks_dir" \
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
    -o "$jobs_dir"; then
    printf 'Harbor could not run %s; see the error above.\n' "$job_path" >&2
    status=1
  fi
  rm -rf "$job_lock"
  job_lock=""
done

exit "$status"
