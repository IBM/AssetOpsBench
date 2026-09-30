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
# Credentials are read from ENV_FILE (default: the repo's .env) by
# `uv run --env-file`, into the Harbor process only; StirrupAgent forwards them
# to the agent phase. They never enter an image. ENV_FILE is the only file read:
# with another file, the repo's .env fills none of its gaps.
#
# Relative paths in -s, -l, -p, ENV_FILE and AOB_CODE_TAR_DIR are relative to
# the directory the script is run from.
#
# One Harbor job per profile, model and reasoning effort, at
# LEADERBOARD_DIR/harbor-jobs/stirrup_agent__<profile>__<model>[__<effort>].
# Re-running resumes that job, finishing only the trials it has not completed,
# which is the equivalent of run.sh's --skip-existing. A resume keeps the
# settings the job started with, so a changed -n applies to new jobs only. Only
# one run.sh works on a job at a time; another skips it.
#
# Exits non-zero when a model's job could not start or resume, including a
# model skipped by the checks before it.

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

# Paths the caller gives are relative to where the script was run, as for any
# command; the defaults are the repo's own. Resolved before the cd below.
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
# StirrupAgent reads this file in place of the nearest .env, so the repo's .env
# cannot fill the gaps of another ENV_FILE.
export AOB_ENV_FILE="$env_file"

# Each job builds from its own copy of the tasks, generated when the job starts
# and never regenerated. Harbor refuses to resume a job whose tasks differ from
# its lock, so regenerating on every run left a job unresumable after any change
# to the template, the suite's scenario files or the generator; and one shared
# folder let a second run.sh delete tasks a running job was still reading. The
# copies hold every scenario's answers (tests/, solution/), so they stay in the
# repo's gitignored datasets/ rather than beside the results.
tasks_root="$repo_root/benchmarks/harbor/datasets/jobs"

# Everything the script creates for itself goes on exit: the runtime image pin,
# a code tar it did not finish writing, and the lock of the job it was on.
runtime_pin=""
code_tar_partial=""
job_lock=""
cleanup() {
  if [[ -n "$runtime_pin" ]]; then docker rmi "$runtime_pin" >/dev/null 2>&1 || true; fi
  if [[ -n "$code_tar_partial" ]]; then rm -f "$code_tar_partial"; fi
  if [[ -n "$job_lock" ]]; then rm -rf "$job_lock"; fi
}
trap cleanup EXIT

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
export AOB_RUNTIME_IMAGE="$runtime_pin"
printf 'Runtime image: %s (%s)\n' "$runtime_image" "${runtime_id:0:12}"

# The code sandbox image, as a tar each trial's Docker-in-Docker daemon loads
# (benchmarks/harbor/overlays/code-sandbox.yaml). It is built on every run,
# which the build cache makes cheap, so a change to Dockerfile.code is picked
# up. The tar is named after the image id and written once, so a new build never
# rewrites the file a running job's trials load. Each job records its tar and a
# resume loads that one; a new job loads this run's. AOB_CODE_TAR from the
# environment is ignored here, since every job sets its own. Old tars stay in
# AOB_CODE_TAR_DIR until removed.
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

# Fail fast when a model cannot be served. Otherwise every trial builds its
# containers, loads its data, retries the model for minutes and exits 1, which
# is how a dropped VPN or an expired key turns into a job of failed trials.
#
# Each router in use must answer, and must not reject its key: GET /models
# answers 401 or 403 to a bad key, and any other answer, 404 included, passes.
# The routers in use are the model's and FMSR_MODEL_ID's. A model with no router
# prefix is not probed, but it needs FMSR_MODEL_ID: the FMSR server otherwise
# runs generate_failure_modes on the agent's model, accepts only router models,
# and every fmsr scenario would run without that tool.
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

# One run.sh per job at a time: two resuming the same job would run its trials
# twice into the same directories. mkdir is atomic, and the lock holds its
# owner's PID, so a lock left by a run.sh that died is taken over.
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

# Trials a resume runs again: those that failed for a reason other than the
# model's own work, i.e. its API or the network, the environment, the verifier,
# or Ctrl-C. Harbor matches the exact class name, so every subclass is listed.
# Kept as results: AgentTimeoutError, ContextWindowExceededError,
# OutputTokenExceededError and AgentSafetyRefusalError. Names from Harbor 0.23;
# src/assetops_harbor/tests/test_run_sh.py checks them against the installed one.
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

# Non-zero when a model's job could not start or resume, or was skipped. A
# trial that fails inside a job does not count: Harbor records it and the loop
# moves on.
status=0

for model_config in "${model_configs[@]}"; do
  read -r model_id reasoning_effort <<< "$model_config"
  [[ -z "${model_id:-}" ]] && continue

  # The profile and effort are in the name because a resume runs the job's own
  # saved settings. Named after the model alone, a second effort or another
  # profile resumed the first job instead of starting its own.
  job_name="stirrup_agent__${profile_slug}__$(slug "$model_id")"
  if [[ -n "${reasoning_effort:-}" ]]; then
    job_name+="__$(slug "$reasoning_effort")"
  fi
  job_path="$jobs_dir/$job_name"
  # Keyed by the job's full path, so the same job name under another
  # LEADERBOARD_DIR gets its own copy.
  tasks_dir="$tasks_root/$job_name-$(printf '%s' "$job_path" | cksum | cut -d' ' -f1)"
  # What a job started on, beside the job rather than in it. Harbor's resume
  # lock covers the task files but not the base they build FROM, so without the
  # runtime image record a resume with another -r would mix two images in one
  # job. The code tar record makes a resume load the job's own code image. The
  # suite record matters because a resume reuses the job's manifests but mounts
  # shared/ from the current -s: another suite would pair one suite's manifests
  # with another's data.
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
    # Drop the trials in retry_error_types so they run again; scored trials
    # are kept, as run.sh's --skip-existing kept scenarios that already had a
    # trajectory. Harbor refuses to resume once the overlays differ from the
    # job's lock. The tasks cannot differ: the job builds from its own copy.
    # Say so rather than skip the model silently.
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

  # A new job, so its own tasks from scratch: the generator overwrites tasks
  # but never removes them, so a folder left from an attempt that never
  # started would join this one.
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

  # harbor run exits 0 when trials fail, recording them in the job, so the loop
  # moves on to the next model as run.sh's --continue-on-error did. Non-zero
  # means the job itself could not run, e.g. a rejected config or StirrupAgent
  # refusing its credentials, which aborts the job as the first trial starts.
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
