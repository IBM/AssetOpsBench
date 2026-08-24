#!/usr/bin/env bash
#
# Check everything benchmarks/run.sh needs, before it burns a model call.
#
#   ./scripts/preflight_run.sh SCENARIO_DIR LEADERBOARD_DIR
#   SCENARIO_DIR=... LEADERBOARD_DIR=... ./scripts/preflight_run.sh
#
# run.sh takes those same two positional arguments and nothing else — every
# other input is an environment variable or an out-of-band prerequisite
# (CouchDB, the sandbox image, the scenario corpus). This script reports all
# of them at once instead of failing one at a time, 50 scenarios deep.
#
# Exit 0 when the run can start; 1 when something would block it.

set -uo pipefail   # deliberately NOT -e: report every problem, not the first

scenario_dir="${1:-${SCENARIO_DIR:-}}"
leaderboard_dir="${2:-${LEADERBOARD_DIR:-}}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0
warn=0

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; D=$'\033[2m'; Z=$'\033[0m'
else
  G=''; R=''; Y=''; D=''; Z=''
fi

ok()   { printf '  %sok  %s %s\n'   "$G" "$Z" "$1"; }
bad()  { printf '  %sFAIL%s %s\n'   "$R" "$Z" "$1"; fail=$((fail+1)); }
soft() { printf '  %swarn%s %s\n'   "$Y" "$Z" "$1"; warn=$((warn+1)); }
hint() { printf '       %s%s%s\n'   "$D" "$1" "$Z"; }
head_() { printf '\n%s\n' "$1"; }

# Load .env the way the agent CLIs do (agent/_cli_common.py calls load_dotenv),
# so this check sees exactly what a run would see.
if [[ -f "$repo_root/.env" ]]; then
  set -a; . "$repo_root/.env"; set +a
  env_source=".env + exported environment"
else
  env_source="exported environment only (no .env file)"
fi

printf 'AssetOpsBench run.sh preflight\n'
printf '%srepo: %s%s\n' "$D" "$repo_root" "$Z"
printf '%sconfig source: %s%s\n' "$D" "$env_source" "$Z"

# ---------------------------------------------------------------- arguments

head_ "run.sh arguments"

if [[ -z "$scenario_dir" ]]; then
  bad "SCENARIO_DIR not given (\$1, or export SCENARIO_DIR)"
  hint "the scenarios_data directory holding scenario_<id>/question.txt"
elif [[ ! -d "$scenario_dir" ]]; then
  bad "SCENARIO_DIR does not exist: $scenario_dir"
else
  ok "SCENARIO_DIR = $scenario_dir"
fi

if [[ -z "$leaderboard_dir" ]]; then
  bad "LEADERBOARD_DIR not given (\$2, or export LEADERBOARD_DIR)"
  hint "output root; run.sh creates trajectories/, reports/ and workspaces/ under it"
else
  mkdir -p "$leaderboard_dir" 2>/dev/null
  if [[ -w "$leaderboard_dir" ]]; then
    ok "LEADERBOARD_DIR = $leaderboard_dir (writable)"
  else
    bad "LEADERBOARD_DIR not writable: $leaderboard_dir"
  fi
fi

# ------------------------------------------------------------ scenario data

head_ "scenario corpus"

# run.sh hardcodes `scenario_ids=open`, which resolves through open.yaml to ~50
# ids. The public repo ships only scenario_1 / scenario_2, so this is the most
# common hard stop — and it fails 50 scenarios in, not at startup.
profile="$repo_root/benchmarks/scenario_suite/open.yaml"
if [[ ! -f "$profile" ]]; then
  soft "open.yaml profile not found; skipping scenario coverage check"
elif [[ -d "$scenario_dir" ]]; then
  ids=$(grep -oE '^\s+-\s+[0-9]+' "$profile" | grep -oE '[0-9]+')
  total=0; missing=0; missing_list=""
  for id in $ids; do
    total=$((total+1))
    if [[ ! -f "$scenario_dir/scenario_${id}/question.txt" ]]; then
      missing=$((missing+1))
      [[ $missing -le 6 ]] && missing_list="$missing_list $id"
    fi
  done
  if [[ $missing -eq 0 ]]; then
    ok "all $total 'open' scenarios present"
  else
    bad "$missing of $total 'open' scenarios have no question.txt (e.g.$missing_list)"
    hint "the public repo ships only scenario_1 / scenario_2 / default / shared;"
    hint "the benchmark corpus comes from the HuggingFace dataset and must be"
    hint "materialised as scenario_<id>/{question.txt,groundtruth.txt,manifest.json}"
    hint "-- or point --scenario-ids at a profile listing ids you actually have"
  fi
fi

# ------------------------------------------------------------- model access

head_ "model gateways"

# run.sh's matrix; both prefixes must resolve or those rows die immediately.
need_litellm=0; need_tokenrouter=0
runsh="$repo_root/benchmarks/run.sh"
if [[ -f "$runsh" ]]; then
  grep -q 'litellm_proxy/' "$runsh" && need_litellm=1
  grep -q 'tokenrouter/'   "$runsh" && need_tokenrouter=1
else
  soft "benchmarks/run.sh not found; checking both gateways anyway"
  need_litellm=1; need_tokenrouter=1
fi

check_gateway() {
  local label="$1" base_var="$2" key_var="$3"
  local base="${!base_var:-}" key="${!key_var:-}"

  if [[ -z "$base" || -z "$key" ]]; then
    bad "$label: $base_var and $key_var must both be set"
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    soft "$label: configured, but curl is unavailable so the key was not tested"
    return
  fi
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
           -H "Authorization: Bearer $key" "${base%/}/models" 2>/dev/null)
  case "$code" in
    200) ok  "$label: key accepted at ${base%/}" ;;
    401) bad "$label: 401 — key rejected" ;;
    403) bad "$label: 403 — key valid but lacks access" ;;
    404) bad "$label: 404 at ${base%/}/models — is /v1 missing from $base_var?" ;;
    000) bad "$label: could not reach ${base%/} (DNS, VPN, or wrong host)" ;;
    *)   bad "$label: HTTP $code from ${base%/}/models" ;;
  esac
}

[[ $need_litellm -eq 1 ]] \
  && check_gateway "LiteLLM proxy" LITELLM_BASE_URL LITELLM_API_KEY
[[ $need_tokenrouter -eq 1 ]] \
  && check_gateway "TokenRouter"   TOKENROUTER_BASE_URL TOKENROUTER_API_KEY

# Catch the reasoning_effort values no provider accepts, before the run.
if [[ -f "$runsh" ]]; then
  badeff=$(grep -oE '"[^"]+ (max|xxhigh|maximum)"' "$runsh" | sed 's/"//g')
  if [[ -n "$badeff" ]]; then
    bad "run.sh uses a reasoning effort no gateway accepts:"
    while IFS= read -r line; do hint "$line"; done <<< "$badeff"
    hint "valid: none minimal low medium high xhigh default  (use xhigh for max)"
  fi
fi

# ------------------------------------------------------------------ couchdb

head_ "CouchDB (every MCP server reads from it)"

couch_url="${COUCHDB_URL:-http://localhost:5984}"
couch_user="${COUCHDB_USERNAME:-admin}"
couch_pass="${COUCHDB_PASSWORD:-password}"

if ! command -v curl >/dev/null 2>&1; then
  soft "curl unavailable; cannot verify $couch_url"
else
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
           -u "$couch_user:$couch_pass" "${couch_url%/}/_all_dbs" 2>/dev/null)
  case "$code" in
    200) ok  "reachable and authenticated at $couch_url" ;;
    401) bad "401 at $couch_url — check COUCHDB_USERNAME / COUCHDB_PASSWORD" ;;
    000) bad "not reachable at $couch_url"
         hint "docker compose -f src/couchdb/docker-compose.yaml up -d" ;;
    *)   bad "HTTP $code from $couch_url" ;;
  esac
fi

# ------------------------------------------------------------ docker sandbox

head_ "Stirrup code sandbox (stirrup_agent defaults to --code-backend docker)"

image="${STIRRUP_CODE_IMAGE:-assetops-code}"
if ! command -v docker >/dev/null 2>&1; then
  bad "docker not on PATH — the code track cannot start"
elif ! docker info >/dev/null 2>&1; then
  bad "docker daemon not reachable${DOCKER_HOST:+ at $DOCKER_HOST}"
  hint "Rancher Desktop: export DOCKER_HOST=unix://\$HOME/.rd/docker.sock"
elif docker image inspect "$image" >/dev/null 2>&1; then
  ok "sandbox image '$image' present"
else
  bad "sandbox image '$image' not built"
  hint "docker build -f src/agent/stirrup_agent/Dockerfile.code -t $image ."
fi

# ------------------------------------------------------------------ toolchain

head_ "toolchain"

if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
  bad "uv not on PATH — run.sh invokes 'uv run'"
fi

# The fmsr generate_* tools disable themselves silently without credentials.
fmsr_model="${FMSR_MODEL_ID:-watsonx/meta-llama/llama-3-3-70b-instruct}"
if [[ "$fmsr_model" == watsonx/* ]]; then
  if [[ -n "${WATSONX_APIKEY:-}" && -n "${WATSONX_PROJECT_ID:-}" ]]; then
    ok "fmsr generate_* tools: WatsonX credentials present"
  else
    soft "fmsr generate_* tools will be DISABLED (no WATSONX_APIKEY / WATSONX_PROJECT_ID)"
    hint "they return {\"error\": \"LLM unavailable\"}; fmsr scenarios will score badly"
    hint "fix: set the WatsonX vars, or FMSR_MODEL_ID=litellm_proxy/<your-model>"
  fi
else
  ok "fmsr generate_* tools routed to $fmsr_model"
fi

# ASSETOPS_SHARED_DIR must live inside a path the Docker VM shares.
shared_dir="${ASSETOPS_SHARED_DIR:-/tmp/assetops_shared}"
case "$(uname -s)" in
  Darwin)
    if [[ "$shared_dir" == "$HOME"/* || "$shared_dir" == /Users/* || "$shared_dir" == /tmp/rancher-desktop/* ]]; then
      ok "ASSETOPS_SHARED_DIR = $shared_dir"
    else
      soft "ASSETOPS_SHARED_DIR = $shared_dir is outside the paths Docker shares on macOS"
      hint "Docker silently mounts an empty in-VM dir instead; put it under /Users/\$USER"
    fi ;;
  *) ok "ASSETOPS_SHARED_DIR = $shared_dir" ;;
esac

tsfm_workdir="${TSFM_WORKDIR:-/tmp/tsfm_work}"
mkdir -p "$tsfm_workdir" 2>/dev/null
if [[ -w "$tsfm_workdir" ]]; then
  ok "TSFM_WORKDIR = $tsfm_workdir (writable)"
else
  bad "TSFM_WORKDIR not writable: $tsfm_workdir"
fi

# --------------------------------------------------------------- disposition

head_ "notes"
if [[ -f "$runsh" ]] && ! grep -q -- '--preserve-workspaces' "$runsh"; then
  printf '  %s-%s run.sh omits --preserve-workspaces: the per-run workspace is\n' "$D" "$Z"
  printf '    deleted on exit, so nothing the agent wrote is kept.\n'
fi
printf '  %s-%s run.sh uses --skip-existing: clear %s\n' "$D" "$Z" \
  "${leaderboard_dir:-\$LEADERBOARD_DIR}/assetopsbench-trajectories"
printf '    to force a re-run of scenarios that already have output.\n'

printf '\n'
if [[ $fail -eq 0 ]]; then
  printf '%sReady.%s %d warning(s).\n' "$G" "$Z" "$warn"
  exit 0
fi
printf '%s%d blocker(s)%s, %d warning(s). Fix the FAIL lines above.\n' \
  "$R" "$fail" "$Z" "$warn"
exit 1
