# Running the Benchmark

`benchmarks/run.sh` runs a model matrix across a set of scenarios, saves a
trajectory per scenario, and scores the results. This page is everything you
need to get from a fresh clone to a leaderboard report.

Related docs: [scenario_suite/README.md](scenario_suite/README.md) for scenario
selectors, [../docs/stirrup-agent.md](../docs/stirrup-agent.md) for the agent
itself, [../INSTRUCTIONS.md](../INSTRUCTIONS.md) for the full environment table.

---

## Quick start

```bash
# 0. dependencies
uv sync

# 1. configuration
cp .env.public .env && $EDITOR .env           # keys + the two paths

# 2. CouchDB — every MCP server reads from it
docker compose -f src/couchdb/docker-compose.yaml up -d

# 3. the code sandbox image
docker build -f src/agent/stirrup_agent/Dockerfile.code -t assetops-code .

# 4. check everything before spending tokens
./scripts/preflight_run.sh /path/to/scenarios_data /path/to/leaderboard

# 5. go  (paths come from .env; pass them as arguments to override)
./benchmarks/run.sh
```

`run.sh` takes exactly two positional arguments and reads everything else from
the environment:

| Argument | Also settable as | What it is |
| --- | --- | --- |
| `$1` | `SCENARIO_DIR` | Directory of `scenario_<id>/` folders — the benchmark corpus |
| `$2` | `LEADERBOARD_DIR` | Output root; trajectories, reports and workspaces are created under it |

Both can live in `.env` instead, in which case `./benchmarks/run.sh` needs no
arguments at all. Precedence is **positional arguments → exported environment →
`.env`**, so a one-off run can always override the file:

```bash
./benchmarks/run.sh /other/corpus /other/output      # wins over .env
```

---

## Prerequisites

### The scenario corpus

**This is the most common hard stop.** `run.sh` sets `scenario_ids=lite`, which
resolves through `scenario_suite/lite.yaml` to roughly 50 scenario ids (151,
301, 902, …). Each one needs a folder:

```
scenarios_data/
├── scenario_151/
│   ├── question.txt        the prompt handed to the agent
│   ├── manifest.json       which data to load into CouchDB
│   └── groundtruth.txt     expected answer
├── scenario_301/
│   └── ...
└── shared/                 data files referenced by manifests
```

`question.txt` and `manifest.json` are hard requirements — a missing one raises
and the scenario fails. A missing `groundtruth.txt` only prints
`warning: missing groundtruth for scenario N` and the agent still runs, but
scoring has nothing to compare against, so the scenario is effectively lost.
Watch for those warnings.

The repository ships only `scenario_1`, `scenario_2`, `default` and `shared`
under `src/couchdb/scenarios_data/`. Those are examples, not the benchmark
corpus. Running `lite` against them fails on the first scenario:

```
Resolved 50 scenario ids from lite
FileNotFoundError: Missing question file for scenario 151:
  .../scenarios_data/scenario_151/question.txt
```

The scenarios are published on
[Hugging Face](https://huggingface.co/datasets/ibm-research/AssetOpsBench) as
tabular rows (`id`, `type`, `text`, `category`, …), not as this folder layout,
so they have to be materialised into it first. Point `--scenario-root` at
wherever you build that.

To run against a subset you *do* have, pass a plain text file of ids instead of
a selector — one id per line, `#` for comments:

```bash
--scenario-ids my_scenarios.txt
```

See [scenario_suite/README.md](scenario_suite/README.md) for the selector
grammar (`fcc_lite`, `fcc+fmsr_all`, `lite`, `all`).

### CouchDB

Every MCP server — `iot`, `tsfm`, `wo`, `fmsr`, `vibration`, `utilities` —
reads from CouchDB. The bundled compose file matches the default credentials:

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
curl -u admin:password http://localhost:5984/_all_dbs      # expect a JSON array
```

The runner resets and reloads CouchDB from each scenario's `manifest.json`
before running it, so scenarios do not contaminate each other. This also means
**the database is wiped repeatedly during a run** — do not point
`COUCHDB_URL` at anything you care about.

### The code sandbox image

`stirrup_agent` defaults to `--code-backend docker` and cannot start the code
track without an image:

```bash
docker build -f src/agent/stirrup_agent/Dockerfile.code -t assetops-code .
```

It is `python:3.12-slim` plus NumPy, pandas and SciPy. If your Docker socket is
not in the default location, set `DOCKER_HOST` (Rancher Desktop:
`unix://$HOME/.rd/docker.sock`).

---

## Environment variables

Only the gateway credentials have no working default. Everything else can be
left alone unless your setup differs.

### Required

| Variable | Needed when | Notes |
| --- | --- | --- |
| `LITELLM_BASE_URL` | any `litellm_proxy/*` model | Usually ends in `/v1` |
| `LITELLM_API_KEY` | any `litellm_proxy/*` model | |
| `TOKENROUTER_BASE_URL` | any `tokenrouter/*` model | e.g. `https://api.tokenrouter.com/v1` |
| `TOKENROUTER_API_KEY` | any `tokenrouter/*` model | |

You only need the pair matching the prefixes in your model matrix. The default
matrix in `run.sh` uses both.

### Defaults that usually work

| Variable | Default | Purpose |
| --- | --- | --- |
| `COUCHDB_URL` | `http://localhost:5984` | Shared database |
| `COUCHDB_USERNAME` | `admin` | |
| `COUCHDB_PASSWORD` | `password` | |
| `TSFM_WORKDIR` | `/tmp/tsfm_work` | Where TSFM writes its `file://` payloads |
| `TSFM_STORE` | `couch` | Set to `memory` only for hermetic tests |
| `STIRRUP_CODE_IMAGE` | `assetops-code` | Code sandbox image |
| `DOCKER_HOST` | SDK default | Set only for a non-standard socket — find yours with `docker context inspect --format '{{.Endpoints.docker.Host}}'` |
| `ASSETOPS_SHARED_DIR` | `/tmp/assetops_shared` | Directory shared between `code_exec` and the host-side MCP servers |
| `FMSR_MODEL_ID` | a `watsonx/*` model | Override to run the fmsr `generate_*` tools through another gateway |

Per-database names (`IOT_DBNAME`, `WO_DBNAME`, `CATALOG_DBNAME`, …) default to
values matching the bundled compose file; see
[../INSTRUCTIONS.md](../INSTRUCTIONS.md) for the full list.

### WatsonX — needed more often than it looks

`run.sh` itself never uses WatsonX. But the `fmsr` server's `generate_*` tools
default to `watsonx/meta-llama/llama-3-3-70b-instruct`, and when the
credentials are absent the server does not fail loudly — it logs
`LLM unavailable (generate_* tools disabled)` at startup and every later call
returns:

```json
{"error": "LLM unavailable"}
```

The `lite` profile includes fmsr scenarios (902, 904, 905, 906, …), so those
run with a tool quietly missing and score badly for a reason that never appears
as an error. Either set `WATSONX_APIKEY` / `WATSONX_PROJECT_ID`, or point that
server at a gateway you already have:

```bash
FMSR_MODEL_ID=litellm_proxy/aws/claude-opus-5
```

### Genuinely not needed

No judge model: evaluation defaults to the `static_json` scorer, which compares
against `groundtruth.txt` without an LLM. `--judge-model` is only required if a
scenario routes to `llm_judge`.

Everything above can live in `.env` at the repo root — start from
`cp .env.public .env`. The Python runners read it via `load_dotenv()`, and
`run.sh` parses it with matching semantics (an exported value always wins over
the file), so the same `.env` drives both halves of a run.

---

## Choosing models

Edit the `model_configs` array in `run.sh`. Each entry is
`"<model-id> <reasoning-effort>"`:

```bash
model_configs=(
  "litellm_proxy/aws/claude-opus-5 high"
  "tokenrouter/MiniMax-M3 high"
)
```

The prefix selects the gateway: `litellm_proxy/` uses `LITELLM_*`,
`tokenrouter/` uses `TOKENROUTER_*`. Everything after the prefix is the model
name sent to that gateway, so it must match a name in its catalogue.

**Valid reasoning-effort values** are `none`, `minimal`, `low`, `medium`,
`high`, `xhigh` and `default`. `max` is *not* valid anywhere — no provider
accepts it. It is dropped or rejected on the wire, so extended thinking never
turns on and no reasoning trace comes back. Use `xhigh`.

Note also that not every model returns reasoning over a channel that can be
captured. Some return it as a separate field (`reasoning_content`,
`thinking_blocks`); others inline it into the answer as `<think>` markup; and
the GPT-5 family returns none at all over Chat Completions, even while billing
reasoning tokens.

---

## Output

```
$LEADERBOARD_DIR/
├── assetopsbench-trajectories/
│   └── stirrup_agent/
│       ├── litellm_proxy-aws-claude-opus-5/
│       │   ├── stirrup_agent_151.json        one per scenario
│       │   └── stirrup_agent_301.json
│       └── tokenrouter-MiniMax-M3/
│           └── ...
├── assetopsbench-reports/
│   └── stirrup_agent/
│       └── litellm_proxy-aws-claude-opus-5/
│           └── _aggregate.json               scores for that model
└── assetopsbench-stirrup-workspaces/
    └── stirrup_agent/
        └── litellm_proxy-aws-claude-opus-5/
            └── stirrup_agent_151/            code the agent wrote
```

Model directory names are the model id with `/` replaced by `-`:
`litellm_proxy/aws/claude-opus-5` → `litellm_proxy-aws-claude-opus-5`.

Each trajectory JSON holds the question, the final answer, and a turn-by-turn
record with tool calls, tool outputs, token usage and timings.

### Keeping the code workspaces

By default the per-run workspace is **deleted when the agent session ends** —
the directory tree is created, filled, and emptied. Everything the agent wrote,
including the `mcp_results/` snapshots of large MCP responses, is discarded.

Add `--preserve-workspaces` to the `scenario_suite_runner` invocation in
`run.sh` to keep them. Budget disk for it: every `mcp_results/` file is over
100 KiB by definition, and there is one per oversized tool result per scenario.

### Re-running

`run.sh` passes `--skip-existing`, so any scenario whose trajectory file
already exists is skipped. That makes an interrupted run resumable, but it also
means a re-run after changing something is a no-op. To force a genuine re-run:

```bash
rm -rf "$LEADERBOARD_DIR/assetopsbench-trajectories"
```

`--continue-on-error` is also set, so one failing scenario does not abort the
matrix. Check the console output for `error: scenario N failed` lines — a run
can "finish" with many scenarios missing.

### The shared directory

`code_exec` runs inside a container; the MCP servers run on the host. They
share no filesystem, so a path valid to one is meaningless to the other — hand
TSFM a `/workspace/...` path and it raises `FileNotFoundError` on a file the
agent just wrote successfully.

`ASSETOPS_SHARED_DIR` is the one directory mounted at the *same absolute path*
on both sides, so a single string works in either place. Each run gets its own
subdirectory (`<base>/<model>/<run_id>`), because nothing prunes this tree and
a stale file from an earlier run is worse than a missing one.

**It must sit inside a path your Docker VM shares with the host.** Docker does
not error when asked to mount a path outside that set — it silently mounts an
empty directory inside the VM, so the container writes happily and the host
never sees a byte. A startup probe catches this and refuses to continue:

| Runtime | Shared with the VM by default |
| --- | --- |
| Docker Desktop (macOS) | `/Users/$USER/...` |
| Rancher Desktop (macOS) | `/Users/$USER/...`, `/tmp/rancher-desktop/...` |
| Colima | `/Users/$USER/...` |
| Linux | anywhere |

The default `/tmp/assetops_shared` works on Linux but **not** on macOS with
Rancher Desktop or Colima. Point it somewhere under your home directory there.

---

## Preflight

`scripts/preflight_run.sh` checks every prerequisite at once instead of failing
one at a time, 50 scenarios deep:

```bash
./scripts/preflight_run.sh "$SCENARIO_DIR" "$LEADERBOARD_DIR"
```

It sources `.env` the same way the runners do, then verifies both arguments,
scenario coverage for the selected profile, both gateway keys (via a free
`GET /models` that costs no tokens), CouchDB reachability *and*
authentication, the Docker daemon and sandbox image, `uv`, and that
`TSFM_WORKDIR` is writable. It exits non-zero if anything would block the run.

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `Missing question file for scenario N` | Scenario corpus not present at `--scenario-root`; see above |
| `Invalid scenario selector '1'` | `--scenario-ids` takes a selector (`fcc_lite`) or a file, not a bare id |
| `401` from a gateway | Wrong key, or key not valid for that gateway |
| `404` from a gateway | Model name not in the catalogue, or `/v1` missing from the base URL |
| Model runs but no reasoning in the trajectory | Effort set to `max` (invalid), or the model returns no separate reasoning channel |
| `Failed to connect to Docker daemon` | Daemon not running, or `DOCKER_HOST` unset for a non-standard socket |
| `code_exec` hits `ModuleNotFoundError` | Sandbox image missing the library; rebuild `assetops-code` |
| `IoT records database not connected` | CouchDB unreachable or credentials wrong |
| TSFM: file not found for a path `code_exec` just wrote | `code_exec` runs in a container; MCP servers run on the host. They share no filesystem — use `ASSETOPS_SHARED_DIR` |
| `... is not shared with the code sandbox` at startup | `ASSETOPS_SHARED_DIR` is outside the set of paths your Docker VM shares; move it under `/Users/$USER` |
| fmsr scenarios score badly; tools return `LLM unavailable` | WatsonX credentials absent and `FMSR_MODEL_ID` not overridden |
| Workspace directory empty after a run | `--preserve-workspaces` not set |
| Re-run does nothing | `--skip-existing` plus existing trajectory files |