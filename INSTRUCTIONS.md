# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [MCP Servers](#mcp-servers) — full reference in [docs/mcp-servers.md](docs/mcp-servers.md)
- [Example queries](#example-queries)
- [Agents](#agents)
- [Observability](#observability)
- [Evaluation](#evaluation)
- [Running Tests](#running-tests)
- [Architecture](#architecture)

---

## Prerequisites

- **Python 3.12+** — required by `pyproject.toml`
- **[uv](https://docs.astral.sh/uv/)** — dependency and environment manager

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
  # or: brew install uv
  ```

- **Docker** — for running CouchDB (IoT data store)

## Quick Start

### 1. Install dependencies

Run from the **repo root**:

```bash
uv sync
```

`uv sync` creates a virtual environment at `.venv/`, installs all dependencies, and registers the CLI entry points (`plan-execute`, `*-mcp-server`). You can either prefix commands with `uv run` (no activation needed) or activate the venv once for your shell session:

```bash
source .venv/bin/activate   # macOS / Linux
```

### 2. Configure environment

Copy `.env.public` to `.env` and fill in the required values (see [Environment Variables](#environment-variables)):

```bash
cp .env.public .env
# Then edit .env and set WATSONX_APIKEY, WATSONX_PROJECT_ID
# CouchDB defaults work out of the box with the Docker setup
```

### 3. Start CouchDB

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
```

Verify CouchDB is running:

```bash
curl -X GET http://localhost:5984/
```

### 4. Run an agent

Servers are stdio processes spawned on-demand by the agent CLIs — no manual startup needed. Pick a runner and pass it a question:

```bash
uv run plan-execute "What sensors are on Chiller 6?"
```

See [MCP Servers](#mcp-servers) for available tools and [docs/mcp-servers.md](docs/mcp-servers.md) for launching a server directly.

---

## Environment Variables

**CouchDB** — `iot` and `wo` servers

| Variable           | Default                 | Description              |
| ------------------ | ----------------------- | ------------------------ |
| `COUCHDB_URL`      | `http://localhost:5984` | CouchDB connection URL   |
| `COUCHDB_USERNAME` | `admin`                 | CouchDB admin username   |
| `COUCHDB_PASSWORD` | `password`              | CouchDB admin password   |
| `IOT_DBNAME`         | `iot`                   | IoT sensor database name      |
| `WO_DBNAME`          | `workorder`             | Work order database name      |
| `VIBRATION_DBNAME`   | `vibration`             | Vibration sensor database name |

**WatsonX** — plan-execute runner (when `--model-id` starts with `watsonx/`)

| Variable             | Default                             | Description                 |
| -------------------- | ----------------------------------- | --------------------------- |
| `WATSONX_APIKEY`     | _(required)_                        | IBM WatsonX API key         |
| `WATSONX_PROJECT_ID` | _(required)_                        | IBM WatsonX project ID      |
| `WATSONX_URL`        | `https://us-south.ml.cloud.ibm.com` | WatsonX endpoint (optional) |

**LiteLLM proxy** — used by every runner whenever `--model-id` carries the `litellm_proxy/` prefix (the default for claude-agent, openai-agent, deep-agent)

| Variable           | Default      | Description                                                          |
| ------------------ | ------------ | -------------------------------------------------------------------- |
| `LITELLM_API_KEY`  | _(required)_ | LiteLLM proxy API key                                                |
| `LITELLM_BASE_URL` | _(required)_ | LiteLLM proxy base URL, e.g. `https://your-litellm-host.example.com` |

**Stirrup code track** — `stirrup-agent` with `--code-backend docker`

| Variable             | Default            | Description                                                                |
| -------------------- | ------------------ | -------------------------------------------------------------------------- |
| `STIRRUP_CODE_IMAGE` | `python:3.12-slim` | Docker image for the code sandbox (build `assetops-code` for numpy/pandas/scipy) |
| `DOCKER_HOST`        | *(SDK default)*    | Daemon socket if non-default (e.g. Rancher: `unix:///<home>/.rd/docker.sock`) |

---

## MCP Servers

Six FastMCP servers cover IoT data, time-series ML, work orders, vibration diagnostics, failure-mode reasoning, and utility tools. They speak MCP over stdio and are spawned on-demand by the agent runners — no manual startup needed.

| Server      | Tools | Backing service                        |
| ----------- | ----- | -------------------------------------- |
| `iot`       | 4     | CouchDB                                |
| `utilities` | 3     | none                                   |
| `fmsr`      | 2     | LiteLLM + `failure_modes.yaml`         |
| `wo`        | 8     | CouchDB                                |
| `tsfm`      | 6     | IBM Granite TinyTimeMixer (torch)      |
| `vibration` | 8     | CouchDB + numpy/scipy DSP              |

Tool signatures, required env vars, and how to launch a server directly: **[docs/mcp-servers.md](docs/mcp-servers.md)**.

---

## Example queries

The CLI examples below use a `$query` shell variable so you can swap in any question without editing the commands. Pick one of these to get started:

```bash
# Simple single-server queries
query="What sensors are on Chiller 6?"
query="Is LSTM model supported in TSFM?"
query="Get the work order of equipment CWC04013 for year 2017."

# Multi-step / multi-server queries
query="What is the current date and time? Also list assets at site MAIN. Also get sensor list and failure mode list for any of the chiller at site MAIN."
```

## Agents

Five runners drive the same MCP servers. Each is a CLI registered by `uv sync` that takes a single positional `question` argument and spawns the MCP servers as stdio subprocesses on demand.

| Runner         | Source                       | Loop                                                          | Default model                                               |
| -------------- | ---------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| `plan-execute` | `src/agent/plan_execute/`    | Custom plan → execute → summarise (no SDK)                    | `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8` |
| `claude-agent` | `src/agent/claude_agent/`    | [`claude-agent-sdk`](https://github.com/anthropics/claude-agent-sdk-python) agentic loop | `litellm_proxy/aws/claude-opus-4-6` |
| `openai-agent` | `src/agent/openai_agent/`    | [`openai-agents`](https://github.com/openai/openai-agents-python) SDK Runner | `litellm_proxy/azure/gpt-5.4`                |
| `deep-agent`   | `src/agent/deep_agent/`      | [LangChain deep-agents](https://docs.langchain.com/oss/python/deepagents/overview) (LangGraph), MCP bridged via `langchain-mcp-adapters` | `litellm_proxy/aws/claude-opus-4-6` |
| `stirrup-agent` | `src/agent/stirrup_agent/` | [Stirrup](https://github.com/ArtificialAnalysis/Stirrup) agent loop (in-process), MCP via its `MCPToolProvider`; **code-capable** (writes/runs Python) | `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8` |

- [Agents](#agents) — Stirrup specifics in [docs/stirrup-agent.md](docs/stirrup-agent.md)

### Usage

```bash
uv run plan-execute "$query"
uv run claude-agent "$query"
uv run openai-agent "$query"
uv run deep-agent   "$query"
uv run stirrup-agent "$query"
```

### Common flags

| Flag                  | Description                                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `--model-id MODEL_ID` | Provider-prefixed model string (defaults in the runner table above)                          |
| `--show-trajectory`   | Print each turn / step (text, tool calls, token usage)                                       |
| `--json`              | Emit the trajectory as JSON                                                                  |
| `--verbose`           | Show INFO-level logs on stderr                                                               |
| `--run-id ID`         | Persist the run under this ID (auto-UUID4 if omitted) — see [Observability](#observability)  |
| `--scenario-id ID`    | Tag the run for benchmark grouping                                                           |

### Runner-specific flags

| Flag                  | Runner                     | Description                                                       |
| --------------------- | -------------------------- | ----------------------------------------------------------------- |
| `--show-plan`         | plan-execute               | Print the generated plan before execution                         |
| `--max-turns N`       | claude-agent, openai-agent | Max agentic-loop turns (default: 30)                              |
| `--recursion-limit N` | deep-agent                 | Max LangGraph recursion steps (default: 100)                      |
| `--code-enabled` / `--no-code` | stirrup-agent | Enable (default) / disable code execution — selects the code track |
| `--code-backend B`             | stirrup-agent | Code sandbox: `docker` (default), `local`, or `e2b`                |
| `--max-tokens N`               | stirrup-agent | Max output tokens per call; keep under provider limit (default 16384) |

### Examples

```bash
# Inspect the plan-execute plan before running
uv run plan-execute --show-plan --model-id watsonx/ibm/granite-3-3-8b-instruct "$query"

# Stream a claude-agent run and pipe to jq
uv run claude-agent --json "$query" | jq .turns

# Direct Anthropic API (no proxy) for claude-agent
uv run claude-agent --model-id claude-opus-4-6 "$query"

# Persist a deep-agent run for benchmark evaluation
AGENT_TRAJECTORY_DIR=./traces/trajectories OTEL_TRACES_FILE=./traces/traces.jsonl \
  uv run deep-agent --run-id bench-001 --scenario-id 304 "$query"

# Stirrup tools-only run (comparable to the other runners), native watsonx route
uv run stirrup-agent --no-code --show-trajectory \
  --model-id watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8 "$query"

# Stirrup code track in a Docker sandbox (writes and runs Python)
STIRRUP_CODE_IMAGE=assetops-code \
  uv run stirrup-agent --code-backend docker "$query"
```

---

## Observability

Each agent run can persist two artifacts joined by `run_id`:

- **Trace** — OpenTelemetry root span with metadata + aggregate metrics (runner, model, IDs, span duration, token totals, turn / tool-call counts).
- **Trajectory** — per-run JSON with per-turn content (text, tool inputs/outputs, per-turn tokens and timing).

Install the optional deps and set either / both / neither env var:

```bash
uv sync --group otel

AGENT_TRAJECTORY_DIR=./traces/trajectories \
OTEL_TRACES_FILE=./traces/traces.jsonl \
  uv run deep-agent --run-id bench-001 --scenario-id 304 "$query"
```

`--run-id` (auto-UUID4 if omitted) and `--scenario-id` are available on every runner. With nothing set, runs work normally with zero persistence overhead.

See [docs/observability.md](docs/observability.md) for span attribute reference, trajectory layout, `jq` recipes, log rotation, and optional Jaeger / Collector replay.

---

## Evaluation

Offline scoring of saved trajectories against ground-truth scenarios. Three-stage flow:

```
agent run  →  trajectory (run_id)  →  uv run evaluate  →  reports/<run_id>.json
```

End-to-end against a ground-truth file:

```bash
# 1. Persist trajectories
export AGENT_TRAJECTORY_DIR=$(pwd)/traces/trajectories
uv run claude-agent "List all failure modes of asset Chiller." --scenario-id 101

# 2. Score with LLM-As-Judge
uv run evaluate \
  --trajectories traces/trajectories \
  --scenarios groundtruth/101.json \
  --scorer-default llm_judge \
  --judge-model litellm_proxy/aws/claude-opus-4-6
```

Output lands under `reports/` — one `<run_id>.json` per trajectory plus `_aggregate.json` for the rollup.

Scorer families follow MLflow's evaluator/scorer split: `llm_judge` is wired up; `exact_string_match`, `numeric_match`, and `semantic_similarity` ship as skeletons (raise `NotImplementedError`).

Full reference — scenario schema, report layout, custom scorers, looping over ground-truth: **[docs/evaluation.md](docs/evaluation.md)**.

---

## Running Tests

```bash
uv run pytest src/ -k "not integration"   # unit tests only — no services required
uv run pytest src/                        # full suite — integration tests auto-skip if their service is unavailable
```

Each integration suite is gated by a `skipif` mark; missing service ⇒ silently skipped, not failed:

| Suite              | Skip unless                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| iot, wo, vibration | CouchDB reachable — `docker compose -f src/couchdb/docker-compose.yaml up -d` |
| fmsr               | `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` set in `.env`                          |
| tsfm               | `PATH_TO_MODELS_DIR`, `PATH_TO_DATASETS_DIR` set in `.env`                    |

Narrow scope by path or name pattern:

```bash
uv run pytest src/servers/wo/tests/                # one package's full suite
uv run pytest src/servers/wo/tests/test_integration.py -v   # one file
uv run pytest src/ -k "integration"                # only files / tests with "integration" in the name
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          agent/                              │
│                                                              │
│   PlanExecuteRunner   ClaudeAgentRunner   StirrupAgentRunner │
│   OpenAIAgentRunner   DeepAgentRunner                        │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │ MCP protocol (stdio)
         ┌─────────────────┼───────────┬──────────┬──────┬───────────┐
         ▼                 ▼           ▼          ▼      ▼           ▼
        iot           utilities      fmsr       tsfm    wo      vibration
      (tools)          (tools)      (tools)   (tools) (tools)    (tools)
```
