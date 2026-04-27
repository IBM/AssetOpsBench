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
| `IOT_DBNAME`         | `chiller`               | IoT sensor database name      |
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

Four runners drive the same MCP servers. Each is a CLI registered by `uv sync` that takes a single positional `question` argument and spawns the MCP servers as stdio subprocesses on demand.

| Runner         | Source                       | Loop                                                          | Default model                                               |
| -------------- | ---------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| `plan-execute` | `src/agent/plan_execute/`    | Custom plan → execute → summarise (no SDK)                    | `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8` |
| `claude-agent` | `src/agent/claude_agent/`    | [`claude-agent-sdk`](https://github.com/anthropics/claude-agent-sdk-python) agentic loop | `litellm_proxy/aws/claude-opus-4-6` |
| `openai-agent` | `src/agent/openai_agent/`    | [`openai-agents`](https://github.com/openai/openai-agents-python) SDK Runner | `litellm_proxy/azure/gpt-5.4`                |
| `deep-agent`   | `src/agent/deep_agent/`      | [LangChain deep-agents](https://docs.langchain.com/oss/python/deepagents/overview) (LangGraph), MCP bridged via `langchain-mcp-adapters` | `litellm_proxy/aws/claude-opus-4-6` |

### Usage

```bash
uv run plan-execute "$query"
uv run claude-agent "$query"
uv run openai-agent "$query"
uv run deep-agent   "$query"
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

## Running Tests

Run the full suite from the repo root (unit + integration where services are available):

```bash
uv run pytest src/ -v
```

Integration tests are auto-skipped when the required service is not available:

- IoT integration tests require `COUCHDB_URL` (set in `.env`)
- Work order integration tests require `COUCHDB_URL` (set in `.env`)
- FMSR integration tests require `WATSONX_APIKEY` (set in `.env`)
- TSFM integration tests require `PATH_TO_MODELS_DIR` and `PATH_TO_DATASETS_DIR` (set in `.env`)

### Unit tests only (no services required)

```bash
uv run pytest src/ -v -k "not integration"
```

### Per-server

```bash
uv run pytest src/servers/iot/tests/test_tools.py -k "not integration"
uv run pytest src/servers/utilities/tests/
uv run pytest src/servers/fmsr/tests/ -k "not integration"
uv run pytest src/servers/tsfm/tests/ -k "not integration"
uv run pytest src/servers/wo/tests/test_tools.py -k "not integration"
uv run pytest src/agent/tests/
```

### Work order integration tests (requires CouchDB + populated `workorder` db)

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
uv run pytest src/servers/wo/tests/test_integration.py -v
```

### Integration tests (requires CouchDB + WatsonX)

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
uv run pytest src/ -v
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                          agent/                              │
│                                                              │
│   PlanExecuteRunner   ClaudeAgentRunner                      │
│   OpenAIAgentRunner   DeepAgentRunner                        │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │ MCP protocol (stdio)
         ┌─────────────────┼───────────┬──────────┬──────┬───────────┐
         ▼                 ▼           ▼          ▼      ▼           ▼
        iot           utilities      fmsr       tsfm    wo      vibration
      (tools)          (tools)      (tools)   (tools) (tools)    (tools)
```
