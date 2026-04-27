# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [MCP Servers](#mcp-servers) — full reference in [docs/mcp-servers.md](docs/mcp-servers.md)
- [Example queries](#example-queries)
- [Plan-Execute Agent](#plan-execute-agent)
- [Claude Agent](#claude-agent)
- [OpenAI Agent](#openai-agent)
- [Deep Agent](#deep-agent)
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

## Plan-Execute Agent

`src/agent/` is a custom MCP client that implements a **plan-and-execute** workflow over the MCP servers. It replaces AgentHive's bespoke orchestration with the standard MCP protocol.

### How it works

```
PlanExecuteRunner.run(question)
  │
  ├─ 1. Discover   query each MCP server for its available tools
  │
  ├─ 2. Plan       LLM decomposes the question into ordered steps,
  │                each assigned to an MCP server
  │
  ├─ 3. Execute    for each step (in dependency order):
  │                  • LLM selects the right tool + generates arguments
  │                  • tool is called via MCP stdio protocol
  │                  • result is stored and passed as context to later steps
  │
  └─ 4. Summarise  LLM synthesises step results into a final answer
```

### CLI

After `uv sync`, the `plan-execute` command is available:

```bash
uv run plan-execute "$query"
```

> **Note:** `plan-execute` spawns MCP servers on-demand for each query — you do **not** need to start them manually first. Servers are launched as subprocesses, used, then exit automatically.

Flags:

| Flag                  | Description                                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--model-id MODEL_ID` | litellm model string with provider prefix (default: `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8`) |
| `--server NAME=SPEC`  | Override MCP servers with `NAME=SPEC` pairs (repeatable); SPEC is an entry-point name or path                    |
| `--show-plan`         | Print the generated plan before execution                                                                        |
| `--show-trajectory`   | Print each step result after execution                                                                           |
| `--json`              | Output answer + plan + trajectory as JSON                                                                           |

The provider is encoded in the `--model-id` prefix:

| Prefix           | Provider      | Required env vars                                                |
| ---------------- | ------------- | ---------------------------------------------------------------- |
| `watsonx/`       | IBM WatsonX   | `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` (optional) |
| `litellm_proxy/` | LiteLLM proxy | `LITELLM_API_KEY`, `LITELLM_BASE_URL`                            |

Examples:

```bash
# WatsonX — default model
uv run plan-execute "$query"

# WatsonX — different model, inspect the plan
uv run plan-execute --model-id watsonx/ibm/granite-3-3-8b-instruct --show-plan "$query"

# LiteLLM proxy
uv run plan-execute --model-id litellm_proxy/GCP/claude-4-sonnet "$query"

# Machine-readable output
uv run plan-execute --show-trajectory --json "$query" | jq .answer
```

---

## Claude Agent

`src/agent/claude_agent/` uses the **claude-agent-sdk** to drive the same MCP servers. Unlike `PlanExecuteRunner`, there is no explicit plan — the SDK's built-in agentic loop handles tool discovery, invocation, and multi-turn reasoning autonomously.

### How it works

```
ClaudeAgentRunner.run(question)
  │
  └─ claude-agent-sdk query loop
       • connects to each MCP server over stdio
       • Claude decides which tools to call and in what order
       • tool calls and results are handled internally by the SDK
       • final answer is returned as ResultMessage
```

### CLI

After `uv sync`, the `claude-agent` command is available:

```bash
uv run claude-agent "$query"
```

Flags:

| Flag                  | Description                                                                  |
| --------------------- | ---------------------------------------------------------------------------- |
| `--model-id MODEL_ID` | Claude model ID (default: `litellm_proxy/aws/claude-opus-4-6`)               |
| `--max-turns N`       | Maximum agentic loop turns (default: 30)                                     |
| `--show-trajectory`      | Print each turn's text, tool calls, and token usage                          |
| `--json`              | Output full trajectory (turns, tool calls, token counts) as JSON             |
| `--verbose`           | Show INFO-level logs on stderr                                               |

The `--model-id` prefix determines the backend:

| Prefix           | Backend       | Required env vars                     |
| ---------------- | ------------- | ------------------------------------- |
| _(none)_         | Anthropic API | `LITELLM_API_KEY`                     |
| `litellm_proxy/` | LiteLLM proxy | `LITELLM_API_KEY`, `LITELLM_BASE_URL` |

Examples:

```bash
# LiteLLM proxy (default)
uv run claude-agent "$query"

# Direct Anthropic API
uv run claude-agent --model-id claude-opus-4-6 "$query"

# Show full trajectory (turns, tool calls, token usage)
uv run claude-agent --show-trajectory "$query"

# Machine-readable trajectory
uv run claude-agent --json "$query" | jq .turns
```

---

## OpenAI Agent

`src/agent/openai_agent/` uses the **[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)** (`openai-agents`) to drive the same MCP servers. Like `ClaudeAgentRunner`, there is no explicit plan — the SDK's built-in agentic loop handles tool discovery, invocation, and multi-turn reasoning autonomously.

### How it works

```
OpenAIAgentRunner.run(question)
  │
  └─ OpenAI Agents SDK Runner.run loop
       • connects to each MCP server over stdio (MCPServerStdio)
       • GPT decides which tools to call and in what order
       • tool calls and results are handled internally by the SDK
       • final answer is returned via result.final_output
```

### CLI

After `uv sync`, the `openai-agent` command is available:

```bash
uv run openai-agent "$query"
```

Flags:

| Flag                  | Description                                                                      |
| --------------------- | -------------------------------------------------------------------------------- |
| `--model-id MODEL_ID` | LiteLLM model string with `litellm_proxy/` prefix (default: `litellm_proxy/azure/gpt-5.4`) |
| `--max-turns N`       | Maximum agentic loop turns (default: 30)                                         |
| `--show-trajectory`   | Print each turn's text, tool calls, and token usage                              |
| `--json`              | Output full trajectory (turns, tool calls, token counts) as JSON                 |
| `--verbose`           | Show INFO-level logs on stderr                                                   |

Required env vars: `LITELLM_API_KEY`, `LITELLM_BASE_URL`

Examples:

```bash
uv run openai-agent --model-id litellm_proxy/azure/gpt-5.4 "$query"

# Show full trajectory (turns, tool calls, token usage)
uv run openai-agent --model-id litellm_proxy/azure/gpt-5.4 --show-trajectory "$query"

# Machine-readable trajectory
uv run openai-agent --model-id litellm_proxy/azure/gpt-5.4 --json "$query" | jq .turns
```

---

## Deep Agent

`src/agent/deep_agent/` uses the **[LangChain deep-agents](https://docs.langchain.com/oss/python/deepagents/overview)** framework to drive the same MCP servers. The deep agent ships with built-in planning (`write_todos`), a virtual filesystem, and sub-agent delegation. MCP servers are bridged to LangChain tools via `langchain-mcp-adapters`.

### How it works

```
DeepAgentRunner.run(question)
  │
  └─ deep-agents agentic loop (LangGraph under the hood)
       • MultiServerMCPClient exposes each MCP server's tools as LangChain tools
       • the deep agent plans, calls tools, and writes to its virtual filesystem
       • final answer is the content of the last AIMessage
```

### CLI

After `uv sync`, the `deep-agent` command is available:

```bash
uv run deep-agent "$query"
```

Flags:

| Flag                   | Description                                                                      |
| ---------------------- | -------------------------------------------------------------------------------- |
| `--model-id MODEL_ID`  | LiteLLM-prefixed or native provider model string (default: `litellm_proxy/aws/claude-opus-4-6`) |
| `--recursion-limit N`  | Maximum graph recursion steps (default: 100)                                     |
| `--show-trajectory`    | Print each turn's text, tool calls, and token usage                              |
| `--json`               | Output full trajectory (turns, tool calls, token counts) as JSON                 |
| `--verbose`            | Show INFO-level logs on stderr                                                   |

Required env vars (for `litellm_proxy/*` models): `LITELLM_API_KEY`, `LITELLM_BASE_URL`

Examples:

```bash
uv run deep-agent --model-id litellm_proxy/aws/claude-opus-4-6 "$query"

# Show full trajectory (turns, tool calls, token usage)
uv run deep-agent --show-trajectory "$query"

# Machine-readable trajectory
uv run deep-agent --json "$query" | jq .turns
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
│  PlanExecuteRunner.run(question)                             │
│  ┌────────────┐   ┌────────────┐   ┌──────────────┐         │
│  │  Planner   │ → │  Executor  │ → │  Summariser  │         │
│  │ LLM breaks │   │ Routes each│   │ LLM combines │         │
│  │ question   │   │ step to MCP│   │ step results │         │
│  │ into steps │   │ via stdio  │   │ into answer  │         │
│  └────────────┘   └────────────┘   └──────────────┘         │
│                                                              │
│  ClaudeAgentRunner.run(question)                             │
│  ┌─────────────────────────────────────────┐                 │
│  │  claude-agent-sdk agentic loop          │                 │
│  │  Claude decides tools + order autonomously               │
│  │  Trajectory (turns, tool calls, tokens) collected        │
│  └─────────────────────────────────────────┘                 │
│                                                              │
│  OpenAIAgentRunner.run(question)                             │
│  ┌─────────────────────────────────────────┐                 │
│  │  openai-agents SDK Runner.run loop      │                 │
│  │  GPT decides tools + order autonomously                  │
│  │  Trajectory (turns, tool calls, tokens) collected        │
│  └─────────────────────────────────────────┘                 │
│                                                              │
│  DeepAgentRunner.run(question)                               │
│  ┌─────────────────────────────────────────┐                 │
│  │  LangChain deep-agents LangGraph loop   │                 │
│  │  MCP tools bridged via langchain-mcp-adapters            │
│  │  Trajectory (turns, tool calls, tokens) collected        │
│  └─────────────────────────────────────────┘                 │
└──────────────────────────┬───────────────────────────────────┘
                           │ MCP protocol (stdio)
         ┌─────────────────┼───────────┬──────────┬──────┬───────────┐
         ▼                 ▼           ▼          ▼      ▼           ▼
        iot           utilities      fmsr       tsfm    wo      vibration
      (tools)          (tools)      (tools)   (tools) (tools)    (tools)
```
