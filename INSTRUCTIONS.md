# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [MCP Servers](#mcp-servers)
  - [IoTAgent](#iotagent)
  - [Utilities](#utilities)
  - [FMSRAgent](#fmsragent)
- [Plan-Execute Runner](#plan-execute-runner)
  - [How it works](#how-it-works)
  - [CLI](#cli)
  - [Python API](#python-api)
  - [Bring your own LLM](#bring-your-own-llm)
  - [Add more MCP servers](#add-more-mcp-servers)
- [Connect to Claude Desktop](#connect-to-claude-desktop)
- [Running Tests](#running-tests)
- [Architecture](#architecture)

---

## Prerequisites

- **Python 3.14+** — required by `pyproject.toml`
- **[uv](https://docs.astral.sh/uv/)** — dependency and environment manager
- **Docker** — for running CouchDB (IoT data store)

## Quick Start

### 1. Install dependencies

Run from the **repo root**:

```bash
uv sync
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
docker compose -f mcp/couchdb/docker-compose.yaml up -d
```

Verify CouchDB is running:

```bash
curl -X GET http://localhost:5984/
```

### 4. Run servers locally

Use `uv run` to start the MCP servers (paths relative to repo root):

```bash
uv run python mcp/servers/utilities/main.py
uv run python mcp/servers/iot/main.py
uv run python mcp/servers/fmsr/main.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `COUCHDB_URL` | IoT server | CouchDB connection URL, e.g. `http://localhost:5984` |
| `COUCHDB_DBNAME` | IoT server | Database name (default fixture: `chiller`) |
| `COUCHDB_USERNAME` | IoT server | CouchDB admin username |
| `COUCHDB_PASSWORD` | IoT server | CouchDB admin password |
| `WATSONX_APIKEY` | `--platform watsonx` | IBM WatsonX API key |
| `WATSONX_PROJECT_ID` | `--platform watsonx` | IBM WatsonX project ID |
| `WATSONX_URL` | `--platform watsonx` | WatsonX endpoint (optional; defaults to `https://us-south.ml.cloud.ibm.com`) |
| `LITELLM_API_KEY` | `--platform litellm` | LiteLLM API key |
| `LITELLM_BASE_URL` | `--platform litellm` | LiteLLM base URL (e.g. `https://your-litellm-host.example.com`) |

---

## MCP Servers

### IoTAgent

**Path:** `mcp/servers/iot/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_DBNAME`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`)

| Tool | Arguments | Description |
|---|---|---|
| `sites` | — | List all available sites |
| `assets` | `site_name` | List all asset IDs for a site |
| `sensors` | `site_name`, `asset_id` | List sensor names for an asset |
| `history` | `site_name`, `asset_id`, `start`, `final?` | Fetch historical sensor readings for a time range (ISO 8601 timestamps) |

### Utilities

**Path:** `mcp/servers/utilities/main.py`
**Requires:** nothing (no external services)

| Tool | Arguments | Description |
|---|---|---|
| `json_reader` | `file_name` | Read and parse a JSON file from disk |
| `current_date_time` | — | Return the current UTC date and time as JSON |
| `current_time_english` | — | Return the current UTC time as a human-readable string |

### FMSRAgent

**Path:** `mcp/servers/fmsr/main.py`
**Requires:** `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` for unknown assets; curated lists for `chiller` and `ahu` work without credentials.
**Failure-mode data:** `mcp/servers/fmsr/failure_modes.yaml` (edit to add/change asset entries)

| Tool | Arguments | Description |
|---|---|---|
| `get_failure_modes` | `asset_name` | Return known failure modes for an asset. Uses a curated YAML list for chillers and AHUs; falls back to the LLM for other types. |
| `get_failure_mode_sensor_mapping` | `asset_name`, `failure_modes`, `sensors` | For each (failure mode, sensor) pair, determine relevancy via LLM. Returns bidirectional `fm→sensors` and `sensor→fms` maps plus full per-pair details. |

---

## Plan-Execute Runner

`mcp/plan_execute/` is a custom MCP client that implements a **plan-and-execute** workflow over the MCP servers. It replaces AgentHive's bespoke orchestration with the standard MCP protocol.

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
plan-execute "What assets are available at site MAIN?"
```

Flags:

| Flag | Description |
|---|---|
| `--model-id MODEL_ID` | litellm model string with provider prefix (default: `watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8`) |
| `--server NAME=PATH` | Override MCP servers with `NAME=PATH` pairs (repeatable) |
| `--show-plan` | Print the generated plan before execution |
| `--show-history` | Print each step result after execution |
| `--json` | Output answer + plan + history as JSON |

The provider is encoded in the `--model-id` prefix:

| Prefix | Provider | Required env vars |
|---|---|---|
| `watsonx/` | IBM WatsonX | `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` (optional) |
| `litellm_proxy/` | LiteLLM proxy | `LITELLM_API_KEY`, `LITELLM_BASE_URL` |

Examples:

```bash
# WatsonX — default model
plan-execute "What assets are at site MAIN?"

# WatsonX — different model, inspect the plan
plan-execute --model-id watsonx/ibm/granite-3-3-8b-instruct --show-plan "List sensors for asset CH-1"

# LiteLLM proxy
plan-execute --model-id litellm_proxy/GCP/claude-4-sonnet "What are the failure modes for a chiller?"

# Machine-readable output
plan-execute --show-history --json "How many observations exist for CH-1?" | jq .answer
```

### Three-server end-to-end example

All three servers (IoTAgent, Utilities, FMSRAgent) are registered by default.
Run a question that exercises all three with independent parallel steps:

```bash
plan-execute --show-plan --show-history \
  "What is the current date and time? Also list assets at site MAIN. Also get failure modes for a chiller."
```

Expected plan (3 parallel steps, no dependencies):

```
[1] Utilities  : current_date_time()
[2] IoTAgent   : assets(site_name="MAIN")
[3] FMSRAgent  : get_failure_modes(asset_name="chiller")
```

Expected execution output (trimmed):

```
[OK] Step 1 (Utilities)
     {"currentDateTime": "2026-02-20T17:28:39Z", "currentDateTimeDescription": "Today's date is 2026-02-20 and time is 17:28:39."}

[OK] Step 2 (IoTAgent)
     {"site_name": "MAIN", "total_assets": 1, "assets": ["Chiller 6"], "message": "found 1 assets for site_name MAIN."}

[OK] Step 3 (FMSRAgent)
     {"asset_name": "chiller", "failure_modes": ["Compressor Overheating: Failed due to Normal wear, overheating", ...]}
```

> **Note:** Curated assets (`chiller`, `ahu`) are served from `failure_modes.yaml` without any LLM call.

### Python API

```python
import asyncio
from plan_execute import PlanExecuteRunner
from llm import LiteLLMBackend

runner = PlanExecuteRunner(llm=LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct"))
result = asyncio.run(runner.run("What assets are available at site MAIN?"))
print(result.answer)
```

`OrchestratorResult` fields:

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Final synthesised answer |
| `plan` | `Plan` | The generated plan with its steps |
| `history` | `list[StepResult]` | Per-step execution results |

### Bring your own LLM

Implement `LLMBackend` to use any model:

```python
from plan_execute.llm import LLMBackend

class MyLLM(LLMBackend):
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        ...  # call your model here

runner = PlanExecuteRunner(llm=MyLLM())
```

### Add more MCP servers

Pass `server_paths` to register additional servers. Keys must match the agent names the planner assigns steps to:

```python
from pathlib import Path
from plan_execute import PlanExecuteRunner

runner = PlanExecuteRunner(
    llm=my_llm,
    server_paths={
        "IoTAgent":  Path("mcp/servers/iot/main.py"),
        "Utilities": Path("mcp/servers/utilities/main.py"),
        "FMSRAgent": Path("mcp/servers/fmsr/main.py"),
    },
)
```

> **Note:** passing `server_paths` replaces the defaults entirely. Include all servers you need.

---

## Connect to Claude Desktop

Add the following to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "utilities": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project",
        "/path/to/AssetOpsBench",
        "python",
        "/path/to/AssetOpsBench/mcp/servers/utilities/main.py"
      ]
    },
    "IoTAgent": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project",
        "/path/to/AssetOpsBench",
        "python",
        "/path/to/AssetOpsBench/mcp/servers/iot/main.py"
      ]
    }
  }
}
```

---

## Running Tests

Run the full suite from the repo root (unit + integration where services are available):

```bash
uv run pytest mcp/ -v
```

Integration tests are auto-skipped when the required service is not available:
- IoT integration tests require `COUCHDB_URL` (set in `.env`)
- FMSR integration tests require `WATSONX_APIKEY` (set in `.env`)

### Unit tests only (no services required)

```bash
uv run pytest mcp/ -v -k "not integration"
```

### Per-server

```bash
uv run pytest mcp/servers/iot/tests/test_tools.py -k "not integration"
uv run pytest mcp/servers/utilities/tests/
uv run pytest mcp/servers/fmsr/tests/ -k "not integration"
uv run pytest mcp/plan_execute/tests/
```

### Integration tests (requires CouchDB + WatsonX)

```bash
docker compose -f mcp/couchdb/docker-compose.yaml up -d
uv run pytest mcp/ -v
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   plan_execute/                      │
│                                                      │
│  PlanExecuteRunner.run(question)                     │
│  ┌────────────┐   ┌────────────┐   ┌──────────────┐ │
│  │  Planner   │ → │  Executor  │ → │  Summariser  │ │
│  │            │   │            │   │              │ │
│  │ LLM breaks │   │ Routes each│   │ LLM combines │ │
│  │ question   │   │ step to the│   │ step results │ │
│  │ into steps │   │ right MCP  │   │ into answer  │ │
│  └────────────┘   │ server via │   └──────────────┘ │
│                   │ stdio      │                     │
└───────────────────┼────────────┼─────────────────────┘
                    │ MCP protocol (stdio)
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      IoTAgent   Utilities   FMSRAgent
      (tools)    (tools)     (tools)
```
