# AssetOpsBench MCP Environment

This directory contains the MCP servers and infrastructure for the AssetOpsBench project.

## Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [MCP Servers](#mcp-servers)
  - [iot](#iot)
  - [utilities](#utilities)
  - [fmsr](#fmsr)
  - [tsfm](#tsfm)
  - [wo](#wo)
  - [vibration](#vibration)
- [Example queries](#example-queries)
- [Plan-Execute Agent](#plan-execute-agent)
  - [How it works](#how-it-works)
  - [CLI](#cli)
- [Claude Agent](#claude-agent)
  - [How it works](#how-it-works-1)
  - [CLI](#cli-1)
- [OpenAI Agent](#openai-agent)
  - [How it works](#how-it-works-2)
  - [CLI](#cli-2)
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

### 4. Run servers

> **Note:** MCP servers use stdio transport — they are spawned on-demand by clients (Claude Desktop, `plan-execute`) and exit when the client disconnects. They are not long-running daemons.

To start a server manually for testing:

```bash
uv run utilities-mcp-server
uv run iot-mcp-server
uv run fmsr-mcp-server
uv run tsfm-mcp-server
uv run wo-mcp-server
uv run vibration-mcp-server
```

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

**LiteLLM** — plan-execute runner (when `--model-id` does not start with `watsonx/`, e.g. `litellm_proxy/…`)

| Variable           | Default      | Description                                                          |
| ------------------ | ------------ | -------------------------------------------------------------------- |
| `LITELLM_API_KEY`  | _(required)_ | LiteLLM proxy API key                                                |
| `LITELLM_BASE_URL` | _(required)_ | LiteLLM proxy base URL, e.g. `https://your-litellm-host.example.com` |

**OpenAI Agents SDK** — openai-agent runner (always routed through LiteLLM proxy via `litellm_proxy/` prefix)

> Uses the same `LITELLM_API_KEY` and `LITELLM_BASE_URL` variables as the plan-execute runner above.

---

## MCP Servers

### iot — IoT Sensor Data

**Path:** `src/servers/iot/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `IOT_DBNAME`)

| Tool      | Arguments                                  | Description                                                             |
| --------- | ------------------------------------------ | ----------------------------------------------------------------------- |
| `sites`   | —                                          | List all available sites                                                |
| `assets`  | `site_name`                                | List all asset IDs for a site                                           |
| `sensors` | `site_name`, `asset_id`                    | List sensor names for an asset                                          |
| `history` | `site_name`, `asset_id`, `start`, `final?` | Fetch historical sensor readings for a time range (ISO 8601 timestamps) |

### utilities — Utilities

**Path:** `src/servers/utilities/main.py`
**Requires:** nothing (no external services)

| Tool                   | Arguments   | Description                                            |
| ---------------------- | ----------- | ------------------------------------------------------ |
| `json_reader`          | `file_name` | Read and parse a JSON file from disk                   |
| `current_date_time`    | —           | Return the current UTC date and time as JSON           |
| `current_time_english` | —           | Return the current UTC time as a human-readable string |

### fmsr — Failure Mode and Sensor Relations

**Path:** `src/servers/fmsr/main.py`
**Requires:** `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` for unknown assets; curated lists for `chiller` and `ahu` work without credentials.
**Failure-mode data:** `src/servers/fmsr/failure_modes.yaml` (edit to add/change asset entries)

| Tool                              | Arguments                                | Description                                                                                                                                             |
| --------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_failure_modes`               | `asset_name`                             | Return known failure modes for an asset. Uses a curated YAML list for chillers and AHUs; falls back to the LLM for other types.                         |
| `get_failure_mode_sensor_mapping` | `asset_name`, `failure_modes`, `sensors` | For each (failure mode, sensor) pair, determine relevancy via LLM. Returns bidirectional `fm→sensors` and `sensor→fms` maps plus full per-pair details. |

### wo — Work Order

**Path:** `src/servers/wo/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `WO_DBNAME`)
**Data init:** Handled automatically by `docker compose -f src/couchdb/docker-compose.yaml up` (runs `src/couchdb/init_wo.py` inside the CouchDB container on every start — database is dropped and reloaded each time)

| Tool                          | Arguments                                             | Description                                                                              |
| ----------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `get_work_orders`             | `equipment_id`, `start_date?`, `end_date?`            | Retrieve all work orders for an equipment within an optional date range                  |
| `get_preventive_work_orders`  | `equipment_id`, `start_date?`, `end_date?`            | Retrieve only preventive (PM) work orders                                                |
| `get_corrective_work_orders`  | `equipment_id`, `start_date?`, `end_date?`            | Retrieve only corrective (CM) work orders                                                |
| `get_events`                  | `equipment_id`, `start_date?`, `end_date?`            | Retrieve all events (work orders, alerts, anomalies)                                     |
| `get_failure_codes`           | —                                                     | List all failure codes with categories and descriptions                                  |
| `get_work_order_distribution` | `equipment_id`, `start_date?`, `end_date?`            | Count work orders per (primary, secondary) failure code pair, sorted by frequency        |
| `predict_next_work_order`     | `equipment_id`, `start_date?`, `end_date?`            | Predict next work order type via Markov transition matrix built from historical sequence |
| `analyze_alert_to_failure`    | `equipment_id`, `rule_id`, `start_date?`, `end_date?` | Probability that an alert rule leads to a work order; average hours to maintenance       |

### tsfm — Time Series Foundation Model

**Path:** `src/servers/tsfm/main.py`
**Requires:** `tsfm_public` (IBM Granite TSFM), `transformers`, `torch` for ML tools — imported lazily; static tools work without them.
**Model checkpoints:** resolved relative to `PATH_TO_MODELS_DIR` (default: `src/servers/tsfm/artifacts/output/tuned_models`)

| Tool                   | Arguments                                                                                                                   | Description                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `get_ai_tasks`         | —                                                                                                                           | List supported AI task types for time-series analysis                                            |
| `get_tsfm_models`      | —                                                                                                                           | List available pre-trained TinyTimeMixer (TTM) model checkpoints                                 |
| `run_tsfm_forecasting` | `dataset_path`, `timestamp_column`, `target_columns`, `model_checkpoint?`, `forecast_horizon?`, `frequency_sampling?`, ...  | Zero-shot TTM inference; returns path to a JSON predictions file                                 |
| `run_tsfm_finetuning`  | `dataset_path`, `timestamp_column`, `target_columns`, `model_checkpoint?`, `save_model_dir?`, `n_finetune?`, `n_test?`, ... | Few-shot fine-tune a TTM model; returns saved checkpoint path and metrics file                   |
| `run_tsad`             | `dataset_path`, `tsfm_output_json`, `timestamp_column`, `target_columns`, `task?`, `false_alarm?`, `ad_model_type?`, ...    | Conformal anomaly detection on top of a forecasting output JSON; returns CSV with anomaly labels |
| `run_integrated_tsad`  | `dataset_path`, `timestamp_column`, `target_columns`, `model_checkpoint?`, `false_alarm?`, `n_calibration?`, ...            | End-to-end forecasting + anomaly detection in one call; returns combined CSV                     |

### vibration — Vibration Diagnostics

**Path:** `src/servers/vibration/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `VIBRATION_DBNAME` (default `vibration`), `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`); `numpy`, `scipy`
**DSP core:** `src/servers/vibration/dsp/` — adapted from [vibration-analysis-mcp](https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp) (Apache-2.0)

| Tool | Arguments | Description |
|---|---|---|
| `get_vibration_data` | `site_name`, `asset_id`, `sensor_name`, `start`, `final?` | Fetch vibration time-series from CouchDB and load into the analysis store. Returns a `data_id`. |
| `list_vibration_sensors` | `site_name`, `asset_id` | List available sensor fields for an asset. |
| `compute_fft_spectrum` | `data_id`, `window?`, `top_n?` | Compute FFT amplitude spectrum (top-N peaks + statistics). |
| `compute_envelope_spectrum` | `data_id`, `band_low_hz?`, `band_high_hz?`, `top_n?` | Compute envelope spectrum for bearing fault detection (Hilbert transform). |
| `assess_vibration_severity` | `rms_velocity_mm_s`, `machine_group?` | Classify vibration severity per ISO 10816 (Zones A–D). |
| `calculate_bearing_frequencies` | `rpm`, `n_balls`, `ball_diameter_mm`, `pitch_diameter_mm`, `contact_angle_deg?`, `bearing_name?` | Compute bearing characteristic frequencies (BPFO, BPFI, BSF, FTF). |
| `list_known_bearings` | — | List all bearings in the built-in database. |
| `diagnose_vibration` | `data_id`, `rpm?`, `bearing_designation?`, `bearing_*?`, `bpfo_hz?`, `bpfi_hz?`, `bsf_hz?`, `ftf_hz?`, `machine_group?`, `machine_description?` | Full automated diagnosis: FFT + shaft features + bearing envelope + ISO 10816 + fault classification + markdown report. |

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
| `--model-id MODEL_ID` | Claude model ID (default: `claude-opus-4-6`)                                 |
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
# Direct Anthropic API
uv run claude-agent "$query"

# LiteLLM proxy
uv run claude-agent --model-id litellm_proxy/aws/claude-opus-4-6 "$query"

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
└──────────────────────────┬───────────────────────────────────┘
                           │ MCP protocol (stdio)
         ┌─────────────────┼───────────┬──────────┬──────┬───────────┐
         ▼                 ▼           ▼          ▼      ▼           ▼
        iot           utilities      fmsr       tsfm    wo      vibration
      (tools)          (tools)      (tools)   (tools) (tools)    (tools)
```
