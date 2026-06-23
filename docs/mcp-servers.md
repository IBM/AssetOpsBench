# MCP Servers

Six FastMCP servers expose the AssetOpsBench domain logic. Each is a standalone stdio process spawned on-demand by clients (`plan-execute`, `claude-agent`, `openai-agent`, `deep-agent`, Claude Desktop). Backing services and credentials are listed per-server below.

## Contents

- [iot — IoT Sensor Data](#iot--iot-sensor-data)
- [utilities — Utilities](#utilities--utilities)
- [fmsr — Failure Mode and Sensor Relations](#fmsr--failure-mode-and-sensor-relations)
- [wo — Work Order](#wo--work-order)
- [tsfm — Time Series Foundation Model](#tsfm--time-series-foundation-model)
- [vibration — Vibration Diagnostics](#vibration--vibration-diagnostics)

## iot — IoT Sensor Data

The IoT server reads from **two** databases: telemetry readings (`IOT_DBNAME`, default `iot`) and an
asset **registry** (`ASSET_DBNAME`, default `asset`, loaded from `asset_profile_sample.json`). The
two answer different questions: `assets()`/`sensors()` reflect TELEMETRY — what actually streams (the
**measured** set); `get_asset()`/`asset_sensors()`/`registry_assets()` reflect the REGISTRY — the
asset nameplate and the **installed** sensor inventory (by name). Comparing `asset_sensors()` against
`sensors()` surfaces sensors that are installed but not streaming. The registry also reconciles ids
across systems (Maximo `assetnum`, telemetry `iot_asset_id`, work-order `wo_assetnum`), so an asset
can be looked up by any of its ids.

**Path:** `src/servers/iot/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `IOT_DBNAME`, `ASSET_DBNAME`)

**Sample assets shipped in the `iot` database** (loaded by `src/couchdb/couchdb_setup.sh`):

| `asset_id`  | Asset class      | Source file                                       |
| ----------- | ---------------- | ------------------------------------------------- |
| `Chiller 6` | Chiller          | `src/couchdb/sample_data/iot/chiller_6.json`         |
| `mp_1`      | Metro pump       | `src/couchdb/sample_data/iot/metro_pump_1.json`      |
| `hyd_1`     | Hydraulic pump   | `src/couchdb/sample_data/iot/hydraulic_pump_1.json`  |

Synthetic motor vibration data (`asset_id: Motor_01`, from `motor_01.json`) ships in a separate `vibration` database for the vibration MCP server.

| Tool              | Arguments                                  | Description                                                                                                  |
| ----------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `sites`           | —                                          | List all sites, discovered dynamically from the asset registry (`siteid`)                                    |
| `assets`          | `site_name`                                | List asset ids registered at a site (telemetry id where present, else `assetnum`)                            |
| `sensors`         | `site_name`, `asset_id`                    | List **measured** sensor names for an asset (union of keys across its telemetry docs)                        |
| `history`         | `site_name`, `asset_id`, `start`, `final?` | Fetch historical sensor readings for a time range (ISO 8601 timestamps)                                      |
| `get_asset`       | `site_name`, `asset_id`                    | Registry/nameplate detail for one asset (description, assettype, status, location, vintage, installed count) |
| `asset_sensors`   | `site_name`, `asset_id`                    | List the **installed** sensors for an asset, by name (registry inventory)                                    |
| `registry_assets` | `site_name`, `assettype?`                  | List registry assets with metadata (assettype, vintage, sensor count), optionally filtered by assettype     |

## utilities — Utilities

**Path:** `src/servers/utilities/main.py`
**Requires:** nothing (no external services)

| Tool                   | Category | Arguments   | Description                                            |
| ---------------------- | -------- | ----------- | ------------------------------------------------------ |
| `json_reader`          | read     | `file_name` | Read and parse a JSON file from disk                   |
| `current_date_time`    | read     | —           | Return the current UTC date and time as JSON           |
| `current_time_english` | read     | —           | Return the current UTC time as a human-readable string |

## fmsr — Failure Mode and Sensor Relations

**Path:** `src/servers/fmsr/main.py`
**Requires:** `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` for unknown assets; curated lists for `chiller` and `ahu` work without credentials.
**Failure-mode data:** `src/servers/fmsr/failure_modes.yaml` (edit to add/change asset entries)

| Tool                              | Category      | Arguments                                | Description                                                                                                                                             |
| --------------------------------- | ------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_failure_modes`               | read, LLM-use | `asset_name`                             | Return known failure modes for an asset. Uses a curated YAML list for chillers and AHUs; falls back to the LLM for other types.                         |
| `get_failure_mode_sensor_mapping` | read, LLM-use | `asset_name`, `failure_modes`, `sensors` | For each (failure mode, sensor) pair, determine relevancy via LLM. Returns bidirectional `fm→sensors` and `sensor→fms` maps plus full per-pair details. |

## wo — Work Order

**Path:** `src/servers/wo/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `WO_DBNAME`)
**Data init:** Handled automatically by `docker compose -f src/couchdb/docker-compose.yaml up` (runs `src/couchdb/init_wo.py` inside the CouchDB container on every start — database is dropped and reloaded each time)

Tools fall into several categories: **read**, **write**, **LLM-use**, and **CPU-centric**. Tools are registered centrally in `main.py`; set `AOB_READONLY=1` to expose only the read tools (8). The default exposes all 14 (8 read + 6 write).

### Read tools

| Tool                                | Category | Arguments                                                                            | Description                                                                |
| ----------------------------------- | -------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `list_workorders`                   | read     | `site_id?`, `status?`, `asset_num?`, `priority?`, `date_from?`, `date_to?`, `page_size?`, `page_num?` | List work orders with optional filters; `page_size=0` returns all matches  |
| `get_workorder`                     | read     | `wonum`, `site_id`                                                                   | Get a single work order by number and site                                 |
| `get_workorder_tasks`               | read     | `wonum`, `site_id`                                                                   | List the child tasks of a parent work order                                |
| `get_workorder_costs`               | read     | `wonum`, `site_id`                                                                   | Actual labor/material/service/tool cost breakdown for a work order         |
| `get_workorder_actuals_vs_planned`  | read     | `wonum`, `site_id`                                                                   | Estimated vs actual hours and cost variance for a work order               |
| `get_workorder_kpis`                | read     | `site_id`, `period_months?`                                                          | Site KPIs: totals, backlog, overdue, avg completion, priority/asset splits |
| `get_schedule_calendar`             | read     | `site_id`, `date_from?`, `date_to?`, `group_by?`                                     | Scheduled (non-terminal) work orders in a date window, bucketed by day     |
| `get_my_assigned_workorders`        | read     | `labor_code`, `site_id?`, `open_only?`                                               | Work orders assigned to a given technician (labor code)                    |

### Write tools

| Tool                  | Category | Arguments                                                                                                   | Description                                                       |
| --------------------- | -------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `generate_work_order` | write    | `description`, `asset_num`, `site_id`, `priority?`, `work_type?`, `reported_by?`, `location?`, `notes?`, `wonum?`, `aob_source?` | Create a work order (status WAPPR); attach `aob_source` provenance |
| `update_workorder`    | write    | `wonum`, `site_id`, `description?`, `priority?`, `location?`, `asset_num?`, `notes?`                         | Update mutable fields on a work order                             |
| `approve_workorder`   | write    | `wonum`, `site_id`                                                                                          | Approve a work order (-> APPR)                                    |
| `assign_technician`   | write    | `wonum`, `site_id`, `labor_code`, `craft?`, `start_date?`, `hours_planned?`                                 | Assign a technician (adds a wplabor line)                         |
| `close_workorder`     | write    | `wonum`, `site_id`, `actual_hours?`, `failure_code?`, `resolution_notes?`                                   | Close a work order (-> COMP) with actuals and resolution          |
| `cancel_workorder`    | write    | `wonum`, `site_id`, `reason?`                                                                               | Cancel a work order (-> CAN)                                      |

### LLM-use tools

_None — the WO server makes no LLM calls; all tools are direct CouchDB operations._

### CPU-centric tools

_None — all tools are lightweight CouchDB queries/mutations (Mango `_find` / `GET` / `PUT`), with no heavy computation._

## tsfm — Time Series Foundation Model

**Path:** `src/servers/tsfm/main.py`
**Requires:** `tsfm_public` (IBM Granite TSFM), `transformers`, `torch` for ML tools — imported lazily; static tools work without them.
**Model checkpoints:** resolved relative to `PATH_TO_MODELS_DIR` (default: `src/servers/tsfm/artifacts/output/tuned_models`)

| Tool                   | Category                 | Arguments                                                                                                                   | Description                                                                                      |
| ---------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `get_ai_tasks`         | read                     | —                                                                                                                           | List supported AI task types for time-series analysis                                            |
| `get_tsfm_models`      | read                     | —                                                                                                                           | List available pre-trained TinyTimeMixer (TTM) model checkpoints                                 |
| `run_tsfm_forecasting` | read, write, cpu-centric | `dataset_path`, `timestamp_column`, `target_columns`, `model_checkpoint?`, `forecast_horizon?`, `frequency_sampling?`, ...  | Zero-shot TTM inference; returns path to a JSON predictions file                                 |
| `run_tsfm_finetuning`  | read, write, cpu-centric | `dataset_path`, `timestamp_column`, `target_columns`, `model_checkpoint?`, `save_model_dir?`, `n_finetune?`, `n_test?`, ... | Few-shot fine-tune a TTM model; returns saved checkpoint path and metrics file                   |
| `run_tsad`             | read, write, cpu-centric | `dataset_path`, `tsfm_output_json`, `timestamp_column`, `target_columns`, `task?`, `false_alarm?`, `ad_model_type?`, ...    | Conformal anomaly detection on top of a forecasting output JSON; returns CSV with anomaly labels |
| `run_integrated_tsad`  | read, write, cpu-centric | `dataset_path`, `timestamp_column`, `target_columns`, `model_checkpoint?`, `false_alarm?`, `n_calibration?`, ...            | End-to-end forecasting + anomaly detection in one call; returns combined CSV                     |

## vibration — Vibration Diagnostics

**Path:** `src/servers/vibration/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `VIBRATION_DBNAME` (default `vibration`), `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`); `numpy`, `scipy`
**DSP core:** `src/servers/vibration/dsp/` — adapted from [vibration-analysis-mcp](https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp) (Apache-2.0)

| Tool | Category | Arguments | Description |
|---|---|---|---|
| `get_vibration_data` | read | `site_name`, `asset_id`, `sensor_name`, `start`, `final?` | Fetch vibration time-series from CouchDB and load into the analysis store. Returns a `data_id`. |
| `list_vibration_sensors` | read | `site_name`, `asset_id` | List available sensor fields for an asset. |
| `compute_fft_spectrum` | read, cpu-centric | `data_id`, `window?`, `top_n?` | Compute FFT amplitude spectrum (top-N peaks + statistics). |
| `compute_envelope_spectrum` | read, cpu-centric | `data_id`, `band_low_hz?`, `band_high_hz?`, `top_n?` | Compute envelope spectrum for bearing fault detection (Hilbert transform). |
| `assess_vibration_severity` | read, cpu-centric | `rms_velocity_mm_s`, `machine_group?` | Classify vibration severity per ISO 10816 (Zones A–D). |
| `calculate_bearing_frequencies` | cpu-centric | `rpm`, `n_balls`, `ball_diameter_mm`, `pitch_diameter_mm`, `contact_angle_deg?`, `bearing_name?` | Compute bearing characteristic frequencies (BPFO, BPFI, BSF, FTF). |
| `list_known_bearings` | read | — | List all bearings in the built-in database. |
| `diagnose_vibration` | read, cpu-centric | `data_id`, `rpm?`, `bearing_designation?`, `bearing_*?`, `bpfo_hz?`, `bpfi_hz?`, `bsf_hz?`, `ftf_hz?`, `machine_group?`, `machine_description?` | Full automated diagnosis: FFT + shaft features + bearing envelope + ISO 10816 + fault classification + markdown report. |

## Running a server manually

Servers are normally spawned on-demand by an agent client. To launch one directly for testing:

```bash
uv run iot-mcp-server
uv run utilities-mcp-server
uv run fmsr-mcp-server
uv run wo-mcp-server
uv run tsfm-mcp-server
uv run vibration-mcp-server
```

They speak MCP over stdio, so they're idle until a client connects on stdin.
