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

**Path:** `src/servers/iot/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `IOT_DBNAME`)

| Tool      | Arguments                                  | Description                                                             |
| --------- | ------------------------------------------ | ----------------------------------------------------------------------- |
| `sites`   | —                                          | List all available sites                                                |
| `assets`  | `site_name`                                | List all asset IDs for a site                                           |
| `sensors` | `site_name`, `asset_id`                    | List sensor names for an asset                                          |
| `history` | `site_name`, `asset_id`, `start`, `final?` | Fetch historical sensor readings for a time range (ISO 8601 timestamps) |

## utilities — Utilities

**Path:** `src/servers/utilities/main.py`
**Requires:** nothing (no external services)

| Tool                   | Arguments   | Description                                            |
| ---------------------- | ----------- | ------------------------------------------------------ |
| `json_reader`          | `file_name` | Read and parse a JSON file from disk                   |
| `current_date_time`    | —           | Return the current UTC date and time as JSON           |
| `current_time_english` | —           | Return the current UTC time as a human-readable string |

## fmsr — Failure Mode and Sensor Relations

**Path:** `src/servers/fmsr/main.py`
**Requires:** `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` for unknown assets; curated lists for `chiller` and `ahu` work without credentials.
**Failure-mode data:** `src/servers/fmsr/failure_modes.yaml` (edit to add/change asset entries)

| Tool                              | Arguments                                | Description                                                                                                                                             |
| --------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_failure_modes`               | `asset_name`                             | Return known failure modes for an asset. Uses a curated YAML list for chillers and AHUs; falls back to the LLM for other types.                         |
| `get_failure_mode_sensor_mapping` | `asset_name`, `failure_modes`, `sensors` | For each (failure mode, sensor) pair, determine relevancy via LLM. Returns bidirectional `fm→sensors` and `sensor→fms` maps plus full per-pair details. |

## wo — Work Order

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

## tsfm — Time Series Foundation Model

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

## vibration — Vibration Diagnostics

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
