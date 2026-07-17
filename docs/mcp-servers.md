# MCP Servers

Six FastMCP servers expose the AssetOpsBench domain logic. Each is a standalone stdio process spawned on-demand by clients (`plan-execute`, `claude-agent`, `openai-agent`, `deep-agent`, Claude Desktop). Backing services and credentials are listed per-server below.

## Contents

- [iot — IoT Asset Registry](#iot--iot-asset-registry)
- [utilities — Utilities](#utilities--utilities)
- [fmsr — Failure Mode and Sensor Relations](#fmsr--failure-mode-and-sensor-relations)
- [wo — Work Order](#wo--work-order)
- [tsfm — Time Series Feature Catalog](#tsfm--time-series-feature-catalog)
- [vibration — Vibration Diagnostics](#vibration--vibration-diagnostics)

## iot — IoT Asset Registry and Telemetry Records

Read-only tools for browsing the asset registry and querying IoT telemetry.

**Path:** `src/servers/iot/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `ASSET_DBNAME`, `IOT_DBNAME`)
**Sample assets:** `Chiller 6`, `mp_1`, and `hyd_1` from `asset_profile_sample.json`

### Registry and discovery tools

| Tool | Arguments | Description |
| ---- | --------- | ----------- |
| `sites` | - | List sorted site identifiers, with `MAIN` as the fallback |
| `asset_ids` | `site_name` | List asset identifiers registered at a site |
| `assets` | `site_name`, `assettype?` | List assets with compact metadata and optional exact type filtering |
| `asset_detail` | `site_name`, `asset_id` | Return registry details and installed-sensor count for one asset |
| `installed_sensors` | `site_name`, `asset_id` | List sensor names assigned in the registry |
| `measured_sensors` | `site_name`, `asset_id` | List measurement fields observed across the telemetry stream |
| `find_assets_by_sensors` | `site_name`, `sensors`, `match?`, `substring?`, `source?` | Find site assets by installed or measured sensor names |

### Telemetry tools

| Tool | Arguments | Description |
| ---- | --------- | ----------- |
| `stream_extent` | `site_name`, `asset_id`, `sensor?`, `start?`, `end?` | Count matching records and return their earliest and latest timestamps |
| `latest_reading` | `site_name`, `asset_id`, `sensor?` | Return the newest record, or the newest non-null value for one sensor |
| `history` | `site_name`, `asset_id`, `start?`, `end?`, `sensors?`, `limit?`, `cursor?` | Return chronological, projected observations with cursor paging |
| `sensor_coverage` | `site_name`, `asset_id` | Scan the full stream for per-sensor non-null counts and time coverage |
| `sensor_stats` | `site_name`, `asset_id`, `sensor?`, `start?`, `end?` | Compute per-sensor numeric counts, range, mean, and population standard deviation |

Telemetry windows are half-open ISO 8601 ranges. `history` supports cursor-based paging with up to
1000 observations per page.

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
**Requires:** LLM credentials for `generate_failure_modes` and `generate_failure_mode_sensor_mapping`; `get_failure_modes` reads the database.
**Failure-mode data:** `src/couchdb/scenarios_data/shared/fmea/failure_modes_sample.json` loaded into the `failure_mode` database collection.

| Tool                              | Category      | Arguments                                | Description                                                                                                                                             |
| --------------------------------- | ------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_failure_modes`               | read          | `asset_class`                            | Return known failure modes for an asset class from the database. Returns `asset_class`, `failure_modes`, `exhaustive`, and `source`.                    |
| `generate_failure_modes`          | read, LLM-use | `asset_class`, `max_modes?`              | Generate or extend a failure-mode list without writing the database. |
| `add_failure_modes`               | write         | `asset_class`, `failure_modes`, `exhaustive?`, `source?` | Persist failure modes for an asset class. |
| `generate_failure_mode_sensor_mapping` | read, LLM-use | `asset_class`, `failure_modes`, `sensors` | Score failure-mode/sensor relevancy via LLM and return bidirectional mappings. |

## wo — Work Order

**Path:** `src/servers/wo/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `WO_DBNAME`, `FAILURE_CODE_DBNAME`)
**Data init:** Handled automatically by `docker compose -f src/couchdb/docker-compose.yaml up` (runs `src/couchdb/init_wo.py` inside the CouchDB container on every start — database is dropped and reloaded each time)

Tools fall into several categories: **read**, **write**, **LLM-use**, and **CPU-centric**. Tools are registered centrally in `main.py`; set `AOB_READONLY=1` to expose only the read tools (9). The default exposes all 15 (9 read + 6 write).

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
| `get_failure_codes`                 | read     | `code?`                                                                               | List FCC failure-code references or fetch one exact code from CouchDB      |

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

## tsfm — Time Series Model & Feature Catalogs

**Path:** `src/servers/tsfm/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`); `numpy`, `pandas`. Set `TSFM_STORE=memory` for the in-memory backend the test suite uses.
**Catalog data:** `src/couchdb/scenarios_data/shared/tsfm/{model,feature}_catalog.json`, loaded by `src/couchdb/init_data.py` into the `model_catalog` and `feature_catalog` collections like every other AssetOpsBench collection. `FEATURE_CATALOG_DBNAME` overrides the feature database name.

Models and features are catalog **data, not tools**. A model card is a *pointer*: it records how to
construct or load a model — `sktime_class` + `params`, and/or `hf_repo` / `artifact_path` /
`remote_endpoint` / `model_checkpoint` — never the weights themselves. Feature transform cards store
executable EFE-style `fit` / `transform` programs; extractor cards store searchable metadata for
scalar feature extractors. The server reads both catalogs from CouchDB; it does not seed them.

### Tasks and evidence tools

Evidence tools take **file pointers** (`dataset_path`) and return typed results plus a pointer to
the full output. The server supplies evidence; the agent makes the decisions.

| Tool | Category | Arguments | Description |
| ---- | -------- | --------- | ----------- |
| `list_tasks` | read | — | List the standardized TSFM tasks and their contracts. |
| `profile_series` | read, cpu-centric | `dataset_path`, `timestamp_column?`, `channels?` | Structured facts about a series: dominant period, trend, gaps, channel count. |
| `characterize_series` | read, cpu-centric | `dataset_path`, `timestamp_column?`, `channels?`, `groups?`, `group_rules?` | Pattern evidence for a dataset; returns an evidence file pointer. |
| `data_quality` | read, write, cpu-centric | `dataset_path`, `timestamp_column?` | NaN stats + removal; emits a cleaned file pointer for downstream tools. |

### Model catalog — discovery

| Tool | Category | Arguments | Description |
| ---- | -------- | --------- | ----------- |
| `list_models` | read | `task_id?`, `domain?`, `status?` | List model cards, optionally filtered by task or domain. |
| `search_models` | read | `text`, `tags?`, `status?` | Case-insensitive substring search over id, description, family, and tags. |
| `find_models` | read | `task_id`, `min_context_length?`, `prediction_length?`, `domain?`, `top_k?` | Filter by task plus structured constraints and return a shortlist. |
| `describe_candidates` | read | `task_id`, `top_k?`, `domain?` | Compact candidate shortlist for a task, for an agent to reason over. |
| `describe_models` | read | `model_ids` | Compact record per id: description, family, `sktime_class`, context length, domain, tags. |
| `count_models` | read | — | Total active models plus a per-task breakdown. |
| `list_domains` | read | `task_id?` | The distinct domains present, with counts — the valid values for the `domain` filter. |
| `get_model_lineage` | read | `model_id` | Fine-tune ancestors and descendants, plus `supersedes` / `superseded_by` links. |
| `resolve_model` | read | `model_id` | Preflight: confirm a card can be loaded and report where its weights come from. Does not download or fit. |
| `hf_stats` | read | `model_id?`, `hf_repo?` | HuggingFace downloads and likes for a card's repo. Read-only; needs network to huggingface.co. |

### Model catalog — authoring and lifecycle

| Tool | Category | Arguments | Description |
| ---- | -------- | --------- | ----------- |
| `model_template` | read | — | The card contract: required and optional fields, the pointer choices, and a worked example. Static. |
| `register_model` | write | `model` | Register a schema-validated model card. A duplicate `model_id` is rejected, not overwritten. |
| `register_finetuned` | write | `model_id`, `checkpoint_path`, `base_model_id`, `context_length`, `prediction_length`, `description`, `domain?` | Point a card at a fine-tune checkpoint; inherits the base's `sktime_class` and records lineage. |
| `update_model` | write | `model_id`, `fields` | Patch fields, stamp `updated_at`, and re-validate against the schema. |
| `deprecate_model` | write | `model_id`, `reason?` | Soft delete: `status=deprecated`, dropping the card from active listings. |
| `new_model_version` | write | `model_id`, `fields`, `new_model_id?` | Register a successor and mark the predecessor superseded, cross-linked. |

### Feature catalog

| Tool | Category | Arguments | Description |
| ---- | -------- | --------- | ----------- |
| `list_features` | read | `kind?`, `status?` | List transform and/or extractor cards. `kind` may be `transform`, `extractor`, or omitted. |
| `search_features` | read | `text?`, `tags?`, `status?` | Search cards by feature id, name, description, or tags. |
| `get_feature` | read | `feature_id` | Return one stored feature card by id. |
| `get_feature_lineage` | read | `feature_id` | Parent and direct-descendant ids for a feature card. |
| `register_feature` | write | `feature`, `overwrite?` | Register a transform card after schema and executable-code validation. |
| `update_feature` | write | `feature_id`, `fields` | Patch metadata fields without rerunning executable validation. |
| `deprecate_feature` | write | `feature_id`, `reason?` | Mark a card deprecated while keeping it for audit and lineage. |
| `new_feature_version` | write | `feature_id`, `fields?`, `new_feature_id?` | Create a validated successor transform card and supersede the predecessor. |

Successful TSFM tool responses include a top-level `message` string. List and search responses also
include `features` or `models`; registration returns `status`, `id`, and `card`; card operations
return the card fields plus `message`; lineage returns `feature_id` / `model_id`, `ancestors`,
`root`, `descendants`, and `message`. Errors return `ErrorResult` with an `error` field.

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
