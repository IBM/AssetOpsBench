# MCP Servers

Six FastMCP servers expose the AssetOpsBench domain logic. Each is a standalone stdio process spawned on-demand by clients (`plan-execute`, `claude-agent`, `openai-agent`, `deep-agent`, Claude Desktop). Backing services and credentials are listed per-server below.

## Contents

- [iot — IoT Asset Registry](#iot--iot-asset-registry)
- [utilities — Utilities](#utilities--utilities)
- [fmsr — Failure Mode and Sensor Relations](#fmsr--failure-mode-and-sensor-relations)
- [wo — Work Order](#wo--work-order)
- [tsfm — Time Series Model and Feature Catalogs](#tsfm--time-series-model-and-feature-catalogs)
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
**Requires:** CouchDB for the catalog lookup tools (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `CATALOG_DBNAME`); `json_reader` and the time tools do not need external services.

| Tool                   | Category | Arguments   | Description                                            |
| ---------------------- | -------- | ----------- | ------------------------------------------------------ |
| `json_reader`          | read     | `file_name` | Read and parse a JSON file from disk                   |
| `get_sensor_catalog`   | read     | `sensor?`   | List sensor catalog entries, or fetch an exact sensor  |
| `get_asset_catalog`    | read     | `asset?`, `category?` | List asset catalog entries, optionally filtered by asset or category |
| `get_failure_mode_catalog` | read | `failure_mode?`, `category?` | List failure-mode catalog entries, optionally filtered by failure mode or category |
| `current_date_time`    | read     | —           | Return the current UTC date and time as JSON           |
| `current_time_english` | read     | —           | Return the current UTC time as a human-readable string |

## fmsr — Failure Mode and Sensor Relations

**Path:** `src/servers/fmsr/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`, `FAILURE_MODE_DBNAME`) for stored modes; LLM credentials for `generate_failure_modes` and `generate_failure_mode_sensor_mapping`.
**Failure-mode data:** `src/couchdb/scenarios_data/shared/fmea/failure_modes_sample.json` loaded into the `failure_mode` database collection by default.

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

## tsfm — Time Series Model and Feature Catalogs

**Path:** `src/servers/tsfm/main.py`
**Requires:** CouchDB (`COUCHDB_URL`, `COUCHDB_USERNAME`, `COUCHDB_PASSWORD`); `numpy`, `pandas`, `sktime`, `scipy`, and the model-specific soft dependencies required by the cards you run. Set `TSFM_STORE=memory` for the in-memory backend the test suite uses.
**Catalog data:** `src/couchdb/scenarios_data/shared/tsfm/{model,feature}_catalog.json`, loaded by `src/couchdb/init_data.py` into the `model_catalog` and `feature_catalog` collections like every other AssetOpsBench collection. `MODEL_CATALOG_DBNAME` and `FEATURE_CATALOG_DBNAME` override the model and feature database names. Runs are stored in `tsfm_runs`, plans in `tsfm_plans`, and typed outputs in each task's result collection.

Models and features are catalog **data, not tools**. A model card is a *pointer*: it records how to
construct or load a model — `sktime_class` + `params`, and/or `hf_repo` / `artifact_path` /
`remote_endpoint` / `model_checkpoint` — never the weights themselves. Feature transform cards store
executable EFE-style `fit` / `transform` programs; extractor cards store searchable metadata for
scalar feature extractors. The server reads both catalogs from CouchDB; it does not seed them.

The TSFM server currently exposes 41 MCP tools across task/evidence discovery, model catalog
management, feature catalog management, recipe execution, evaluation, and the run/result ledger.

### Tasks and evidence tools

Evidence tools take **file pointers** (`dataset_path`) and return typed results plus a pointer to
the full output. The server supplies evidence; the agent makes the decisions.

| Tool | Category | Arguments | Description |
| ---- | -------- | --------- | ----------- |
| `list_tasks` | read | — | List the standardized TSFM tasks and their contracts. Use this first when you need the canonical task definitions. |
| `profile_series` | read, cpu-centric | `dataset_path`, `timestamp_column?`, `channels?` | Summarize a file-pointer-backed series with factual evidence such as series length, channel count, dominant period, and other basic temporal characteristics. |
| `characterize_series` | read, cpu-centric | `dataset_path`, `timestamp_column?`, `channels?`, `groups?`, `group_rules?` | Produce pattern evidence for a dataset, including grouped states, changepoint phases, relations, and a file pointer to the full evidence payload. |
| `data_quality` | read, write, cpu-centric | `dataset_path`, `timestamp_column?` | Report NaN / cleaning statistics and emit a cleaned file pointer for downstream tools. |

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
| `hf_stats` | read | `model_id?`, `hf_repo?` | Read-only HuggingFace popularity lookup for a model card or repo. Returns downloads and likes; needs network to huggingface.co. |

For anomaly-detection tasks, choose the estimator from this catalog before executing a run:
`find_models(task_id="tsfm_anomaly_detection")` is the direct path, while `search_models` is useful
when the request names a detector family or tag. The selected card must include
`tsfm_anomaly_detection` in `task_ids`; pass its `model_id` to `run_recipe` in the recipe estimator.

### Model catalog — authoring and lifecycle

| Tool | Category | Arguments | Description |
| ---- | -------- | --------- | ----------- |
| `model_template` | read | — | The card contract: required and optional fields, the pointer choices, and a worked example. Static. |
| `register_model` | write | `model` | Register a schema-validated model card. A duplicate `model_id` is rejected, not overwritten. |
| `register_finetuned` | write | `model_id`, `checkpoint_path`, `base_model_id`, `context_length`, `prediction_length`, `description`, `domain?` | Card a fine-tune checkpoint: inherits the base's `sktime_class`, points `params.model_path` at the checkpoint, records lineage, and pins the **serving** regime to `zero_shot` (serving loads and predicts, it does not re-train). Rejects inspectable base wrappers that cannot accept `model_path`. |
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
| `count_features` | read | — | Extractor / transform / total counts in the catalog. |
| `describe_features` | read | `names` | kind + name + description for the named features (extractors or transforms). |
| `extract_features` | read, cpu-centric | `dataset_path`, `extractors`, `target_columns`, `timestamp_column?`, `window?` | Apply the named scalar extractors to a series and return the raw feature values - no model. `window=None` -> one vector for the whole series; `window=W` -> a (windows x features) matrix over non-overlapping W-length tiles. |
| `select_features` | read, cpu-centric | `dataset_path`, `channel`, `extractors`, `timestamp_column?`, `reference_feature?`, `cd_margin?` | Rank a candidate extractor set on one series by self-supervised one-step-ahead forecasting and return the shortlist that beats `reference_feature` by `cd_margin`. No labels needed. |

The extractor library backing `extract_features` / `select_features` is
`reasoning.feature_selection.EXTRACTORS` (228 scalar extractors); pick names from
`list_features(kind="extractor")`. These are the inspectable feature-engineering half of the
workflow: `select_features` narrows the library to what matters for a series, `extract_features`
computes the survivors, and the values feed `run_tabular_recipe`.

### Compose and run

The catalog describes models; these tools **run** them. A recipe names an estimator (a `model_id`
from the catalog, or an inline `sktime_class` + `params`) plus optional blocks. `run_recipe`
compiles it, backtests it, produces a forecast or anomaly output, and persists a run record.
`recipe_template` returns the contract so an agent does not have to guess the shape.

For time-series anomaly detection, use the model-catalog tools first, then execute:
`find_models(task_id="tsfm_anomaly_detection")` or `search_models` -> select a model card ->
`run_recipe` with `recipe={"task": "tsfm_anomaly_detection", "estimator": {"model_id":
"<selected_model_id>"}}`. Base the final anomaly segment/JSON answer on the returned anomaly count,
indices, labels, and `results_file` payload.

| Tool | Category | Arguments | Description |
| ---- | -------- | --------- | ----------- |
| `recipe_template` | read | — | The recipe contract: `estimator_spec`, `optional_blocks` (`fh`, `transforms`, `ensemble`, `conformal`, `finetune`, `save_to`, `anomaly`, `impute`, `eval`), `rules`, and executable `examples`. Static. |
| `run_recipe` | write, cpu-centric | `dataset_path`, `timestamp_column`, `target_columns`, `recipe`, `asset_id?`, `parent_run_id?` | Run a forecasting or anomaly recipe on a series file pointer. Chooses the backtest by regime (`zero_shot` = one holdout; `fit_on_series`/`fine_tune` = expanding folds); returns `backtest_score`, `folds`, a results-file pointer, and — when the recipe carries `save_to` — the `checkpoint_path` of the persisted weights. |
| `run_tabular_recipe` | write, cpu-centric | `dataset_path`, `recipe`, `label_column?`, `asset_id?` | Series→tabular run (regression / classification / clustering): each CSV row is an instance; FLOps features are extracted, then the estimator fits. Omit `label_column` for clustering. |
| `run_plan` | write, cpu-centric | `plan_spec`, `asset_id?`, `scenario_id?` | Execute a recipe DAG: file-pointer chaining across a HuggingGPT-style task list. |
| `evaluate` | write, cpu-centric | `recipe`, `configs` | GIFT-Eval-style scoring of a recipe across several dataset configs: seasonal-naive-normalized MASE + CRPS, geometric mean over configs. |
| `list_runs` | read | `asset_id?` | List run records and plans, optionally filtered by asset. |
| `get_run` | read | `run_id` | One run record: its recipe config and outcome. |
| `list_results` | read | `task_type`, `asset_id?`, `scenario_id?` | List persisted results for a task type. |
| `get_result` | read | `task_type`, `result_id` | One persisted result — the record a run produced and stored. |

**How a run's output reaches the agent.** Every execution returns a `results_file` - a `file://`
pointer to a JSON document in `TSFM_WORKDIR` (default `/tmp/tsfm_work`, override with the env var)
holding the full payload (forecast head, prediction intervals, or anomaly labels). The tool response
carries the pointer plus the headline numbers (`backtest_score`, `folds`, `n_anomalies`); the agent
opens the pointer only when it needs the detail. The same run is persisted three ways: the response
pointer, a **run record** in `tsfm_runs` (read by `list_runs` / `get_run`), and a **typed result**
in the task's result collection - `forecast_result`, `anomaly_result`, ... - read by `list_results`
/ `get_result`, so a later agent can find a run by `task_type` + `asset_id` without knowing its
`run_id`. The run record and the result index live in CouchDB (durable) under the default backend;
the payload file lives on the local filesystem of whichever host ran the recipe, so on ephemeral or
multi-host deployments treat `results_file` as host-local and rely on the CouchDB record for the
durable summary.

**The fine-tune lifecycle is entirely on this surface.** `run_recipe` with a `finetune` block trains
the estimator and, with `save_to`, writes an HF-format checkpoint and returns its path;
`register_finetuned` cards that checkpoint with lineage and a pinned serving regime; a further
`run_recipe` on the new card loads those weights and forecasts. No dropping to raw sktime.

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
