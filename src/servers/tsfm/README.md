# TSFM MCP Server

A time-series-AI server for AssetOpsBench, built on **sktime** as the execution substrate. The
agent **discovers components, reads evidence, composes a recipe, and runs it** — forecasting,
anomaly detection, regression/classification/clustering, imputation, and evaluation — over the
standard AssetOpsBench MCP contract. Models and features are **catalog data (cards), not tools**
(the HuggingGPT principle); bulk data crosses the boundary as **file pointers**.

Substrate is sktime end-to-end (TTM/Chronos/MOIRAI/Moment/TimesFM via sktime's foundation
adapters; TSPulse / SubLOF / PyOD for anomaly; classical forecasters and tabular estimators).
There is **no legacy / `tsfm_public`-specific path** — foundation models resolve through sktime
and only need `tsfm_public`+`torch` at run time.

## Quick start

```bash
# tests (in-memory store, no CouchDB / torch needed)
TSFM_STORE=memory python -m pytest src/servers/tsfm --import-mode=importlib -q

# production: load the catalogs into CouchDB (standard AssetOpsBench loader), then serve
tsfm-mcp-server          # entry point → tsfm.main:main, FastMCP over stdio, reads CouchDB
```

## House contract

Entry point `tsfm-mcp-server` → `tsfm.main:main`, a module-level `mcp = FastMCP("tsfm")` served
over stdio, matching the sibling AssetOpsBench servers. Every tool:

- is decorated `@mcp.tool(...)` and returns a **typed Pydantic result** (`Union[XResult, ErrorResult]`);
- **validates inputs** → returns `ErrorResult(error=…)` rather than throwing across the wire;
- takes bulk data as a **file pointer** (`dataset_path` in, `results_file`/`evidence_file` out).

## Layered package

```
core/        store (MemoryStore|CouchStore), schemas (ModelCard/FeatureCard), tasks, glossary, results_models
substrate/   resolver — sktime as the execution substrate (training_regime, is_foundation, resolve)
stores/      model_store, feature_store (lifecycle + lineage + search), results
reasoning/   profile (evidence), param_space, feature_selection (FLOps, 109 extractors), patterns (pattern evidence), dataquality
engine/      composition (run_recipe forecast+anomaly, run_tabular_recipe, discover_components), plan (recipe DAG), feature_runner (EFE), evolve (AlphaEvolve)
eval/        gifteval — GIFT-Eval scoring + leaderboard
io/          refs (file pointers: load/write series, materialize IoT), window
main.py      the MCP tool surface     ·  config.py  env knobs  ·  bootstrap.py  seed loader
```

## The surface (36 tools)

- **Discover / evidence:** `list_tasks`, `discover_components`, `describe_candidates`,
  `find_models`, `find_features`, `get_component`, `profile_series`, `select_features` (FLOps),
  `characterize_series` (pattern evidence), `data_quality`.
- **Compose & run:** `run_recipe` (forecasting **and** anomaly — anomaly via
  `recipe.task=tsfm_anomaly_detection`, `method=detector|conformal`), `run_tabular_recipe`,
  `run_plan` (recipe DAG), `evaluate` (GIFT-Eval).
- **Catalog write-back / lifecycle:** `register_model`/`register_feature`, `register_finetuned`,
  `list_models`/`search_models`, `update_/deprecate_/new_*_version`, lineage, `list_extractors`.
- **Results:** `get_result`, `list_results`, `get_run`, `list_runs`.
- **Evolve (AlphaEvolve):** `evolve_ask`, `evolve_tell`, `evolve_best`.

See `docs/TOOLS.md` for the full as-built table.

## Data & catalog model

- **Bulk data is a file pointer.** IoT/vibration data lives in CouchDB; the IoT server
  materialises it to a local CSV; TSFM takes `dataset_path` and returns a `results_file`. No bulk
  arrays cross the MCP boundary.
- **Catalog = two CouchDB collections** — `model_catalog` (48 cards) and `feature_catalog`
  (6 transforms + 109 FLOps extractors). They load like every other AssetOpsBench collection from
  `src/couchdb/scenarios_data/shared/tsfm/{model,feature}_catalog.json`; `bootstrap.load_seeds`
  reads from there (override with `$TSFM_SEEDS_DIR`). See `docs/COUCHDB_LOADING.md`.
- **Stores:** `CouchStore` is the default/production backend; `MemoryStore` is a **test double**
  only (`TSFM_STORE=memory`). `make_store()` picks by env.

## Workflows

- **Forecasting** — `run_recipe(dataset_path, …, recipe={estimator:{model_id:"ttm_96_28"}, fh:[…]})`
  → transforms + single/ensemble + optional conformal → `results_file`. Zero-shot is the default;
  fine-tune is opt-in. See `docs/FORECASTING_WORKFLOW.md`.
- **Anomaly detection** — same tool, `recipe.task=tsfm_anomaly_detection`: `method=detector`
  (TSPulse zero-shot, SubLOF, PyOD) or `method=conformal` (forecast + sktime `ConformalIntervals`
  → flag out-of-band points).
- **Pattern evidence** — `characterize_series` describes a signal's *shape* (per channel-group
  state + rate over changepoint phases, plus cross-channel relation) as structured, **fault-free**
  evidence for an LLM to reason over. Generic (any signals; grouping is opt-in). See
  `docs/PATTERN_EVIDENCE.md`.

The principle throughout: the **server supplies evidence; the agent decides** the recipe, the
parameters, and the diagnosis. Results hand off downstream (FMSR/WO/Spot) — no alerts here.

## Config

`TSFM_STORE=couch|memory` (default `couch`) · `TSFM_SEEDS_DIR` (override catalog seed location) ·
`COUCHDB_URL` / credentials (couch backend) · `LOG_LEVEL`.

Apple-Silicon note: TSPulse/torch may hit a Metal (MPS) `gatherND` assertion at predict time; run
foundation models on CPU (e.g. disable MPS) until the upstream PyTorch fix lands.

## Status

sktime-native surface (36 tools), conformal AD, FLOps/EFE feature layers, AlphaEvolve loop, and
the pattern-evidence engine are **done and tested** — `121 passed / 5 skipped` (TSPulse skips
cleanly without Python ≥3.11 + `granite-tsfm`). Foundation-model inference (TTM/TSPulse/…) is
env-gated on `tsfm_public`+`torch`. Docs live in `docs/` (STRUCTURE, TOOLS, FORECASTING_WORKFLOW,
PATTERN_EVIDENCE, COUCHDB_LOADING, GLOSSARY, …).
