# TSFM MCP Server

A strong, runnable time-series AI server for AssetOpsBench: **model store + feature store +
per-task result tables**, integrating three IBM papers — **AnomalyKiTS** (AD operators &
pipelines), **EFE** (evolvable feature transforms), **FLOps** (dynamic feature selection).

It runs **with no torch / CouchDB / mcp** (MemoryStore + StubCompute) for deterministic tests
and the benchmark, and swaps to CouchDB + real model backends for production via env flags.

## Quick start

```bash
pip install numpy pytest                  # core deps
python -m pytest tsfm_server/tests/ -q    # 8 end-to-end tests, all green
python -m tsfm.run_demo            # forecast · anomaly · finetune+register · FLOps · export
# serve over MCP (needs `pip install mcp`):
python -m tsfm.server
```

## Architecture (modules)

```
store.py            CORE: Store (MemoryStore | CouchStore), get/put/find/delete/export_state
model_store.py      model catalog: find_models (task+context+domain rank), register_finetuned, resolve_checkpoint
feature_store.py    feature catalog: find_features (by scenario category), register_feature (validity-gated), select_features
feature_selection.py FLOps: select_features + discover_lookback (130+ extractor library, reference + CD filter)
feature_runner.py   EFE: load + validate + fit/transform/inverse a stored program (sandbox)
runner.py           model resolution: serves_task + availability (local | hf_hub | api-async | toolkit)
compute.py          ComputeBackend: StubCompute (deterministic) | RealCompute (TTM/TSPulse/AnomalyKiTS — TODO)
io.py               data access: read sensor window (iot/vibration; synthesizes for the demo)
results.py          result tables: one per task type, write/get/list
pipeline.py         ORCHESTRATION: read → lookback → select/transform → model → compute → write
server.py           the 15 MCP tools (FastMCP)
bootstrap.py        load seeds into a store
seeds/              model_catalog.json (TTMs + fine-tune), anomalykits_models.json (4 AD pipelines), feature_catalog.json
```

## The pipeline (all three papers in one flow)

```
io.read_window(asset, channels)
  → FLOps: discover_lookback + select_features        (feature_selection.py)
  → apply transforms (EFE/AnomalyKiTS operators)      (feature_runner.py)
  → find_models(task, min_context_length=lw, domain)  (model_store.py)
  → resolve availability (local/hf/api-async/toolkit) (runner.py)
  → compute: forecast | anomaly(score+label+contribution) | classify | finetune  (compute.py)
  → write_result(task_type, summary, provenance)      (results.py)
```

## Tasks & result tables

| task_id | result table | model families |
|---|---|---|
| tsfm_forecasting | forecast_result | TTM, Chronos, TimesFM, MOMENT, fine-tunes |
| tsfm_anomaly_detection | anomaly_result | **AnomalyKiTS** DeepAD/RelationshipAD/ReconstructAD/WindowAD, TSPulse |
| tsfm_classification | classification_result | (PHM fault) |
| tsfm_imputation | imputation_result | TSPulse/MOMENT |
| tsfm_forecasting_evaluation | evaluation_result | — |

Anomaly output is AnomalyKiTS-style: per-ts `anomaly_score` + `anomaly_label(+1/-1)` +
per-variable `contribution` → the `top_contributors` (kpi_name + High/Low) are the FMSR handoff.

## Key design choices

- **Fine-tune is decoupled**: `run_tsfm_finetuning` returns a checkpoint path + metrics and
  registers nothing; `register_model` (agent-decided) points the catalog at that location.
- **Catalog = pointer index**; weights live at artifact_path / hf_repo / remote_endpoint /
  toolkit. New model on a known framework = a row; new framework = one adapter.
- **FLOps look-back → context_length**: the discovered window drives model selection.
- **Everything is a Store collection** → captured by `export_state()` (issue #394 / PR #400)
  for deterministic scenario scoring. Out of scope (other servers): alerts / SR / WO /
  verification, which consume `anomaly_result`.

## Config

`TSFM_STORE=memory|couch` · `TSFM_COMPUTE=stub|real` · `TSFM_RESULTS_DIR` ·
`COUCHDB_URL/USERNAME/PASSWORD` (couch).

## Status

Core + stores + pipeline + 15 tools: **done, tested**. To productionize: implement
`RealCompute` (TTM/TSPulse via `tsfm_public`, AnomalyKiTS pipelines), point `seeds/` at real
checkpoints, run with `TSFM_STORE=couch`.
