# Forecasting end-to-end (the run_tsfm_forecasting replacement)

The old `run_tsfm_forecasting` tool is gone (the whole `legacy/` server was removed). Forecasting
is a **`run_recipe` call with a forecaster card** — one expression of the general recipe engine,
composable (ensemble, conformal, swap models) and consistent with every other task.

## The end-to-end workflow
```
CouchDB (iot DB)
   │  IoT server: history(site, asset, start, final)        ← time-series DATA lives in CouchDB
   ▼
local storage  ──►  /shared/iot/chiller6.csv                ← IoT materialises it to a local file
   │  agent passes the path
   ▼
TSFM.run_recipe(dataset_path=…, timestamp_column=…, target_columns=[…],
                recipe={estimator:{model_id:"ttm_96_28"}, fh:[1..H]})
   │  resolve TTM card (sktime TinyTimeMixerForecaster) → zero-shot predict (no training)
   ▼
results_file (file pointer)  ──►  handoff downstream (FMSR → WO → Spot)
```

**Responsibility split**
- **time-series data**: IoT owns it — CouchDB → local file. TSFM never reads the data from CouchDB.
- **catalog (models/features)**: TSFM reads from CouchDB (`model_catalog`, `feature_catalog`).
- **forecast on a file pointer → results_file**: TSFM `run_recipe`.
- **chaining IoT→TSFM→downstream**: the agent (or `run_plan` within one server).

## The forecast call
```jsonc
run_recipe(
  dataset_path     = "/shared/iot/chiller6.csv",   // file pointer (IoT-materialised)
  timestamp_column = "timestamp",
  target_columns   = ["temp"],
  recipe           = { "estimator": {"model_id": "ttm_96_28"}, "fh": [1, …, 28] }
)
→ RecipeResult{ run_id, results_file, metric, backtest_score, training_regime:"zero_shot" }
```
`ttm_96_28` resolves to `sktime.forecasting.ttm.TinyTimeMixerForecaster` with
`model_path=ibm-granite/granite-timeseries-ttm-r2` — **sktime wraps the `tsfm_public` TTM**, run
through the uniform fit/predict interface and the file-pointer contract.

Swap the card to forecast with anything in the catalog (Chronos, MOIRAI, AutoARIMA, an ensemble):
```jsonc
"estimator": {"model_id": "amazon__chronos-t5-small"}
"ensemble":  {"members": [{"model_id":"ttm_96_28"}, {"model_id":"autoarima"}], "combine":"mean"}
```

## Prediction-based anomaly detection (conformal)
The same forecaster cards drive AD: forecast a recent window, wrap the forecaster in sktime
`ConformalIntervals`, and flag any actual that falls outside the calibrated band.
```jsonc
run_recipe(dataset_path, "timestamp", ["temp"], recipe={
  "task": "tsfm_anomaly_detection", "method": "conformal",
  "estimator": {"model_id": "ttm_96_28"},      // any forecaster card
  "conformal": {"coverage": 0.9},               // false-alarm budget = 1 − coverage
  "fh": [1, …, 24]                              // the window to screen
})
→ RecipeResult{ n_anomalies, anomaly_indices_head, labels, coverage, results_file }
```
(`method:"detector"` — the default — instead runs a detector card: TSPulse zero-shot, SubLOF, PyOD.)

## Status
- ✅ **Runnable model cards** — the TTM cards carry `sktime_class` + `model_path` (Granite/Chronos/
  MOIRAI/… migrated cards already did). 38 forecasting cards resolve. `run_recipe` takes the file
  pointer and returns a `results_file`. The zero-shot fast path skips retraining.
- ✅ **Workflows tested end-to-end** (`tests/test_workflows.py`, classical cards so no ML deps):
  (1) forecasting via `run_recipe`; (2) prediction-based AD with conformal flagging an injected
  spike; (3) data_quality → conformal AD chain.
- ⏳ **Env-gated (needs `tsfm_public` + `torch`)** — actual TTM/foundation inference. Smoke it with
  ```bash
  TSFM_STORE=memory python -m servers.tsfm.scripts.forecast_ttm                 # synthetic CSV
  TSFM_STORE=memory python -m servers.tsfm.scripts.forecast_ttm /shared/iot/chiller6.csv temp 28
  ```

## Follow-ups (clean, not yet wired)
- **Data-quality block** — `data_quality` is a standalone tool today; optionally inline it as a
  `recipe.data_quality` pre-step.
- **Multivariate / exogenous (conditional_columns)** targets in `run_recipe`'s forecasting path.
