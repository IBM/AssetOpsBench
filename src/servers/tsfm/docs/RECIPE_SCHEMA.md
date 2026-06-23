# Recipe schema — where every parameter goes

A legacy TSFM call mixes ~30 parameters on one function. In the redesign they separate into
**three buckets by what they describe**, so a recipe stays reusable and every value is reasoned.

## 1. Data / task contract → MCP tool arguments (not the recipe)
Describe the *data instance*; identical regardless of model. Stay as tool args (as in legacy).

```
dataset_path · timestamp_column · target_columns · conditional_columns · id_columns
frequency_sampling · autoregressive_modeling
```

## 2. Model architecture → the catalog CARD (`param_schema`)
Properties of the checkpoint; the agent reads them from the card (legacy reads `model_config`
from `config.json`) and overrides only within `param_space` limits.

```
context_length · prediction_length · patch_length · decoder_mode
```

## 3. Run-time algorithm choices → the RECIPE
The knobs the agent actually reasons per run. Carried as recipe fields; **defaults fill the
rest** (exactly like legacy merges `training_config_dic` over `_ttm_main_config()`), so recipes
stay terse. Each is governed by `param_space` hints + `validate_block`.

### `recipe.estimator` / `recipe.transforms` — per-component `params`
Constructor params of the chosen sktime estimator/transform (`sp`, `strategy`, `n_neighbors`,
`window_size`, `n_clusters`, …). Schema + hints from `param_space.param_schema(card)`;
validated by `param_space.validate_params(card, params)`.

### `recipe.finetune` — the `_ttm_main_config` knobs (training_regime → fine_tune)
```jsonc
{ "estimator": {"model_id": "ttm_512_96"},
  "finetune": { "n_finetune": 0.05, "n_test": 0.05, "n_calibration": 0.0,
                "lr": 0.001, "epochs": 4, "batch_size": 32, "head_dropout": 0.7,
                "backbone_frozen": false, "decoder_mode": "mix_channel",
                "scaling": "standard", "scheduler": "OneCycleLR", "es_patience": 15,
                "p_validation": 0.1, "seed": 42 },
  "fh": [1,2,3] }
```
Hints: `param_space.FINETUNE_HINTS`. Presence flips `training_regime` to `fine_tune`.

### `recipe.anomaly` — the conformal-AD knobs
```jsonc
{ "estimator": {"model_id": "ttm_96_28"},
  "anomaly": { "ad_model_type": "timeseries_conformal_adaptive", "false_alarm": 0.05,
               "n_calibration": 0.2, "threshold_function": "weighting",
               "window_size": null, "nonconformity_score": "absolute_error", "task": "fit" } }
```
Hints: `param_space.ANOMALY_HINTS` (`false_alarm` = 1 − coverage; `window_size` ≈ dominant_period).

### `recipe.conformal` — calibrated prediction intervals on any forecaster
```jsonc
{ "conformal": {"coverage": 0.9} }     // → predict_interval; CRPS in evaluate
```

## The reasoning loop (same as per-model params)
```
discover_components → recipe_blocks {finetune, anomaly} hints   ← the agent reads the menu
profile_series      → evidence (dominant_period, length, …)
        ↓  agent fills recipe.finetune / recipe.anomaly from hints + evidence
run_recipe          → validate_block(...) → block_audit recorded in the run
        ↓
scoring grades the choices (lr/epochs sane, false_alarm = 1-coverage, window ≈ period)
```

## Status
Schema + hints + `validate_block` + `block_audit` in the run record are **built and tested**.
The forecasting/conformal path consumes `conformal` today; `finetune` flips the regime and is
audited. The compute that *consumes* `finetune`/`anomaly` end-to-end is the env-gated sktime
migration (Phase 2–4) — the blocks are where those params will land when wired.

Files: `reasoning/param_space.py` (FINETUNE_HINTS / ANOMALY_HINTS / block_schema / validate_block),
`engine/composition.py` (block_audit + discover_components.recipe_blocks).
