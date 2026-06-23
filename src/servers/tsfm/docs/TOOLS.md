# TSFM MCP tool surface (as-built)

**Entry:** `tsfm-mcp-server` → `tsfm.main:main` (module-level `mcp`, stdio), house contract
matching the sibling AssetOpsBench servers.

**Contract for every tool:**
- decorated `@mcp.tool(title=...)`, returns a **typed Pydantic result** as `Union[XResult, ErrorResult]`;
- **inputs validated** → `ErrorResult(error=...)` on bad/empty args (never an exception across the wire);
- **bulk data is a FILE POINTER** — `dataset_path` in, `results_file` (a `file://` pointer) out;
- models & features are **catalog data**, not tools (HuggingGPT principle).

## 36 tools, one sktime-native surface (no legacy)

Forecasting **and** anomaly both run through `run_recipe` (anomaly via `recipe.task` +
`recipe.method`); there is no separate anomaly tool and no `tsfm_public`-specific compat layer.

### Core surface (23) — discover / evidence / compose+run / clean / write-back / results / evolve

| # | Tool | Result | Notes |
|---|------|--------|-------|
| 1 | `list_tasks` | TasksResult | the 8 standardized TS-AI tasks + contracts |
| 2 | `discover_components(task)` | ComponentsResult | menu: models/foundation/transforms/combiners/regimes + `recipe_blocks` |
| 3 | `describe_candidates(task_id, top_k, domain?)` | CandidatesResult | HuggingGPT-ranked cards |
| 4 | `find_models(task_id, …)` | ModelsResult | structured filter → ranked shortlist |
| 5 | `find_features(category?, task?, model?)` | FeaturesResult | feature transforms |
| 6 | `get_component(component_id)` | ComponentResult | card (+ `param_schema` for models) |
| 7 | `profile_series(dataset_path, …)` | ProfileResult | evidence only (seasonality/stationarity/channels) |
| 8 | `select_features(dataset_path, …)` | FeatureSelectionResult | FLOps multi-config; `detail_file` pointer |
| 8b | `characterize_series(dataset_path, …, groups?, group_rules?)` | CharacterizeResult | **pattern EVIDENCE** (shape only, no fault): per-group state+rate over changepoint phases + bivariate relation (decoupled/co_move/lead_lag). Generic; grouping opt-in. |
| 9 | `run_recipe(dataset_path, timestamp_column, target_columns, recipe, …)` | RecipeResult | **forecasting** (default) OR **anomaly** (`recipe.task=tsfm_anomaly_detection`: `method=detector` for TSPulse/SubLOF, `method=conformal` for prediction-based AD) → `results_file` |
| 10 | `run_tabular_recipe(dataset_path, recipe, label_column?)` | TabularResult | regression/classification/clustering: FeatureUnion → estimator |
| 11 | `run_plan(plan_spec, …)` | PlanResult | recipe DAG, file-pointer chaining |
| 12 | `evaluate(recipe, configs)` | EvaluateResult | GIFT-Eval (MASE+CRPS, geo-mean) |
| 12b | `data_quality(dataset_path, timestamp_column?)` | DataQualityResult | NaN-clean + summary → cleaned file pointer (pre-step for forecast/AD) |
| 13 | `register_model(model)` | RegisterResult | validated against `ModelCard` |
| 14 | `register_feature(feature)` | RegisterResult | validated against `FeatureCard` |
| 15 | `get_result(task_type, result_id)` | ResultRecord | per-task result table |
| 16 | `list_results(task_type, …)` | ResultsListResult | |
| 17 | `get_run(run_id)` | RunRecord | run lineage (recipe/plan) |
| 18 | `list_runs(asset_id?)` | RunsResult | runs + plans |
| 19 | `evolve_ask(task, kind, dataset_path?, …)` | EvolveAskResult | AlphaEvolve: sample parents + inspirations + evidence to mutate |
| 20 | `evolve_tell(task, kind, program, …)` | EvolveTellResult | validate + evaluate→fitness + MAP-Elites archive + lineage |
| 21 | `evolve_best(task, kind?, top_k)` | EvolveBestResult | the evolved frontier (elites per behaviour cell) |

### Catalog lifecycle (12) — pull / update / version / add, per store

| Tool | Result | Notes |
|------|--------|-------|
| `list_models(task_id?, domain?, status?)` | ModelsResult | enumerate model cards (mirror of list_extractors; unranked) |
| `search_models(text, tags?, status?)` | ModelsResult | free-text/tag search over the model catalog |
| `get_model_lineage(model_id)` | LineageResult | version chain (supersedes / superseded_by) |
| `update_model(model_id, fields)` | CardResult | patch a model card (re-validated) |
| `deprecate_model(model_id, reason?)` | CardResult | retire a model card |
| `new_model_version(model_id, fields, new_model_id?)` | CardResult | successor version + lineage link |
| `register_finetuned(model_id, checkpoint_path, base_model_id, …)` | CardResult | **add a fine-tuned model** (lineage to base) |
| `search_features(text, tags?, status?)` | FeaturesResult | free-text/tag search over the feature catalog |
| `list_extractors(category?)` | FeaturesResult | browse the FLOps extractor library |
| `get_feature_lineage(feature_id)` | LineageResult | EFE evolution chain (parent/generation) |
| `update_feature(feature_id, fields)` | CardResult | patch a feature card (re-validated) |
| `deprecate_feature(feature_id, reason?)` | CardResult | retire a feature card |
| `new_feature_version(feature_id, fields, new_feature_id?)` | CardResult | successor version + lineage link |

All re-validate the card against `ModelCard`/`FeatureCard` on write — so seed data must be valid
(linted by `test_catalog_growth.test_every_seed_model_card_validates`). **35 tools total.**

### Legacy removed
The pre-sktime `legacy/` server and its 4 compat tools (`run_tsfm_forecasting`,
`run_tsfm_finetuning`, `run_tsad`, `run_integrated_tsad`) and the 2 static tools (`get_ai_tasks`,
`get_tsfm_models`) are gone. Forecasting → `run_recipe` + a forecaster card; anomaly →
`run_recipe(task=tsfm_anomaly_detection)`; conformal AD comes from sktime `ConformalIntervals`.
The substrate is sktime end-to-end (TTM/Chronos/… resolve through sktime's foundation adapters,
which need `tsfm_public`/`torch` at run time).

## Recipe blocks (run-time params — see RECIPE_SCHEMA.md)
`recipe.estimator.params` / `transforms` (per-component), `recipe.finetune` (training knobs),
`recipe.anomaly` (conformal-AD knobs), `recipe.conformal` (intervals). All carry `param_space`
hints; `run_recipe` records a `param_audit` / `block_audit`.

## Result provenance
Run records carry: `results_file` pointer + summary + provenance (model · features · dataset) +
`param_audit` + `block_audit` + `training_regime`. Anomaly/forecast results hand off downstream
(FMSR → WO → Spot); **no alerts inside TSFM**. State is exportable (`export_state`, #394).

## Tests
`tests/test_tool_surface.py` exercises **every** tool through the real `mcp.call_tool` boundary
(success + validation/error paths), `tests/test_main_filepointers.py` the file-pointer contract,
`tests/test_server.py` the registered surface. No `tsfm_public`/torch required for the suite.
