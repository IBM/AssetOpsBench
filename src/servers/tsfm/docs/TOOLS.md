# TSFM MCP server — tool surface & capabilities

**Design rule (HuggingGPT): models/features are *data* (catalog cards), not tools.** The agent
gets a small, stable set of **~16 MCP tools** and composes the *large* catalog through recipes.
You do not add a tool per model — adding a model is adding a card.

## How many tools: 16 (in 6 groups)

| # | Tool | Group | Backed by | What it does |
|---|---|---|---|---|
| 1 | `list_tasks` | Discover | task_spec | the 8 TS-AI tasks + each one's contract (inputs, scitype, eval protocol, supervised?) |
| 2 | `discover_components` | Discover | composition | the menu for a task: installed models, foundation models, feature transforms, combiners, metrics, splitters |
| 3 | `describe_candidates` | Discover | model_store | HuggingGPT model-selection surface: top-K cards by description+downloads |
| 4 | `find_models` | Discover | model_store | structured filter (task/context/domain/tags) → ranked shortlist (feeds T-Daub) |
| 5 | `find_features` | Discover | feature_store | feature transforms by scenario category / task / model |
| 6 | `get_component` | Discover | model_store/feature_store | full card for a model_id or feature_id (incl. lineage) |
| 7 | `profile_series` | Evidence | profile | facts about a `data_ref` (seasonality, stationarity, channels, length) — no decisions |
| 8 | `select_features` | Feature-learn | feature_store (FLOps) | dynamic, dataset-specific feature selection (reference + Critical-Difference) |
| 9 | `run_recipe` | Compose+run | composition | compile a recipe (transforms + single/ensemble + conformal) → sktime → backtest + forecast + per-member diagnostics |
| 10 | `run_plan` | Compose+run | plan | execute a recipe **DAG** (HuggingGPT task-list: `dep` + `@resource` file-pointer chaining) |
| 11 | `evaluate` | Compose+run | gifteval | GIFT-Eval leaderboard: seasonal-naive-normalized MASE+CRPS, geo-mean, mean-rank over configs/recipes |
| 12 | `finetune` | Compose+run | composition/sktime | fine-tune a base model on a `data_ref` → returns a checkpoint **file pointer** (registers nothing) |
| 13 | `register_model` | Write-back | model_store | agent-decided: point the catalog at a fine-tune checkpoint (provenance + lineage) |
| 14 | `register_feature` | Write-back | feature_store | add an evolved transform (EFE), validity-gated; lineage |
| 15 | `get_result` / `list_results` | Results | results | per-task result tables (forecast/anomaly/classification/…) |
| 16 | `get_run` / `list_runs` | Results | composition/plan | run + plan **lineage** (the iterate trajectory) |

> Tools 6, 15, 16 each pair a get/list — count them as one tool each (overloaded). Net **16
> tools**. All register on one `FastMCP("tsfm")`; no read-only split.

### The whole agent loop uses just these
```
list_tasks / discover_components / describe_candidates / find_models / find_features / profile_series   ← see the menu + data
select_features                                                                                          ← FLOps
run_recipe / run_plan / evaluate                                                                         ← compose, run, score (GIFT-Eval)
register_model / register_feature / finetune                                                            ← improve & persist
get_result / get_run                                                                                     ← inspect & iterate
```

## Capabilities (the "many", reached via the 16 tools)

Tools stay ~16; the **capability surface scales through the catalog** (verified counts on the
installed sktime 1.0.1 + our seeds):

| Capability axis | Count / content |
|---|---|
| **TS-AI tasks** | **8**: forecasting, regression, classification, anomaly_detection, imputation, evaluation, similarity_search, clustering |
| **Forecasters** (incl. 11 foundation models) | **141** — TTM, Chronos/Chronos2, MOIRAI, TimesFM/2, MOMENT, TimeMoE, PatchTST, LagLlama, AutoARIMA, BATS, … |
| **Anomaly detectors** | **28** — PyOD zoo (IsolationForest/LOF/…), SubLOF, **TSPulseAnomalyDetector**, AnomalyKiTS pipelines |
| **Classifiers** | **77** — incl. **TSPulseClassifier**, Rocket |
| **Transformers (feature ops)** | **148** sktime + **FLOps 130+ extractors** + EFE-evolved |
| **Clusterers** | **10** (TimeSeriesKMeans, …) |
| **Ensemble combiners** | **6**: mean, median, min, max, weighted, stack |
| **Probabilistic** | conformal intervals on **any** forecaster (69 expose intervals natively) → CRPS |
| **Metrics** | MASE, sMAPE, MAPE, MAE, CRPS (GIFT-Eval point + probabilistic) |
| **Splitters** | expanding / sliding window backtest |
| **Selection rankers** | 3 — description (HuggingGPT), tags (sktime), budget T-Daub |

So: **16 tools** expose **8 tasks × hundreds of models/transforms × ensembles × conformal ×
GIFT-Eval scoring** — because the catalog (not the tool list) holds the variety.

## Why 16 and not 100

- **One tool per model = unmaintainable + token-bloated.** HuggingGPT's lesson: keep the
  controller's tool set tiny; put models in a catalog the controller *reads*. Adding TTM, a new
  PyOD detector, or an evolved feature = a catalog card, **zero new tools**.
- **The recipe is the universal verb.** `run_recipe`/`run_plan` execute *any* composition
  (single model, ensemble, +conformal, multi-step DAG) — so you don't need `run_ttm`,
  `run_chronos`, `run_ensemble`, … separately.
- **Selection is reasoning, not a tool per strategy.** `describe_candidates` + `find_models`
  + `select_features` give the agent the evidence; the agent decides; `evaluate` (GIFT-Eval)
  scores. The server stays a broker + grader.

## Optional trims / adds
- **Minimal (10)**: drop `finetune`, `register_*`, `get_run/list_runs`, `list_tasks` for a
  read+compose-only profile (no catalog mutation, no fine-tune).
- **Extended (+2)**: `data_profile_multi` (profile many `data_ref`s for fan-out) and
  `explain_selection` (return the per-decision rationale for audit/benchmark scoring).

## Source / status
Tools 2,3,4,5,7,8,9,10,11,13,14 are backed by **implemented + tested** functions
(`composition.py`, `plan.py`, `gifteval.py`, `model_store.py`, `feature_store.py`,
`profile.py`, `feature_selection.py`). `server.py` should register exactly this set on
`FastMCP("tsfm")` (it currently exposes the earlier pre-recipe set — update to these 16).
See `DOCS_INDEX.md` for the full design map.
