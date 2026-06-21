# Model Store & Feature Store — definitive specification

The two registries at the heart of the TSFM MCP server. Both are **sktime-native card
catalogs** on the CouchDB core (`store.py`): an entry is a small JSON **card** that points at
an sktime estimator/transformer (or a sklearn/PyOD estimator via an adapter, or a custom
`Base*` subclass). sktime resolves and **downloads** the weights; the store never holds data or
weights — only cards.

## 0. Principles (both stores)

1. **Card, not code/weights.** A card = identity + a pointer (`sktime_class` + `params`) +
   metadata. `resolve(card)` imports and instantiates the sktime object; sktime downloads any
   weights on first use.
2. **Catalog ⊇ sktime registry.** The CouchDB catalog is a *superset* of sktime's in-memory
   `all_estimators`: it also holds **not-installed**, **remote**, and **fine-tuned** entries
   with provenance/lineage/metrics, is **agent-queryable**, and is **state-exportable** (#394).
3. **Data is out of scope.** IoT data is **not** in the stores. The agent downloads a window to
   CSV/Parquet and passes a **file pointer** (`data_ref`) in the recipe; `io_refs.load_series`
   resolves it to an sktime container. Stores hold cards; the recipe carries data pointers.
4. **Three complementary selection rankers** over one catalog: **description** (HuggingGPT),
   **tags** (sktime), **budget/metric** (T-Daub / GIFT-Eval).
5. **Validated, versioned, lineage-tracked.** Every card is pydantic-validated on write; updates
   are append-aware; versions supersede with links; provenance records parentage.

---

## 1. MODEL STORE  (`model_store.py`, collection `model_catalog`)

### 1.1 ModelCard schema (every field)

```jsonc
{
  // ── identity ──
  "_id": "model:ttm_512_96", "model_id": "ttm_512_96",
  "version": "r2", "status": "active|deprecated|experimental|superseded",
  "created_by": "...", "created_at": "...", "updated_at": "...",

  // ── substrate (how to instantiate; sktime-native) ──
  "scitype": "forecaster",                       // forecaster|regressor|classifier|detector|clusterer|transformer
  "sktime_class": "sktime.forecasting.ttm.TinyTimeMixerForecaster",
  "params": {"context_length": 512, "prediction_length": 96},
  "tags": {"capability:global_forecasting": true, "capability:pred_int": false},  // sktime tags
  "soft_deps": ["torch","transformers","tsfm_public"],
  "source": "sktime|huggingface|local|toolkit|api|sklearn_adapter",
  "availability": "downloadable|installed|remote|unavailable",   // resolved at call time

  // ── capabilities (selection + applicability) ──
  "task_ids": ["tsfm_forecasting","tsfm_forecasting_evaluation"],
  "modality": "timeseries", "output_type": "forecast|score|class|embedding|assignment|value",
  "usage_modes": ["zero_shot","fine_tune"],
  "context_length": 512, "prediction_length": 96,
  "input_spec": {"channels":"multivariate","min_length":512,"requires_y":false},
  "domain": "general", "frequency": "any",

  // ── selection metadata (HuggingGPT) ──
  "description": "IBM Granite TinyTimeMixer foundation forecaster; strong zero-shot, long context.",
  "downloads": 50000, "likes": 120,
  "metrics": [{"metric":"mase","value":0.83,"dataset":"gifteval","vs":"seasonal_naive"}],

  // ── provenance / lineage ──
  "provenance": "pretrained|finetuned|trained|external_hf|external_service|toolkit",
  "base_model_id": null, "supersedes": null, "superseded_by": null,
  "fine_tune": {"dataset":"chiller_6","config":{"epochs":50},"checkpoint":"file://..."},

  // ── pointer (where weights live; any one) ──
  "artifact_path": null, "hf_repo": "ibm-granite/granite-timeseries-ttm-r2",
  "remote_endpoint": null, "model_checkpoint": "ttm_512_96"
}
```

### 1.2 Lifecycle (write)
- `register_model(card, overwrite=False)` — pydantic-validate (`schemas.ModelCard`), reject
  duplicates; a `finetuned` card **must** carry `base_model_id`.
- `update_model(id, fields)` — patch; `metrics` are **appended**, not replaced; stamps `updated_at`.
- `deprecate_model(id, reason)` — soft delete (`status=deprecated`).
- `new_version(id, fields)` — register vN+1, mark predecessor `superseded` + cross-link.
- `register_finetuned(...)` — the agent-decided write-back: point the catalog at a fine-tune
  checkpoint with `provenance=finetuned`, `base_model_id`, `metrics`. (Ephemeral tunes are never
  registered — keeping the catalog curated.)

### 1.3 Read & select
- `get_model(id)`, `list_models(task_id, domain, modality, framework, usage_mode, status)`.
- **`find_models(task_id, min_context_length, prediction_length, domain, top_k, explain)`** —
  structured filter + ranked shortlist (domain match → eval MAE → context). Feed the shortlist
  to **T-Daub** (`selector.tdaub_select`) for budget-aware final ranking on real data.
- **`describe_candidates(task_id, top_k, domain)`** — the HuggingGPT surface: compact
  `{model_id, description, downloads, family, sktime_class, context_length, tags}` ranked by a
  popularity/quality prior; the `{{Candidate Models}}` the agent reasons over to pick/ensemble.
- `search(text, tags)`, `get_lineage(id)` (ancestors via `base_model_id` + descendants).

### 1.4 Resolution & execution
`sktime_resolver.resolve(card)` → `import sktime_class; Est(**params)`. `run(card, task, …)`
dispatches by scitype verb (predict/score/transform/assign). sktime downloads weights lazily;
`availability` is computed at call time (installed / downloadable / remote / unavailable —
unavailable cards stay discoverable + plannable, fail compute cleanly).

### 1.5 Seeded families (one catalog, many kinds)
statistical (NaiveForecaster=**zero model**, AutoARIMA, BATS, HW), ML/reduction (WindowRF/SVR),
DL, **TSFMs** (TTM, Chronos/Chronos2, MOIRAI, TimesFM/2, MOMENT, TimeMoE, PatchTST, LagLlama),
**AD** (PyODDetector→IsolationForest/LOF, SubLOF, **TSPulseAnomalyDetector**, AnomalyKiTS
DeepAD/RelationshipAD/ReconstructAD/WindowAD), **classifiers** (**TSPulseClassifier**, Rocket),
**clusterers** (TimeSeriesKMeans), and the **ConformalIntervals** wrapper (probabilistic for
any forecaster). All verified to resolve from a card. (`seeds/sktime_method_cards.json`.)

---

## 2. FEATURE STORE  (`feature_store.py`, collection `feature_catalog`)

A catalog of two `kind`s, both on the sktime transform contract (`fit/transform[/inverse]`):
**transforms** (sktime `BaseTransformer` or EFE-evolved program code) and **extractors**
(FLOps scalar/vector/functional features).

### 2.1 FeatureCard schema

```jsonc
{
  "_id":"feature:efe_time_robust_norm_v1", "feature_id":"efe_time_robust_norm_v1",
  "kind":"transform|extractor", "name":"...", "modality":"timeseries",
  "interface":"fit_transform|fit_transform_inverse", "class_name":"Transformation",
  "invertible": true,
  // implementation: EITHER stored code (EFE) OR an sktime/extractor pointer
  "code":"import numpy as np\nclass Transformation: ...",            // EFE program
  "sktime_class":"sktime.transformations.series.difference.Differencer",  // sktime transform
  "extractor_name":"slope",                                          // FLOps library ref
  "output_cardinality":"series|scalar|vector|functional",
  // selection
  "scenario_categories":["Future State Prediction"], "scenario_types":["Forecasting"],
  "target_task":"tsfm_forecasting", "target_model": null, "dataset": null,
  "description":"Invertible robust median/IQR normalization for TSFM inputs.",
  "metrics":[{"metric":"flops_importance","value":0.61,"dataset":"..."},
             {"metric":"mase_gain_vs_identity","value":0.07}],
  // provenance / lineage / validity
  "provenance":"handwritten|evolved|library", "method":"EFE-Time",
  "parent_feature_id": null, "generation": 0,
  "validity":{"entry_points":true,"no_inplace":true,"invertible_ok":true,"leakage_checked":true},
  "status":"active", "version":"1", "created_by":"seed", "created_at":"..."
}
```

### 2.2 Three operations (the three papers)
- **provide** — `find_features(category, target_task, target_model, kind)`, `list_extractors`,
  `get_feature`. The agent reads candidate transforms (with descriptions) to pick.
- **generate** (EFE) — `register_feature(card)`: pydantic-validate **and** run the EFE validity
  gate via `feature_runner` (entry points, **no in-place mutation**, **invertibility round-trip**,
  schema/leakage) before accepting; lineage via `parent_feature_id`/`generation`. `new_version`
  for evolution chains.
- **select** (FLOps) — `select_features(series, reference_feature, cd_margin)` ranks extractors
  on *this* data vs a **Reference Feature** with a **Critical-Difference** cut + auto look-back;
  `select_features_from_catalog(..., write_back=True)` restricts to catalog candidates and
  **writes the importance back** onto extractor cards (the catalog learns per domain).

### 2.3 Extractor library (FLOps 130+, typed)
`register_extractor_library` indexes the FLOps extractors (Data Profiling, Data Quality,
temporal, frequency) as `kind=extractor` cards with `output_cardinality` ∈
{scalar, vector, functional} and `scenario_categories`. `select_features` chooses among them.

### 2.4 Invertibility & lifecycle
Transforms that map back to input space carry `interface=fit_transform_inverse`; the gate
verifies the round-trip. Same `register/update/deprecate/new_version/search/get_lineage` as the
model store.

---

## 3. How the stores plug into the recipe + file pointers

```
agent: download IoT window → CSV → data_ref (file pointer)
recipe step:
  transforms: [ {feature_id:"efe_time_robust_norm_v1"} | {sktime_class:"...Differencer"} ]
  model/ensemble members: [ {model_id:"ttm_512_96"} | {sktime_class:"...ChronosForecaster"} ]
run: resolve cards → sktime objects; load_series(data_ref) → fit/predict on sktime;
     write outputs as file pointers; score via GIFT-Eval; persist run (lineage).
selection: describe_candidates (HuggingGPT) → agent picks/ensembles, OR find_models → T-Daub shortlist.
```
Members/transforms resolve through the stores; data flows by pointer; results are pointers +
CouchDB rows → all state-exported for scoring.

## 4. CouchDB collections & seeds
`model_catalog` (pk model_id), `feature_catalog` (pk feature_id); seeds in `seeds/*.json`
(sktime method cards, EFE transforms, anomalykits). Loaded by `bootstrap.load_seeds` (Memory)
or `init_data` (CouchDB). Both stores + the result/run tables ride along in `export_state()`.

## 5. Deliberate scope / critique
- **No data in the store** — IoT data is file pointers in the recipe; this keeps the stores
  small, cacheable, and shareable, and lets GB windows move by reference (your instruction).
- **No bespoke estimator framework** — sktime base classes + tags + registry are the contract;
  the `operators.py` algebra is **superseded** (kept only as conceptual notes; sktime realizes
  it). A model not in sktime is added as a custom `Base*` subclass and pointed at by a card.
- **Soft deps**: cards list them; uninstalled methods are discoverable/plannable, fail compute
  cleanly (the pointer-catalog rule). sktime downloads installed-but-not-cached weights.
- **Selection is plural by design** — description / tags / metric rankers answer different
  questions (what does it do / does it fit the data shape / does it actually win); the agent
  combines them.

## Files
`model_store.py`, `feature_store.py`, `schemas.py` (cards), `sktime_resolver.py` (resolve +
download), `selector.py` (T-Daub + label-free), `feature_selection.py` (FLOps), `feature_runner.py`
(EFE gate), `seeds/`. Companion design: `SKTIME_NATIVE_DESIGN.md`, `CAPABILITY_MATRIX.md`,
`CORE_CAPABILITY.md`, `GIFTEVAL_APPROACH.md`, `HUGGINGGPT_FILEPOINTER_DESIGN.md`.
