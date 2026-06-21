> 🟡 **HISTORICAL (still mostly valid).** The unified 8-task taxonomy, T-Daub selection, and
> GIFT-Eval framing here all hold. The one update: the estimator/transform **contract is now
> sktime** (not the custom algebra in §1) — see `SKTIME_NATIVE_DESIGN.md` and `STORES.md` for the
> current substrate. Read this for the task taxonomy + selection rationale; read STORES.md for
> the implemented stores.

# TSFM-Kit — a unified MCP server for time-series AI tasks

*A from-scratch synthesis of four IBM systems — AnomalyKiTS (AAAI'22), AutoAI-TS (SIGMOD'21),
FLOps (BigData'20), and Evolutionary Feature Engineering (NeurIPS'26) — into one rigorous,
agent-operable building block that standardizes the full family of TS AI tasks.*

---

## 0. Thesis

Every time-series AI task — forecasting, regression, classification, anomaly detection,
imputation, evaluation, similarity search, clustering — is the **same object**:

```
   Pipeline  =  T_k ∘ … ∘ T_1  →  E  →  T_1⁻¹ ∘ … ∘ T_k⁻¹        (transforms, estimator, inverse)
```

a (possibly invertible) composition of **transforms** feeding an **estimator**, evaluated by a
**task protocol**, and selected under a **compute budget**. The four papers each own one layer
of this object; the MCP server makes the object **agent-operable** and the choices
**benchmark-scorable**. Standardizing on this algebra — rather than a bag of bespoke tools — is
what makes the server strong and extensible.

| Layer of the Pipeline object | Paper | Contribution used |
|---|---|---|
| The transforms `T_i` (with `T_i⁻¹`) | **AutoAI-TS** | Difference / Flatten / Localized / Normalized (T2R), inverse-in-reverse |
| Expanding the transform space | **EFE** | LLM-evolved `fit/transform/inverse` programs, selected by downstream metric |
| Choosing transforms per dataset | **FLOps** | importance scoring vs a Reference Feature + Critical-Difference filter; auto look-back |
| The estimator `E` + std. output + label-free ranking | **AnomalyKiTS** | sklearn Operators/Pipelines; score+label(+1/−1)+contribution; EM/AL ranking |
| Selecting `E`/Pipeline under budget | **AutoAI-TS** | **T-Daub** reverse-progressive data allocation; backtest; zero-conf |

Everything else (model store, feature store, result tables, reasoning, eval-by-state) is the
agentic + persistence machinery built around this object.

---

## 1. The two contracts (everything reduces to these)

**Transform** (feature store unit) — sklearn-compatible, optionally invertible:
```
fit(X, y=None, meta) -> state
transform(X, state) -> X'
inverse_transform(X', state) -> X          # required iff the transform is invertible
```
Kinds: `scaler` (Normalized), `stationarizer` (Difference), `windowizer` (Flatten/T2R/lookback),
`extractor` (FLOps scalar features), `selector` (FLOps/channel), `learned` (EFE-evolved).

**Estimator** (model store unit) — sklearn-compatible, capability-tagged:
```
fit(X, y=None, meta) -> self            # zero-shot models are a no-op fit
predict(X) -> Y           | score(X) -> per-step anomaly score
transform(X) -> embedding | assign(X) -> cluster id
capabilities: [task_id...]   family: stat | ml | hybrid | dl | tsfm | ad_pipeline | encoder
```
Both share the **state/lineage/validity envelope** (provenance, version, parent, metrics,
EFE validity checks). A Pipeline is just `[Transform…] + Estimator`, and AutoAI-TS's invertible
transform stack and AnomalyKiTS's pipelines are *instances* of it.

---

## 2. Standardized TS-AI task taxonomy

A `TSTask` spec declares the contract once; tasks differ only in `inputs`, `output_type`,
`eval_protocol`, `selection`, and `supervised`. This is the standardization the server enforces.

| task_id | inputs | output_type | eval protocol | selection signal | result table |
|---|---|---|---|---|---|
| **forecasting** | X(T×C), horizon H, [exog] | forecast (+quantiles) | rolling **backtest** | MASE / WQL / sMAPE (T-Daub) | forecast_result |
| **regression** | windowed X → y | scalar/vector ŷ | holdout / blocked CV | R²/MAE (FLOps features) | regression_result |
| **classification** | windowed X → class | label + proba | stratified blocked CV | F1 / AUROC | classification_result |
| **anomaly_detection** | X(T×C) | score + label(±1) + contribution | range-based PR/AUROC **or label-free** | AUC-PR / **EM·AL** | anomaly_result |
| **imputation** | X + mask | filled X | **mask-and-score** held-out observed | MAE/CRPS on masked | imputation_result |
| **evaluation** | ŷ, y (or pipeline+data) | metrics | backtest / holdout | n/a (meta) | evaluation_result |
| **similarity_search** | X → embedding; query | top-k neighbors | retrieval P@k / MAP | recall@k | similarity_result |
| **clustering** | {X_i} | assignments | silhouette / ARI | silhouette (label-free) | clustering_result |

Specializations folded in, not duplicated: **prognostics/RUL** = regression with a degradation
target + monotonicity-aware metric; **forecasting_finetune** = forecasting + a write-back of
the trained estimator. Every row is one `TSTask` object → uniform tools, uniform scoring.

Two structural facts a TS researcher will check, handled explicitly:
- **Temporal leakage**: all splits are *blocked/sequential* (no shuffling); transforms `fit` on
  train only; T-Daub uses most-recent slices. Encoded in `eval_protocol`, enforced by the runner.
- **Supervised vs unsupervised**: `supervised=False` tasks (anomaly, clustering, similarity) use
  label-free selection (EM/AL score, silhouette) — AnomalyKiTS's contribution — so selection
  works without ground truth.

---

## 3. Building block I — Feature / Transform Store

Three operations, one library:
- **provide** — `find_features(category, task, modality)`: AutoAI-TS canonical transforms +
  AnomalyKiTS operators + seeds, all under the Transform contract.
- **generate** — `register_feature(program)` (EFE): LLM-evolved transforms, **validity-gated**
  (entry points, no in-place mutation, invertibility round-trip, no leakage) and lineage-tracked
  (`parent_feature_id`, `generation`).
- **select** — `select_features(series, …)` (FLOps): score each candidate extractor on *this*
  dataset, rank vs a **Reference Feature**, keep those past the **Critical-Difference** margin;
  write importance back to the catalog (the catalog *learns* per domain).

Invertibility is first-class (AutoAI-TS/EFE): a forecasting pipeline applies scaler→difference→
flatten forward and the exact inverse stack to map predictions back to physical units. The
store records `interface ∈ {fit_transform, fit_transform_inverse}` and verifies the round-trip
on register.

## 4. Building block II — Model / Estimator Store

A **pointer catalog** of estimators/pipelines spanning AutoAI-TS families (stat/ML/hybrid/DL),
TSFMs (TTM, TSPulse, Chronos, MOMENT, Moirai, TimesFM), AnomalyKiTS AD pipelines (DeepAD/
RelationshipAD/ReconstructAD/WindowAD), encoders (TSPulse embeddings), and registered
fine-tunes. Each card: capabilities (`task_ids`), family, modality, context_length,
input_spec, source (local/hf/toolkit/api), provenance + `base_model_id` lineage, metrics.

Selection is **T-Daub** (AutoAI-TS), not a hand-rank: see §5. Fine-tunes are **agent-decided**
write-backs that point the catalog at a checkpoint; ephemeral tunes never get registered.

## 5. Selection & search — T-Daub + label-free ranking

For supervised tasks, choosing among many candidate pipelines naively (train all on full data)
is infeasible. **T-Daub (reverse progressive data allocation)**:
1. **Fixed allocation** — train each pipeline on increasing *recent* slices `T1[L−mΔ:L]`, score
   on holdout `T2`; **fit a linear learning curve and project the score at full length L**; rank.
2. **Acceleration** — give geometrically increasing data only to the **top** pipeline(s);
   re-project, re-rank.
3. **Score** — return ranked pipelines; train the winner to completion.

This makes "search 10³ catalog models" tractable and is the principled answer to model
selection under budget (sibling of successive-halving/Hyperband, but for sequential TS with a
learning-curve projection on recent data). For **unsupervised** tasks, rank by AnomalyKiTS
**EM/AL** scores or **silhouette** — no labels needed. Conformal calibration (AnomalyKiTS)
sets anomaly thresholds (static otsu / dynamic).

## 6. Evaluation protocols (per task, leakage-safe)

`backtest` (rolling-origin) for forecasting; `blocked_cv` for regression/classification;
`mask_and_score` for imputation (hide observed points, score reconstruction); `range_pr` /
label-free for anomaly; `retrieval@k` for similarity; `silhouette/ARI` for clustering;
`meta` for evaluation. Each `TSTask` carries its protocol + metric set; the runner refuses a
non-blocked split for sequential data.

## 7. Reasoning ownership — agent decides; server is evidence + grader

The server does **not** pre-decide lookback/context/horizon/pipeline/threshold (that would make
the benchmark grade the server, not the agent). Instead:
- **evidence tools** (`profile_series`: detrended dominant period, stationarity, channel
  correlation, length; `available_contexts`, `available_features`) — facts, no advice;
- the **agent** chooses every parameter (tools require them; no silent defaults);
- the server records a factual **param_audit** (e.g. context≥lookback) into the result, and the
  **benchmark** grades the choices vs hidden references (`scoring.py`): lookback ∈ [1×,3×]
  period, context ≥ lookback, horizon == requested, AD pipeline fits channel structure,
  thresholding == dynamic iff non-stationary. (Auto look-back from AutoAI-TS exists as an
  *optional advisor / ablation*, not the default path.)

## 8. Results & state — eval-by-snapshot

Every registry + result table is a CouchDB collection → captured by `export_state()` (#394 /
PR #400). A scenario is scored on the **end-state**: right task object instantiated, right
transforms/model selected (and *why* — the audited parameters), correct result row vs the
utterance's `characteristic_form`, fine-tune registered with correct lineage only when
warranted. This grades the *decision chain*, not just the final number.

## 9. MCP tool surface (grouped)

evidence: `profile_series`, `available_contexts`, `available_features`
feature store: `find_features`, `get_feature`, `select_features`, `register_feature`, `update/deprecate`
model store: `find_models` (T-Daub), `get_model`, `register_model`, lineage/search, `update/deprecate`
run (per task, explicit params): `run_forecast`, `run_regression`, `run_classification`,
  `run_anomaly`, `run_imputation`, `run_similarity`, `run_clustering`, `evaluate`
results: `get_result`, `list_results`
(All on one `FastMCP("tsfm")`; no read-only split; downstream alert/SR/WO/verification live on FMSR/WO/Spot.)

## 10. Package structure

```
tsfm/
  store.py            core registry (Memory|Couch) + export_state
  schemas.py          ModelCard / FeatureCard (pydantic, validated)
  task_spec.py        the TSTask registry — 8 standardized tasks (the standardization)
  transforms/         Transform contract + AnomalyKiTS ops + AutoAI-TS transforms + EFE runner
  feature_store.py    provide / generate(EFE) / select(FLOps) + extractor library
  feature_selection.py FLOps scorer + auto look-back
  model_store.py      estimator catalog + lineage + register
  selector.py         T-Daub reverse progressive allocation (+ label-free EM/AL/silhouette)
  estimators/         StubCompute + RealCompute (tsfm_public, AnomalyKiTS)  by family
  pipeline.py         Transform*→Estimator→Inverse* execution per TSTask
  profile.py          evidence tools (no decisions)
  scoring.py          reasoning rubric + param_audit (benchmark-only)
  results.py          per-task result tables
  runner.py           availability (local/hf/toolkit/api) + leakage-safe splits
  tools.py            MCP tool registration
  seeds/  tests/
```

---

## 11. Self-critique → refinements (the review cycles)

**C1 — "Is forecasting really the same object as clustering?"** Risk: forcing 8 tasks into one
abstraction becomes a leaky generalization. *Refinement:* the shared object is the **Pipeline
(transform→estimator)**; tasks differ only in the estimator's *output verb*
(`predict/score/transform/assign`) and the **protocol**. The `TSTask` spec carries those two
differences explicitly, so the abstraction is honest, not forced. (Validated in `task_spec.py`.)

**C2 — "Invertibility doesn't apply to classification/anomaly."** *Refinement:* invertibility is
a per-transform property (`fit_transform_inverse`), required only when the task maps back to the
input space (forecasting, imputation). Classification/anomaly use forward-only transforms.
The contract makes inverse *optional*; the validity gate only enforces the round-trip when
declared. No over-reach.

**C3 — "T-Daub is forecasting-specific."** *Refinement:* T-Daub's mechanism is *learning-curve
projection under data allocation* — task-agnostic given a per-task score and a blocked split. We
reuse it for regression/classification (CV score) and gate it off for label-free tasks (use
EM/AL/silhouette instead). Selection is thus uniform but signal-appropriate.

**C4 — "Server-side reasoning kills the benchmark."** Caught earlier; *refinement:* moved all
parameter decisions to the agent, server provides evidence + grades (§7). The AutoAI-TS
auto-look-back becomes an *advisor/ablation*, not the default.

**C5 — "Leakage."** *Refinement:* blocked/sequential splits everywhere; transforms fit on train
only; T-Daub on recent slices; runner rejects shuffled CV for TS. Encoded in `eval_protocol`.

**C6 — "Scale of the catalog vs availability."** *Refinement:* catalog is a pointer index;
availability (local/hf/toolkit/api) resolved at call time; unavailable models are still
selectable/plannable but fail compute cleanly — so the catalog can be huge while few are installed.

**C7 — "Naive T-Daub mis-ranks."** Building it surfaced a real bug: linearly *extrapolating* an
exponential learning curve overshoots, and steeper curves (worse pipelines) get the most
optimistic projections, so the true best never enters the accelerated set (verified: it picked
a worse pipeline). *Refinement:* rank the fixed phase by the score at the **largest observed
allocation** (most reliable), accelerate the top-K to full length, and finalize on the
**measured-at-L** score; keep the linear projection only as a reported auxiliary. Now it selects
the true best at ~0.65× the train-all budget on 24 candidates (the saving grows with the catalog
size). This is the realistic regime for T-Daub — the win is large precisely when there are many
pipelines, which is the catalog case.

## 12. Literature map (grounding, beyond the four)

TSFMs: TTM, TSPulse, Chronos, MOMENT, Moirai, TimesFM (zero-shot estimators).
Feature libraries for FLOps: tsfresh, tsfel, catch22, STUMPY.
AD pipelines for AnomalyKiTS: DeepAD, TadGAN, DAGMM, graphical-lasso, conformal AD; benchmarks
TSB-AD, range-based PR (Wu & Keogh's critique → we use range-aware metrics).
Selection lineage for T-Daub: successive halving / Hyperband (we add learning-curve projection
on recent data for sequential TS).
Transform generation for EFE: AlphaEvolve / OpenEvolve (LLM program search).

## 13. Novelty (what makes this *the* TSFM MCP server)

1. **One Pipeline algebra** unifies 8 TS-AI tasks behind a single contract — not a tool zoo.
2. **Feature store = generate(EFE) + select(FLOps) + canonical(AutoAI-TS) operators** under one
   invertible transform contract.
3. **Model store with T-Daub budget-aware selection** over a pointer catalog of TSFMs +
   classical + AD pipelines + fine-tunes.
4. **Agent owns the reasoning** (lookback/context/horizon/pipeline/threshold); server is an
   honest evidence-broker and an objective grader → a genuine test of TS reasoning.
5. **Eval-by-state**: the whole decision chain is CouchDB state, scored deterministically.

This is the standardization a TS-AI agent benchmark has been missing: rigorous enough to be
correct, general enough to cover the field, and operable + scorable as an MCP server.
