# sktime-native redesign — sktime as the substrate

**Decision (accepting the critique): drop the hand-rolled Operator algebra; build on sktime.**
Different foundation models do take different code paths — and sktime has already solved that:
all 11 TSFMs are `BaseForecaster`s behind one fit/predict API. Reinventing that contract was
the mistake. The MCP server becomes a thin agentic **catalog + selection + reasoning +
persistence** layer over sktime, not a new ML framework.

Verified in this sandbox (sktime 1.0.1): a catalog *card* resolves+fits+predicts for a naive
forecaster, a transformer, **and `TinyTimeMixerForecaster` — through the identical mechanism**;
the registry exposes 141 forecasters, 28 detectors, 148 transformers, and these foundation
forecasters: Chronos/Chronos2, HFTransformers, LagLlama, MOIRAI, MomentFM, PatchTST, TimeMoE,
TimesFM/TimesFM2, TinyTimeMixer.

## 1. sktime IS the building block (what we no longer write)

| We were hand-building | sktime already provides |
|---|---|
| Operator role taxonomy | **scitypes**: forecaster, global_forecaster, classifier, regressor, clusterer, transformer, detector, aligner, early_classifier, splitter, metric, param_est, reconciler |
| Estimator/Transform fit/apply/inverse contract | `BaseForecaster/BaseClassifier/BaseRegressor/BaseClusterer/BaseTransformer/BaseDetector` (`_fit/_predict/_transform/_inverse_transform`) |
| Model registry + capability tags | `all_estimators(estimator_types, filter_tags)`, `all_tags`, the **tag** system |
| Invertible transform stacks (AutoAI-TS) | `TransformedTargetForecaster`, `ForecastingPipeline`, `Differencer`, `Imputer` (built-in `inverse_transform`, reverse order) |
| Windowizer/T2R + ML reduction | `make_reduction` (recursive/direct/multioutput) |
| Backtest / evaluation | `ExpandingWindowSplitter`, `SlidingWindowSplitter`, `evaluate`, metrics (MASE, sMAPE, …) |
| Foundation-model adapters | `TinyTimeMixerForecaster`, `ChronosForecaster`, `MOIRAIForecaster`, `TimesFMForecaster`, … |
| Feature extractors (FLOps library) | tsfresh / catch22 / `SummaryTransformer` / `Catch22` are sktime transformers |

So the 8 standardized TS-AI tasks **ride on sktime scitypes** (validated by the live map):
forecasting→forecaster, regression→regressor, classification→classifier, anomaly→detector,
imputation→transformer(Imputer), clustering→clusterer, similarity→transformer+distances,
evaluation→metric+splitter.

## 2. The card = a pointer to an sktime estimator

```jsonc
{ "model_id": "ttm_512_96", "scitype": "forecaster",
  "sktime_class": "sktime.forecasting.ttm.TinyTimeMixerForecaster",
  "params": {"context_length": 512, "prediction_length": 96},
  "tags": {"capability:global_forecasting": true, "capability:pred_int": false},
  "source": "sktime", "provenance": "pretrained", "base_model_id": null,
  "metrics": [], "status": "active" }
```
`resolve(card)` = import `sktime_class` + instantiate with `params`; `run(card, task, …)` calls
the scitype's verb. A TTM and an ARIMA differ only in `sktime_class`/`params` — same code path.
(`sktime_resolver.py`, tested.)

## 3. What the MCP server adds (sktime does NOT provide these)

1. **Persistent catalog = a superset of sktime's registry.** sktime's `all_estimators` lists
   *installed* estimators in-process. Our CouchDB **model_catalog**/**feature_catalog** also
   hold not-installed/remote/**fine-tuned** models with provenance, lineage, metrics; are
   agent-queryable; and are **state-exportable** (#394) for scoring. `find_models` =
   catalog query (∪ `all_estimators` for what's installed) filtered by sktime tags.
2. **T-Daub selection over sktime.** sktime has tuning (grid/random) but not AutoAI-TS's
   reverse-progressive **data allocation**; our `selector.py` runs T-Daub *using
   `sktime.evaluate` + a splitter* as the scoring function — budget-aware ranking over sktime
   estimators. Unsupervised → label-free EM/AL/silhouette.
3. **Agentic layer**: MCP tools + the **reasoning split** (evidence `profile_series` → agent
   chooses lookback/context/horizon/model/threshold → `param_audit` + `scoring.py`). sktime is
   a library, not an agent interface.
4. **AnomalyKiTS** = custom sktime **`BaseDetector`s** (DeepAD/RelationshipAD/ReconstructAD/
   WindowAD) + EM/AL ranking; **EFE** = custom **`BaseTransformer`s** (evolved, with
   `inverse_transform`) registered into the feature store; **FLOps** = sktime feature
   transformers + our reference-feature/CD selector.
5. **Eval-by-state** + AssetOpsBench scenario integration (result tables in CouchDB).

## 4. Mapping the four papers onto sktime

- **AutoAI-TS**: transforms → sktime `Differencer`/`LogTransformer`/`Imputer` +
  `TransformedTargetForecaster` (inverse-in-reverse is built in); pipeline families → sktime
  estimators (ARIMA, AutoETS, BATS via tbats, RandomForest via `make_reduction`); **Zero Model**
  → `NaiveForecaster(strategy="last")`; **T-Daub** → our selector over `sktime.evaluate`.
- **AnomalyKiTS**: 4 pipelines → custom `BaseDetector`s in the `detector` scitype; Static/
  Dynamic thresholding → detector params; EM/AL → our label-free selector; contribution → the
  detector's output annotation.
- **FLOps**: 130+ extractors → sktime feature transformers (`Catch22`, `TSFresh*`,
  `SummaryTransformer`); selection (reference feature + CD) → our meta-selector.
- **EFE**: evolved `fit/transform/inverse` programs → custom `BaseTransformer` subclasses,
  validity-gated, archived in the feature store; inserted via `ForecastingPipeline`.

## 5. Revised module structure (sktime-native)

```
tsfm/
  store.py            CouchDB|Memory catalog core + export_state            [keep]
  schemas.py          ModelCard/FeatureCard — card now carries sktime_class + params + tags  [revise]
  task_spec.py        8 tasks, each bound to an sktime SCITYPE              [revise: add scitype]
  sktime_resolver.py  card → sktime estimator; run via scitype verb; registry discovery  [NEW — replaces runner.py + operators.py]
  model_store.py      catalog (superset of all_estimators) + tag filter + lineage + register  [keep, tag-aware]
  feature_store.py    sktime transformers + EFE(custom BaseTransformer) + FLOps select  [revise]
  selector.py         T-Daub over sktime.evaluate + label-free                [keep, wire to sktime]
  detectors/          AnomalyKiTS BaseDetector implementations               [NEW]
  transforms/         EFE BaseTransformer implementations + FLOps selector    [NEW]
  pipeline.py         build sktime ForecastingPipeline/TransformedTargetForecaster per task  [revise]
  profile.py scoring.py results.py tools.py seeds/ tests/                    [keep]
  # operators.py  → REMOVED (sktime base classes supersede it)
```

## 6. Critique of the sktime pivot (eyes open)

- **Data containers.** sktime is opinionated about I/O (`pd.Series`/`pd.DataFrame` with
  PeriodIndex/DatetimeIndex; panel mtypes for classification). We need a thin **data adapter**
  from CouchDB `iot`/`vibration` rows → sktime mtypes (and `check_is_mtype`/`convert`). This is
  real work but localized.
- **Detector scitype maturity.** sktime's anomaly/segmentation (`detector`) API is younger than
  forecasting; AnomalyKiTS pipelines become custom detectors — acceptable, but pin the API.
- **Soft dependencies / weight.** each FM pulls its own stack (torch, transformers, gluonts,
  jax for TimesFM). Keep them optional; the catalog lists them but availability resolves at
  call time (unchanged from our pointer-catalog design). Pin sktime + per-model extras.
- **`fit` semantics.** foundation models require `fit()` for API consistency (often a no-op).
  Our `run()` calls it — matches sktime; zero-shot is just a no-op fit.
- **What sktime does NOT cover** (so we still build): persistent agent-queryable catalog,
  T-Daub, EM/AL, the reasoning/eval-by-state layer, MCP. These are the genuine contributions;
  the substrate is sktime.

## 7. What we drop, honestly

`operators.py` (the typed algebra) is **superseded** — sktime's base classes + tags + pipelines
+ registry already provide a mature, community-maintained version of exactly that contract,
with 11 foundation-model adapters we'd otherwise have to write and maintain. We keep the
*ideas* it encoded (leakage-safe fit/transform, invertibility, quality gating, zero-model) —
but realize them via sktime (`fit/transform`, `inverse_transform`, soft-dep checks,
`NaiveForecaster`) instead of a bespoke framework.

## Files
`sktime_resolver.py` (card→sktime, verified), `task_spec.py` (tasks↔scitypes), the catalog/
selection/reasoning layers (unchanged), this design. Next: data adapter (CouchDB→mtype),
AnomalyKiTS detectors, EFE transformers, and wiring `selector.py` to `sktime.evaluate`.
