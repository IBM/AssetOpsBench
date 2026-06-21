# Feature extraction (FLOps) — the plan, on sktime

FLOps = **extract → score → rank vs a Reference Feature → Critical-Difference filter → reuse**,
done dynamically per dataset. On the sktime substrate almost every piece already exists; our
job is the *selection/learning* layer + persistence + the agent loop. (Verified: sktime
extractors + `FeatureUnion` run; our `feature_selection.select_features` picks among the library.)

## 1. The FLOps pipeline (5 stages → sktime mapping)

| FLOps stage | What it does | sktime / our piece |
|---|---|---|
| **Library** | 130+ extractors (scalar/vector/functional), grouped (Data Profiling, Data Quality, temporal, frequency) | `Catch22`, `TSFreshFeatureExtractor`, `SummaryTransformer`, `FourierFeatures`, `WindowSummarizer`, `Signature*`, `(Mini)Rocket` + our FLOps refs (`feature_selection.EXTRACTORS`) + EFE-evolved |
| **Extract** | apply the chosen extractors, concatenate | `FeatureUnion([... ])` (compose many) → tabular |
| **Score** | per-feature importance, multiple configs (F-test / MI / model importance) | sktime `FeatureSelection` (filter/wrapper) + `TSFreshRelevantFeatureExtractor` (built-in hypothesis-test relevance) + our `select_features` |
| **Rank + filter** | rank vs a **Reference Feature**, keep those past the **Critical-Difference** margin | our `feature_selection` (reference + CD) — the FLOps signature step |
| **Reuse** | persist the selected set; reapply on new data | a fitted **feature_set card** = `FeatureUnion(selected) → ColumnSelect`, lineage-tracked |

## 2. Dynamic, dataset-specific selection (the core FLOps idea)

`select_features(series, reference_feature, cd_margin)` (built, tested):
1. discover **look-back** from spectral period (FLOps: `lw ≈ 1.25 × seasonal length`),
2. tabulate, score each candidate extractor on *this* dataset,
3. rank vs the Reference Feature, keep those beating it by the CD margin,
4. (`select_features_from_catalog(..., write_back=True)`) write the importance back onto the
   extractor cards so the catalog **learns** which features matter per domain.

So the feature set is chosen *for the data in front of the agent*, not fixed.

## 3. Where extracted features go, per task (data-shape aware)

- **classification / regression / clustering / similarity** (series → tabular): extraction is
  the *main* transform → `Tabularizer`/`FeatureUnion` → the estimator (or via `make_reduction`).
  RUL/PHM regression is exactly this (FLOps' original setting).
- **forecasting**: FLOps features become **exogenous regressors** `X` (or reduction features)
  alongside the look-back windowizer — sktime `ForecastingPipeline` / `make_reduction`.
- **anomaly**: residual / spectral / trend features feed the detector (AnomalyKiTS pipelines).

`output_cardinality` on each FeatureCard (series / scalar / vector / functional) tells the
recipe compiler how the shape changes, so composition is type-checked.

## 4. In the recipe (agent mixes-and-matches features too)

```jsonc
{ "task":"tsfm_regression",
  "transforms":[ {"feature_id":"channel_select_v1","params":{"channel_indices":[0,1,2]}},
                 {"sktime_class":"sktime.transformations.panel.catch22.Catch22"},
                 {"feature_id":"flops_selected_set@chiller_6"} ],   // a persisted FLOps set
  "estimator":{"sktime_class":"...RandomForestRegressor via reduction"},
  "eval":{"metrics":["r2","mae"]} }
```
The agent: `find_features(category)` → `select_features(data_ref)` (FLOps) → compose the chosen
extractors into the recipe → `run_recipe` (sktime) → CV / GIFT-Eval score → iterate.

## 5. EFE generation when the library isn't enough

If no library extractor scores above the reference, the agent (EFE) **generates** a new
`fit/transform` extractor (custom sktime `BaseTransformer`), which is validity-gated
(entry points, no in-place, no leakage), archived with lineage (`parent_feature_id`,
`generation`), and re-entered into selection. Library-first, generate-on-demand.

## 6. Parameter reasoning for extractors (same as models)

Extractors have parameters too (`window_size`, `n_jobs`, `Catch22` feature subset, `FourierFeatures`
sp_list/fourier_terms). Each FeatureCard carries a `param_schema` (introspected) + hints
(`window_size ≈ 1× dominant_period`, `sp_list = [dominant_period]`); the agent reasons them from
`profile_series`; `validate_params` checks them. (Same machinery as `param_space.py`.)

## 7. Output: a reusable, scored, lineage-tracked feature set

A FLOps run yields a **feature_set card** (`kind=feature_set`): the selected extractors +
`ColumnSelect`, with `metrics:[{flops_importance}, {cv_gain_vs_baseline}]`, `dataset`,
`scenario_categories`, lineage. It is registered (validity-gated), discoverable by
`find_features`, and re-applicable to new windows — so selection is *amortized*, not redone
every call.

## 8. Build steps

1. **Seed the extractor catalog** from sktime feature transformers (Catch22, TSFresh, Summary,
   Fourier, WindowSummarizer, Rocket) + the FLOps library refs — `register_extractor_library`
   (done for the FLOps refs; add the sktime classes as cards).  *[small]*
2. ✅ **FLOps selector v2** (DONE): multi-config scoring (|corr| + F-test + mutual-info +
   model-importance) aggregated by **mean rank**, on top of reference + CD. `select_features`
   takes `scorers=[…]` (default all four; `["corr"]` = fast path) and returns `per_scorer`
   detail. 4 tests. *(next: wire sktime `TSFreshRelevant`/`FeatureSelection` as extra scorers.)*
3. **Persist feature sets** (`kind=feature_set`) with importance + CV gain + lineage; expose via
   `find_features`. *[small]*
4. ✅ **Recipe compile for extraction — tabular** (DONE): `run_tabular_recipe` compiles a
   FeatureUnion of extractors (our dependency-free library / sktime panel transformers /
   `flops_select` column selection) → estimator for regression / classification / clustering;
   CV-scored (accuracy/r2) or silhouette; same recipe grammar + persistence + regime as
   forecasting. 5 tests. *(next: exogenous-feature path for forecasting.)*
5. **Param reasoning for extractors** (param_space hints for window_size/sp_list/etc.). *[small]*
6. **EFE generate-on-demand** when library < reference. *[later]*

## 9. Why this is strong
- The extractor **library is sktime's** (battle-tested: Catch22/TSFresh/Rocket) — we don't
  reimplement features; we **select** them (FLOps) and **generate** new ones (EFE).
- Selection is **dynamic + dataset-specific + principled** (reference + CD), **persisted**
  (amortized), and **graded** (importance + CV/GIFT-Eval gain) — the full FLOps contribution,
  agent-operable.
- It composes into the same recipe/DAG and is scored the same way (GIFT-Eval / CV), so feature
  engineering is part of the agent's mix-and-match search, not a separate pipeline.

## Files
`feature_selection.py` (FLOps selector + look-back), `feature_store.py` (catalog +
register_extractor_library + select_features_from_catalog), `feature_runner.py` (EFE gate),
`param_space.py` (extractor params). Design: `STORES.md` §2, `DESIGN.md` (FLOps mapping).
