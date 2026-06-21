# Core capability — the agentic composition loop (mix-and-match · ensemble · iterate)

The defining capability of the TSFM MCP server: the agent **discovers** all components,
**composes** a recipe (mix-and-match, including ensembles), the server **runs/backtests** it on
the sktime substrate and returns **diagnostics**, and the agent **iterates** to a revised
recipe — a closed loop, GIFT-Eval-style ensemble search driven by the agent. Built on sktime,
persisted as state. (`composition.py`, verified end-to-end on real sktime.)

## The loop (4 MCP tools)

```
 discover_components ─▶ [agent composes a RECIPE] ─▶ run_recipe ─▶ diagnostics
        ▲                                                              │
        └──────────────  agent revises (drop/reweight/swap)  ◀────────┘   (parent_run_id link)
```

1. **`discover_components(task)`** — returns everything composable: installed sktime models for
   the task's scitype, the **foundation forecasters** (TTM, Chronos, MOIRAI, TimesFM, MOMENT,
   TimeMoE, PatchTST, …), catalog models (incl. fine-tunes), transforms (feature store),
   **combiners** (mean/median/min/max/weighted/stack), metrics, splitters. The agent reasons
   over this menu.

2. **The RECIPE** (declarative, the agent authors it):
```jsonc
{ "task": "tsfm_forecasting", "fh": [1..H],
  "transforms": [ {"name":"diff","sktime_class":"...Differencer"} ],     // optional, auto-inverse
  // EITHER a single model …
  "estimator": {"model_id":"ttm_512_96"} | {"sktime_class":"...","params":{}},
  // … OR an ensemble (mix-and-match)
  "ensemble": {"combine":"mean|median|weighted|stack",
               "members":[ {"model_id":"ttm_512_96"}, {"sktime_class":"...ChronosForecaster"},
                           {"sktime_class":"...NaiveForecaster","params":{"strategy":"drift"}} ],
               "weights":[0.5,0.3,0.2]},
  "eval": {"splitter":"expanding","initial_window":...,"step":...,"metrics":["smape","mase"]} }
```

3. **`run_recipe(recipe)`** — compiles to a single sktime forecaster:
   - ensemble → `EnsembleForecaster` (aggfunc / weights) or `StackingForecaster`
   - transforms → `TransformedTargetForecaster` (forward + **automatic reverse-order inverse**)
   then **backtests** via `ExpandingWindowSplitter` + `sktime.evaluate`, fits on full history,
   and returns: `backtest_score`, **`per_member_score`** (each ensemble member alone — the key
   diagnostic for iteration), `forecast_head`, and a persisted `run_id`.

4. **Iterate** — the agent reads `per_member_score`, drops the weak member / reweights toward
   the strong one / swaps in another foundation model, and calls `run_recipe(..., parent_run_id=
   previous)`. Each run is stored in `tsfm_runs` with its parent → the **refinement trajectory
   is state** (and `improved` is reported vs the parent).

## Verified (real sktime, in `test_composition.py` + a live demo)

- discover lists 50+ installed forecasters + the 11 foundation models + combiners.
- single drift model: sMAPE 0.149.
- 4-member mean ensemble: 0.187, with per-member `{last:0.152, mean:0.662, drift:0.149,
  trend:0.210}` — the loop **exposes** that the `mean` member is dragging the ensemble.
- agent iterates (drop `mean`, weight toward `drift`) → **0.150, improved=True**, with the
  parent link persisted. This is the agent making the ensemble better, not the server.

## Why this is the right core capability

- **It is the agent's reasoning surface.** The agent doesn't call a black-box "forecast"; it
  *composes* from a discovered menu and *learns from diagnostics* — exactly the mix-and-match +
  feedback you described, and what separates a strong agent from a naive one (benchmark-able).
- **Ensembles are first-class** (GIFT-Eval): combine foundation models + classical + features by
  mean/median/weighted/stack — the recipe makes this one line, sktime executes it.
- **Closed-loop refinement** with persisted lineage: round 2 uses round 1's per-member scores;
  the trajectory (and whether it improved) is in `tsfm_runs`, so a scenario can score *the
  search*, not just the final number.
- **Substrate-clean**: the recipe compiles to sktime; nothing bespoke runs the models. New
  foundation model = a new catalog card; it's immediately composable.

## How it composes the rest of the stack

- members/estimator resolve through the **model store** (card → sktime class) — fine-tunes and
  remote models are composable too.
- transforms resolve through the **feature store** (sktime transformers / EFE-evolved).
- selection: the agent can compose *by hand* (this loop) **or** call `find_models` (T-Daub) to
  get a ranked shortlist to ensemble — the two compose.
- results + runs land in CouchDB → `export_state()` scoring.

## MCP tools to expose
`discover_components`, `run_recipe`, `get_run`, `list_runs` (lineage). With `find_models`
(T-Daub shortlist) and `profile_series` (evidence) these give the agent: see the data → see the
menu → compose → run → diagnose → iterate.

## Files
`composition.py` (discover/build/run/iterate), `tests/test_composition.py` (3 tests),
`sktime_resolver.py` (card→sktime), `selector.py` (T-Daub shortlist to feed ensembles).
