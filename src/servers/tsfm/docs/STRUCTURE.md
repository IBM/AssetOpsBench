# Code structure

Layered package — each layer depends only on layers above it (core has no intra-package deps).

```
tsfm_server/
  __init__.py            public API + layer map
  config.py              env knobs + collection names (single source of truth)
  server.py              the MCP tool surface (current, recipe-based)
  bootstrap.py           fresh_store() = MemoryStore + seeds

  core/                  the data model + task contracts (no intra-package deps)
    store.py             Store / MemoryStore / CouchStore, make_store()
    schemas.py           ModelCard, FeatureCard (pydantic v2) + enums
    tasks.py             TSTask + TASKS registry (8 standardized TS-AI tasks)

  substrate/             sktime as the execution substrate
    resolver.py          resolve(card) → sktime estimator; discover(); training_regime()

  stores/                catalog = pointer index (models & features are DATA)
    model_store.py       find/get/register/version/lineage + describe_candidates
    feature_store.py     find/register + FLOps select + EFE register (validity-gated)
    results.py           per-task result tables (write_result / get / list)

  reasoning/             evidence the agent reasons from (server decides nothing)
    profile.py           profile_series — facts only (seasonality, stationarity, channels)
    param_space.py       per-model param schema + reasoning hints + validate_params
    feature_selection.py FLOps multi-config selection (|corr|+F-test+MI+model, mean-rank, CD)

  engine/                the recipe engine (the pipeline is composed, not fixed)
    composition.py       run_recipe (forecast + anomaly) + run_tabular_recipe + discover_components
    plan.py              run_plan — recipe DAG, file-pointer chaining (HuggingGPT task-list)
    feature_runner.py    EFE evolved-transform exec (fit/transform/inverse, validity gates)
    evolve.py            AlphaEvolve MAP-Elites archive + ask/tell

  eval/                  GIFT-Eval scoring
    gifteval.py          evaluate_config / evaluate_recipe / leaderboard (MASE+CRPS, geo-mean)

  io/                    data I/O
    refs.py              file-pointer data model (load_series / write_series / materialize_iot)
    window.py            store window reader (read_window from the iot collection)

  tests/                 the test suite (MemoryStore double; TSFM_STORE=memory)
  docs/                  design docs + ARCHITECTURE.svg

  Catalog seed data lives with the other CouchDB collections at
  src/couchdb/scenarios_data/shared/tsfm/{model_catalog,feature_catalog}.json
  (single source of truth; bootstrap reads it, $TSFM_SEEDS_DIR overrides).
  legacy/                pre-sktime modules (operators, pipeline, compute, runner, planner,
                         run_demo, server_legacy) — reference only, superseded by engine/+substrate/
```

## Dependency direction
`core` ← `substrate` ← `stores` ← `reasoning` ← `engine` ← `eval`/`server`. `io` is leaf-level.
Imports are **absolute** (`from tsfm.<layer>.<module> import …`) so a module's location
never depends on who imports it.

## Why this shape
- **core is dependency-free** — the data model and task contracts don't import anything else, so
  they can't pick up cycles.
- **substrate is the only place that knows sktime internals** — swap/upgrade sktime in one layer.
- **stores hold data, engine holds verbs** — adding a model/feature is a card in a store, never a
  new module or tool (HuggingGPT principle).
- **reasoning is isolated and side-effect-free** — it produces evidence + grades; it never runs
  models, matching "the agent decides, the server informs."
- **legacy is quarantined** — superseded code is importable for reference but out of the live path.

## Entry point
`python -m tsfm.server` builds the FastMCP surface (`build_server()`); models/features are
catalog data reached through recipes, so the tool set stays small while capability scales.
