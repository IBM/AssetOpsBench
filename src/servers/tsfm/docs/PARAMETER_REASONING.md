# Per-model parameter reasoning

Every model has its own parameters, and the agent must **reason a value for each** — not accept
defaults. The server exposes a parameter **schema with reasoning hints**, the agent fills the
values from data evidence, and the server **validates** (and the scorer grades) the choices.
Built + tested (`param_space.py`).

## The card carries a parameter schema (auto + curated)

For any card, `param_schema(card)` returns, per parameter:
- **auto-introspected** from the sktime class constructor: `default`, `required`, `type`, plus
  sktime's `get_test_params()` example configs;
- **curated `param_hints`**: a `description`, what data evidence it `depends_on`, a `suggest`
  rule, and an allowed `range`/`choices`.

Example (real introspection):

| model | parameter | default | reasoning hint |
|---|---|---|---|
| NaiveForecaster | `strategy` | last | choices [last, mean, drift]; **drift if trending, last if persistent, mean if noisy** |
| | `sp` | 1 | **= dominant_period** from `profile_series`; range [1,1024] |
| TinyTimeMixer | `context_length` | — | **≥ 2× dominant_period** (cover the cycle); range [8,2048] |
| | `prediction_length` | — | **match the requested horizon** |
| SubLOF | `n_neighbors` | None | **~√(window_size)**, 10–50 typical; range [2,200] |
| | `window_size` | None | **~1× dominant_period** |
| TimeSeriesKMeans | `n_clusters` | 8 | **choose by silhouette**, start 2–8; range [2,50] |
| ConformalIntervals | `coverage` | 0.9 | 0.9 default; 0.8/0.95 alternates |

## The reasoning loop (per parameter)

```
profile_series(data_ref)  →  evidence {dominant_period, stationarity, n_channels, length, …}
get_component(model_id)    →  param_schema {param: {default, type, hint{depends_on, suggest, range}}}
        ↓  the AGENT reasons each value from evidence + hint
recipe.params = {context_length: 2*period, sp: period, strategy: "drift", n_neighbors: int(sqrt(w)), …}
        ↓
run_recipe / run_plan  →  validate_params(card, params)  →  param_audit recorded in the result
        ↓
scoring.py grades the choices (e.g. context_length ≥ lookback; sp ≈ period; strategy fits trend)
```

So parameter selection is the agent's reasoning, the schema+hints make the decision space
explicit, validation prevents nonsense (verified: it rejects `strategy="wizard"`, `sp=99999`,
and unknown params), and the audit/score makes "did the agent set the parameters well" a
graded, benchmarkable signal.

## Integration (no new top-level tool)
- `get_component(model_id)` now returns the card **+ its `param_schema`** (the agent reads
  hints there). `describe_candidates` can include a one-line `key_params` summary.
- `run_recipe`/`run_plan` call `validate_params` before resolve and write the `param_audit`
  into the result `summary` → state-exported for scoring.
- `scoring.py` extends to grade per-parameter choices vs the evidence-derived references.

## Why this matters
- It makes the server honest about complexity: **each model is a parameter-reasoning problem**,
  not a button. A naive agent that takes defaults (e.g. `sp=1` on a seasonal series, or a
  context shorter than the cycle) is measurably worse.
- It's uniform across hundreds of models because the schema is **introspected from sktime** and
  enriched with a small, shared hint library plus per-card overrides.

## Files
`param_space.py` (introspect / param_schema / validate_params + hint library),
`tests/test_param_space.py` (4 tests). Ties to `profile.py` (evidence), `scoring.py` (grading),
`STORES.md` (card `param_hints` field).
