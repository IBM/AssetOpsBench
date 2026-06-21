# The agent does the reasoning — the server gives evidence and scores it

Correction to the "tools are complex" design: **the server must not pre-decide** the lookback /
context / horizon / pipeline / thresholding. If it does, the agent rubber-stamps a finished
plan and the benchmark stops measuring the agent. The right split:

```
  EVIDENCE (server)            REASONING (agent)              VALIDATE + SCORE (server/benchmark)
  profile_series  ─────────▶   choose lookback, context,  ─▶  param_audit (factual flags in result)
  available_contexts          horizon, channels, features,    score_*_choices (graded rubric vs
  available_features          AD pipeline, thresholding       a hidden reference) — not seen by agent
```

## 1. Server gives facts, not decisions (`profile.py`)

- `profile_series(asset)` → n_obs, n_channels, **dominant_period** (trend-detrended so a strong
  trend doesn't masquerade as the period), seasonality_strength, trend/stationarity,
  inter-channel correlation, missingness, value range. **No `recommended_lookback`.**
- `available_contexts(task)` → each candidate model's context_length / domain / pipeline — so
  the agent can match context to the lookback it chose.
- `available_features(category)` → transforms + extractors it may apply.

These are the raw signals an engineer would look at. The agent combines them with the question
to decide every parameter.

## 2. Agent decides — tools require explicit params (no silent defaults)

`run_tsfm_forecasting` / `run_tsad` take the agent's chosen `lookback`, `model_id` (⇒
context_length), `forecast_horizon`, `channels`, `feature_ids`, `thresholding`, `mode`. The
compute tool records a factual `param_audit` into the result `summary`
(e.g. `context_covers_lookback`, `lookback_to_period_ratio`) — evidence of *what the agent
chose*, captured in state.

## 3. Benchmark scores the agent's reasoning (`scoring.py`, hidden)

`score_forecasting_choices` / `score_anomaly_choices` grade the agent's parameters against
defensible references derived from the evidence — NOT shown to the agent:
- lookback within 1×–3× the seasonal period
- context_length ≥ lookback
- forecast_horizon == the horizon named in the utterance
- AD pipeline appropriate to channel count / correlation
- thresholding == dynamic iff the series is non-stationary

Verified: reasoned choices score **1.0**, naive defaults **0.4** (forecasting); anomaly
**0.67 vs 0.0**. A defaults-only agent is measurably worse — which is the whole point.

## 4. `planner.py` is demoted

It stays as an **optional advisor / ablation baseline** ("agent-with-advisor" vs
"agent-reasons-alone") and as the reference logic reused by `scoring.py`. It is **not** the
default path and is not exposed as a primary tool in the scored configuration.

## 5. Net tool surface for reasoning

| Tool | role | exposed to agent? |
|---|---|---|
| `profile_series`, `available_contexts`, `available_features` | evidence | **yes** |
| `run_tsfm_forecasting`, `run_tsad` (explicit params) | act on the agent's decisions | **yes** |
| `param_audit` (inside compute) | factual flags into result | yes (as output) |
| `score_*_choices` | grade reasoning | **no** (benchmark only) |
| `plan_*` | optional advisor / ablation | only in the "advisor" condition |

This makes the benchmark a genuine test of whether the agent can reason about lookback,
context, horizon, channels, model, and thresholding — with the server staying an honest broker
of facts and an objective grader.

## Files
`profile.py` (evidence), `scoring.py` (graded rubric + param_audit), `planner.py` (optional
advisor/oracle), `tests/test_reasoning.py` (evidence-not-decisions; reasoned > naive).
