# TSFM tools are complex — they require reasoning, not defaults

A core claim of the TSFM revision: calling a TSFM tool well is a **reasoning problem**, not a
parameter-free button. The agent must decide a web of **interacting** parameters, each with
alternatives and a real cost if wrong. `planner.py` makes this explicit and gradeable.

## The decision space (per tool)

**Forecasting** — 5+ coupled decisions:
| decision | depends on | risk if wrong |
|---|---|---|
| **lookback window** | data seasonality (spectral period) | too short misses the cycle; too long feeds stale regime |
| **context_length** | must cover the lookback | shorter than lookback truncates the input |
| **model** | context_length + domain + horizon | wrong context/domain underperforms |
| **forecast horizon** | the question ("next 48 h", "7-day") | mismatch to the decision window |
| **channels** | relevant_sensors | noise from irrelevant, signal loss from missing |
| **features** | task + data (FLOps) | leaky/weak features degrade or invalidate the result |

**Anomaly detection** — pipeline + thresholding are *data-dependent*:
| decision | depends on | risk if wrong |
|---|---|---|
| **detection window** | seasonal period | flags normal swings / smears spikes |
| **AD pipeline** (DeepAD/RelationshipAD/ReconstructAD/WindowAD) | #channels, correlation, (non)linearity | misses the anomaly *mode* (point vs contextual vs collective) |
| **thresholding** (static otsu / dynamic) | stationarity | static on drift floods alerts; dynamic on stable misses steady faults |
| **mode** (batch / train-test) | stationarity / data availability | wrong baseline |

The killer is the **interdependency**: lookback ← seasonality, context_length ≥ lookback,
model ← (context_length, domain, horizon). You cannot pick the model before reasoning about
the data's lookback. A naive agent that calls with defaults gets a model whose context can't
hold the cycle, or a static threshold on a drifting signal.

## How the server surfaces the reasoning

`plan_forecasting(...)` / `plan_anomaly(...)` derive every parameter and return, for each, a
**rationale + alternatives + risk_if_wrong + source**, plus a complexity score. The agent
inspects/accepts/overrides the plan, then calls the compute tool with the chosen params. So
the tool exposes *why*, not just *what*.

Verified, deterministically:
- **lookback → context → model is enforced**: chiller_6 gets lookback 256 (2× spectral period
  128) ⇒ a model with context_length ≥ 256 (`ttm_energy_512_96`, also domain-matched); horizon
  48 parsed from "next 48 hours".
- **pipeline is data-driven**: 3 correlated channels (chiller_6) ⇒ `RelationshipAD`; 2-channel
  motor_01 ⇒ `DeepAD` — same tool, different reasoned choice.
- **thresholding reasons about stationarity**: injected trend ⇒ non-stationary ⇒ `dynamic`
  thresholding + `train_test` mode.

## Why this matters for the benchmark

1. **It separates strong agents from naive ones.** A defaults-only agent fails: wrong context
   length, mismatched horizon, static threshold on drift. The benchmark rewards the agent that
   reasons (right lookback for the seasonality, context ≥ lookback, dynamic threshold on a
   trending signal).
2. **It is gradeable from state.** The chosen parameters land in the result record's `summary`
   (lookback, context_length, horizon, pipeline, thresholding) → `export_state()` scoring can
   check each decision against the scenario's expectation, not just the final number.
3. **It models real PdM practice.** Engineers don't accept defaults; they choose a window that
   matches the asset's cycle, a model that covers it, and a threshold that suits the regime.
   The planner encodes that judgement so the agent must demonstrate it.

## Scoring hooks (per scenario)

A scenario can assert on the *decisions*, e.g.:
- `lookback` within [1×, 3×] of the true seasonal period
- `context_length ≥ lookback`
- `forecast_horizon` == the horizon named in the utterance
- AD `pipeline` appropriate to channel count / correlation
- `thresholding == dynamic` when the series is non-stationary

These are exactly the fields the planner emits and the result tables store — so "did the agent
reason correctly about the lookback window" becomes a concrete, automatically-scored check.

## Files
- `planner.py` — `plan_forecasting` / `plan_anomaly` (the reasoning layer).
- `tests/test_planner.py` — verifies the interdependencies and data-driven choices.
- Surfaces as MCP tools `plan_forecasting` / `plan_anomaly` (add to `tools.py`); compute tools
  accept the plan's parameters.
