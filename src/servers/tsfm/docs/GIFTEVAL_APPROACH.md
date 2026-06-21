# Adopting the GIFT-Eval approach as the server's evaluation backbone

Salesforce **GIFT-Eval** (arXiv:2410.10393; HF Space `Salesforce/GIFT-Eval`) is the de-facto
general-TS-forecasting benchmark. We adopt its *protocol* as how the TSFM MCP server scores —
so the agent's mix-and-match / ensemble search is judged the way the field judges foundation
models. Built and verified (`gifteval.py`, tests green).

## What GIFT-Eval does (and we mirror)

| GIFT-Eval | In our server |
|---|---|
| **Many configs** = dataset × frequency × horizon × {uni\|multi} (≈97) | a recipe is evaluated over a **list of configs**, not one split |
| Point metric **MASE** + probabilistic **CRPS** | `evaluate_config` returns MASE; CRPS from `predict_quantiles` when the recipe is conformal/probabilistic |
| **Normalize each task by a seasonal-naïve baseline** | normalized = recipe_metric ÷ seasonal-naïve metric — and seasonal-naïve **is our Zero Model**, so this is free and consistent |
| **Aggregate by geometric mean** | `geomean_norm_mase`, `geomean_norm_crps` across configs |
| **Mean rank across configs** (no dataset dominates) | `leaderboard()` ranks recipes per config, reports the **mean rank** |

## Verified behaviour

On trending multi-config synthetic data:
- `naive_last` (the baseline) → **normalized MASE ≈ 1.0** (sanity: a recipe equal to the
  per-config baseline normalizes to 1).
- leaderboard by normalized MASE: **ensemble (mean_rank 1.0, geomean 0.46) > drift (2.0, 0.76)
  > naive_last (3.0, 1.0)** — a real model beats seasonal-naïve and ranks first.
- conformal recipe → both MASE and **CRPS** (normalized CRPS 0.85 < 1, beats the baseline's
  CRPS). So the probabilistic axis works end-to-end.

## Why this is the right backbone

1. **It scores the agent the way the field scores models.** The composition loop's objective
   becomes the GIFT-Eval aggregate (geo-mean of seasonal-naïve-normalized CRPS/MASE) and the
   mean rank — robust, scale-free, multi-config. A good agent's ensemble must *win the
   leaderboard*, not one dataset.
2. **Seasonal-naïve normalizer = our Zero Model.** GIFT-Eval's baseline and AutoAI-TS's Zero
   Model are the same object, so normalization and the "beats-baseline" gate are one mechanism.
3. **CRPS ↔ conformal.** GIFT-Eval's probabilistic axis is exactly what `{"conformal":{…}}`
   provides — the recipe that adds calibrated intervals is also the one that gets a CRPS score.
4. **It powers selection too.** T-Daub's per-pipeline score and `find_models` ranking can use
   the GIFT-Eval normalized metric, so discovery, selection, and the leaderboard share one ruler.

## How it closes the agent loop (GIFT-Eval-driven ensemble search)

```
discover_components → agent composes recipes (single / ensemble / +conformal)
        → gifteval.leaderboard(recipes, configs)   # per-config MASE+CRPS, seasonal-naive-normalized,
                                                    # geo-mean + mean-rank
        → agent reads the board + per-config rows → drops/ reweights/ swaps members → re-evaluate
        → best recipe by mean rank is shipped; the whole search trajectory is state (tsfm_runs)
```
This is GIFT-Eval-style ensemble forecasting, but **agent-driven and benchmark-scored**, on the
sktime substrate.

## Notes / fidelity
- **Datasets**: the real GIFT-Eval ships 23–28 datasets / 7 domains / 10 frequencies; our
  `configs` list is the same shape — point it at the GIFT-Eval HF datasets
  (`Salesforce/GiftEval`) for the official suite, or at AssetOpsBench `iot`/`vibration` configs
  for the PdM setting. Same evaluator either way.
- **CRPS**: computed as the empirical quantile (pinball) decomposition (CRPS ≈ 2·mean-pinball
  over the quantile grid) — matches GIFT-Eval's probabilistic scoring; swap in sktime's `CRPS`
  metric object for distributional forecasters.
- **Term buckets**: add short/medium/long horizon grouping by tagging each config; the
  aggregator already keys on configs.

## Files
`gifteval.py` (evaluate_config / seasonal_naive_scores / evaluate_recipe / leaderboard),
`tests/test_gifteval.py`; composes with `composition.py` (recipes) + `selector.py` (T-Daub) +
`sktime_resolver.py` (substrate).

## Source
GIFT-Eval — https://huggingface.co/spaces/Salesforce/GIFT-Eval ·
paper https://arxiv.org/abs/2410.10393 · datasets https://huggingface.co/datasets/Salesforce/GiftEval
