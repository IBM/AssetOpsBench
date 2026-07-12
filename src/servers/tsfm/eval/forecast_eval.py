"""forecast_eval.py: multi-config forecasting evaluation for the composition loop.

A generic, seasonal-naive-normalized forecasting scorer. Its scoring conventions follow the
Salesforce GIFT-Eval protocol (arXiv:2410.10393); it uses none of the GIFT-Eval datasets or
package, only the methodology:
  * many CONFIGS = (dataset x frequency x horizon x {uni|multi}), not one test;
  * point metric MASE + probabilistic metric CRPS;
  * each config NORMALIZED by a seasonal-naive baseline (our Zero Model);
  * aggregate across configs by the geometric mean of the normalized scores;
  * a leaderboard that ranks recipes per config and reports the mean rank
    (so no single config dominates).

So the agent's mix-and-match / ensemble search (composition.py) is judged GIFT-Eval style:
better aggregate-normalized CRPS/MASE and better mean rank; robust, scale-free, multi-config.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
def _mase(y_true, y_pred, y_train, sp: int) -> float:
    y_true, y_pred, y_train = map(
        lambda a: np.asarray(a, float).ravel(), (y_true, y_pred, y_train)
    )
    denom = (
        np.mean(np.abs(y_train[sp:] - y_train[:-sp]))
        if len(y_train) > sp
        else np.mean(np.abs(np.diff(y_train))) or 1e-8
    )
    return float(np.mean(np.abs(y_true - y_pred)) / (denom + 1e-12))


def _crps_from_quantiles(y_true, qdf: pd.DataFrame, levels: List[float]) -> float:
    """Empirical CRPS via the quantile (pinball) decomposition: CRPS ~= 2 * mean_k pinball_k."""
    y = np.asarray(y_true, float).ravel()
    tot = 0.0
    for a in levels:
        col = [c for c in qdf.columns if abs(c[-1] - a) < 1e-6]
        if not col:
            continue
        q = np.asarray(qdf[col[0]], float).ravel()[: len(y)]
        diff = y - q
        tot += np.mean(np.where(diff >= 0, a * diff, (a - 1) * diff))
    return float(2.0 * tot / max(len(levels), 1))


def _split(y: pd.Series, h: int):
    return y.iloc[:-h], y.iloc[-h:]


# --------------------------------------------------------------------------- #
def evaluate_config(
    build_forecaster: Callable,
    recipe: dict,
    y: pd.Series,
    *,
    fh: List[int],
    sp: int,
    quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict:
    """Fit on train, score on the held-out horizon: raw MASE (+ CRPS if probabilistic)."""
    h = len(fh)
    ytr, yte = _split(y, h)
    fc = build_forecaster(recipe)
    fc.fit(ytr, fh=fh)
    pred = np.asarray(fc.predict(), float).ravel()[:h]
    mase = _mase(yte, pred, ytr, sp)
    crps = None
    if recipe.get("conformal"):
        try:
            q = fc.predict_quantiles(alpha=list(quantile_levels))
            crps = _crps_from_quantiles(yte, q, list(quantile_levels))
        except Exception:
            crps = None
    return {
        "mase": round(mase, 4),
        "crps": (round(crps, 4) if crps is not None else None),
    }


def seasonal_naive_scores(
    y: pd.Series, *, fh: List[int], sp: int, quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9)
) -> dict:
    """The GIFT-Eval normalizer = our Zero Model (seasonal naive)."""
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.conformal import ConformalIntervals

    h = len(fh)
    ytr, yte = _split(y, h)
    base = NaiveForecaster(strategy="last", sp=sp if sp > 1 else 1)
    base.fit(ytr, fh=fh)
    pred = np.asarray(base.predict(), float).ravel()[:h]
    mase = _mase(yte, pred, ytr, sp)
    try:
        cf = ConformalIntervals(
            NaiveForecaster(strategy="last", sp=sp if sp > 1 else 1)
        ).fit(ytr, fh=fh)
        crps = _crps_from_quantiles(
            yte,
            cf.predict_quantiles(alpha=list(quantile_levels)),
            list(quantile_levels),
        )
    except Exception:
        crps = None
    return {
        "mase": round(mase, 4),
        "crps": (round(crps, 4) if crps is not None else None),
    }


def _geomean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None and v > 0]
    return round(float(np.exp(np.mean(np.log(vals)))), 4) if vals else None


def evaluate_recipe(
    store, recipe: dict, configs: List[dict], *, baselines: Optional[List[dict]] = None
) -> dict:
    """Score a recipe across configs: per-config normalized MASE/CRPS + geo-mean aggregate.

    config = {"name", "y", "fh", "sp"}. Normalized score = recipe_metric / seasonal_naive_metric
    (so <1 means 'beats seasonal naive', exactly the relative reporting GIFT-Eval uses).
    `baselines` optionally supplies the precomputed seasonal-naive scores per config (they are
    recipe-independent), so a leaderboard sweep computes each baseline once instead of per recipe.
    """
    from ..engine import composition as C

    bf = lambda r: C.build_forecaster(r, store)
    rows, n_mase, n_crps = [], [], []
    for i, cfg in enumerate(configs):
        y = pd.Series(np.asarray(cfg["y"], float))
        sn = (
            baselines[i]
            if baselines is not None
            else seasonal_naive_scores(y, fh=cfg["fh"], sp=cfg["sp"])
        )
        sc = evaluate_config(bf, recipe, y, fh=cfg["fh"], sp=cfg["sp"])
        nm = round(sc["mase"] / sn["mase"], 4) if sn["mase"] else None
        nc = round(sc["crps"] / sn["crps"], 4) if (sc["crps"] and sn["crps"]) else None
        rows.append(
            {
                "config": cfg["name"],
                "mase": sc["mase"],
                "crps": sc["crps"],
                "norm_mase": nm,
                "norm_crps": nc,
            }
        )
        if nm:
            n_mase.append(nm)
        if nc:
            n_crps.append(nc)
    return {
        "per_config": rows,
        "agg": {
            "geomean_norm_mase": _geomean(n_mase),
            "geomean_norm_crps": _geomean(n_crps),
            "n_configs": len(configs),
        },
    }


def leaderboard(
    store, recipes: Dict[str, dict], configs: List[dict], by: str = "norm_crps"
) -> dict:
    """Rank recipes per config (lower=better) and report the MEAN RANK (GIFT-Eval aggregation)."""
    # the seasonal-naive baseline is recipe-independent: compute it once per config and reuse it
    # across every recipe (avoids an O(recipes x configs) recompute).
    baselines = [
        seasonal_naive_scores(
            pd.Series(np.asarray(cfg["y"], float)), fh=cfg["fh"], sp=cfg["sp"]
        )
        for cfg in configs
    ]
    evals = {
        name: evaluate_recipe(store, r, configs, baselines=baselines)
        for name, r in recipes.items()
    }
    fallback = "norm_mase" if by == "norm_crps" else "norm_crps"
    ranks = {name: [] for name in recipes}
    for i, cfg in enumerate(configs):
        scored = []
        for name in recipes:
            row = evals[name]["per_config"][i]
            v = row.get(by) if row.get(by) is not None else row.get(fallback)
            scored.append((name, v if v is not None else float("inf")))
        scored.sort(key=lambda t: t[1])
        for rank, (name, _) in enumerate(scored, 1):
            ranks[name].append(rank)
    board = sorted(
        (
            (name, round(float(np.mean(rk)), 3), evals[name]["agg"])
            for name, rk in ranks.items()
        ),
        key=lambda t: t[1],
    )
    return {
        "ranked_by": by,
        "leaderboard": [
            {"recipe": n, "mean_rank": mr, "agg": agg} for n, mr, agg in board
        ],
    }