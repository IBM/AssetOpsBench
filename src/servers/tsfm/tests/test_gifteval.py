"""GIFT-Eval-native evaluation: seasonal-naive normalization, geo-mean, mean-rank leaderboard."""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from tsfm.core.store import MemoryStore
from tsfm.eval import gifteval as G

NF = "sktime.forecasting.naive.NaiveForecaster"
TR = "sktime.forecasting.trend.TrendForecaster"


def _cfg(name, n, h, slope, seed):
    t = np.arange(n)
    y = slope * t + np.random.RandomState(seed).normal(0, 0.5, n)
    return {"name": name, "y": y.tolist(), "fh": list(range(1, h + 1)), "sp": 1}


def test_seasonal_naive_normalizes_to_one():
    s = MemoryStore()
    cfg = [_cfg("A", 150, 8, 0.05, 0)]
    r = G.evaluate_recipe(s, {"estimator": {"sktime_class": NF, "params": {"strategy": "last"}}}, cfg)
    # the recipe == the per-config baseline ⇒ normalized MASE ≈ 1.0
    assert abs(r["per_config"][0]["norm_mase"] - 1.0) < 0.05


def test_leaderboard_ranks_better_recipe_first():
    s = MemoryStore()
    cfgs = [_cfg("A", 160, 8, 0.05, 1), _cfg("B", 150, 6, 0.08, 2)]
    recipes = {
        "naive_last": {"estimator": {"sktime_class": NF, "params": {"strategy": "last"}}},
        "drift": {"estimator": {"sktime_class": NF, "params": {"strategy": "drift"}}},
        "ensemble": {"ensemble": {"combine": "mean", "members": [
            {"name": "drift", "sktime_class": NF, "params": {"strategy": "drift"}},
            {"name": "trend", "sktime_class": TR, "params": {}}]}},
    }
    lb = G.leaderboard(s, recipes, cfgs, by="norm_mase")
    names = [row["recipe"] for row in lb["leaderboard"]]
    assert names[0] in ("ensemble", "drift")            # a real model beats seasonal-naive on trend
    assert names[-1] == "naive_last"
    # winner's geo-mean normalized MASE < 1 (beats the baseline)
    assert lb["leaderboard"][0]["agg"]["geomean_norm_mase"] < 1.0
