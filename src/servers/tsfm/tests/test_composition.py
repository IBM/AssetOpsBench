"""Core capability: agentic discover → compose → run → diagnose → iterate (sktime substrate)."""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, pandas as pd
from tsfm.core.store import MemoryStore
from tsfm.engine import composition as C

NF = "sktime.forecasting.naive.NaiveForecaster"
_Y = pd.Series(np.sin(np.arange(160) / 4.0) + 0.03 * np.arange(160))


def test_discover_components():
    d = C.discover_components(MemoryStore())
    assert d["scitype"] == "forecaster"
    assert "weighted" in d["combiners"] and "stack" in d["combiners"]
    assert any("TinyTimeMixer" in m for m in d["foundation_models"])


def test_single_then_ensemble_with_per_member():
    s = MemoryStore()
    single = C.run_recipe(s, _Y, {"estimator": {"name": "drift", "sktime_class": NF,
                                                "params": {"strategy": "drift"}}, "fh": [1, 2, 3, 4]})
    assert single["backtest_score"] > 0
    ens = C.run_recipe(s, _Y, {"ensemble": {"combine": "mean", "members": [
        {"name": "last", "sktime_class": NF, "params": {"strategy": "last"}},
        {"name": "mean", "sktime_class": NF, "params": {"strategy": "mean"}},
        {"name": "drift", "sktime_class": NF, "params": {"strategy": "drift"}}]}, "fh": [1, 2, 3, 4]})
    assert set(ens["per_member_score"]) == {"last", "mean", "drift"}     # diagnostics for iterate


def test_iterate_improves_and_lineage():
    s = MemoryStore()
    ens = {"ensemble": {"combine": "mean", "members": [
        {"name": "last", "sktime_class": NF, "params": {"strategy": "last"}},
        {"name": "mean", "sktime_class": NF, "params": {"strategy": "mean"}},
        {"name": "drift", "sktime_class": NF, "params": {"strategy": "drift"}}]}, "fh": [1, 2, 3, 4]}
    r1 = C.run_recipe(s, _Y, ens)
    pm = {k: v for k, v in r1["per_member_score"].items() if isinstance(v, (int, float))}
    worst = max(pm, key=pm.get)
    keep = [m for m in ens["ensemble"]["members"] if m["name"] != worst]
    r2 = C.run_recipe(s, _Y, {"ensemble": {"combine": "mean", "members": keep}, "fh": [1, 2, 3, 4]},
                      parent_run_id=r1["run_id"])
    assert r2["backtest_score"] <= r1["backtest_score"]          # dropping the worst helps (or ties)
    # lineage persisted
    child = s.get(C.RUNS, r2["run_id"])
    assert child["parent_run_id"] == r1["run_id"]
