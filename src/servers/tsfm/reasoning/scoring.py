"""Reasoning scorer — grades the AGENT's parameter choices against defensible references.

Used by the benchmark, NOT exposed to the agent. The agent reasons from the evidence tools
(profile.py) and supplies its own lookback / context / horizon / channels / pipeline /
thresholding; this module checks whether those choices were sound, given the data. So the
score measures the agent's reasoning, not the server's.

`param_audit` is the lightweight, factual validation the compute tools record into the result
summary (e.g. context_covers_lookback). `score_*_choices` is the graded rubric.
"""

from __future__ import annotations

import re
from typing import Optional

from tsfm.reasoning import profile


def _parse_horizon(question, freq_per_day=24):
    if not question:
        return None
    m = re.search(r"(\d+)\s*(hour|hr|day|week|month)", question.lower())
    if not m:
        return 7 * freq_per_day if "week" in question.lower() else None
    n, u = int(m.group(1)), m.group(2)
    return n * {"hour": 1, "hr": 1, "day": freq_per_day, "week": 7 * freq_per_day,
                "month": 30 * freq_per_day}[u]


def _check(name, ok, detail):
    return {"check": name, "pass": bool(ok), "detail": detail}


# --------------------------------------------------------------------------- #
def param_audit(evidence: dict, params: dict) -> dict:
    """Factual flags the compute tool records (not a grade)."""
    period = evidence.get("dominant_period")
    lb = params.get("lookback")
    ctx = params.get("context_length")
    return {"context_covers_lookback": (ctx is not None and lb is not None and ctx >= lb),
            "lookback_to_period_ratio": round(lb / period, 2) if (lb and period) else None,
            "non_stationary": evidence.get("non_stationary")}


def score_forecasting_choices(store, asset_id, params: dict, *, question: Optional[str] = None) -> dict:
    ev = profile.profile_series(store, asset_id)
    period = ev["dominant_period"] or 1
    lb = params.get("lookback") or 0
    ctx = params.get("context_length") or 0
    h = params.get("forecast_horizon")
    want_h = _parse_horizon(question)
    checks = [
        _check("lookback_matches_seasonality", period <= lb <= 3 * period,
               f"lookback {lb} vs period {period} (want 1x-3x)"),
        _check("context_covers_lookback", ctx >= lb, f"context {ctx} >= lookback {lb}"),
        _check("horizon_matches_request", (want_h is None) or (h == want_h),
               f"horizon {h} vs requested {want_h}"),
        _check("features_chosen", bool(params.get("feature_ids")),
               f"features {params.get('feature_ids')}"),
        _check("model_chosen", bool(params.get("model_id")), f"model {params.get('model_id')}"),
    ]
    passed = sum(c["pass"] for c in checks)
    return {"task": "tsfm_forecasting", "asset_id": asset_id, "checks": checks,
            "score": round(passed / len(checks), 3), "evidence": ev}


def score_anomaly_choices(store, asset_id, params: dict) -> dict:
    ev = profile.profile_series(store, asset_id)
    period = ev["dominant_period"] or 1
    w = params.get("detection_window") or 0
    pipe = params.get("model_id") or ""
    thr = params.get("thresholding")
    multivariate_corr = ev["n_channels"] >= 3 and ev["max_abs_channel_corr"] >= 0.3
    checks = [
        _check("window_matches_seasonality", 0.5 * period <= w <= 2 * period,
               f"window {w} vs period {period} (want 0.5x-2x)"),
        _check("pipeline_fits_data",
               ("relationship" in pipe.lower()) if multivariate_corr else True,
               f"pipeline {pipe}; multivariate_corr={multivariate_corr}"),
        _check("thresholding_fits_stationarity",
               (thr == "dynamic") if ev["non_stationary"] else (thr == "static"),
               f"thresholding {thr}; non_stationary={ev['non_stationary']}"),
    ]
    passed = sum(c["pass"] for c in checks)
    return {"task": "tsfm_anomaly_detection", "asset_id": asset_id, "checks": checks,
            "score": round(passed / len(checks), 3), "evidence": ev}
