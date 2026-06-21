"""Evidence tools — give the AGENT the facts to reason from. No decisions here.

These tools answer "what does the data look like?" and "what can the catalog offer?" — the
raw signals an agent needs to choose lookback / context / horizon / channels / pipeline /
thresholding itself. Deliberately NO recommended values: the reasoning is the agent's job
(the server must not pre-decide, or the benchmark stops testing the agent).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from tsfm.io import window as io
from tsfm.stores import model_store
from tsfm.stores import feature_store


def profile_series(store, asset_id: str, channels: Optional[List[str]] = None) -> dict:
    """Factual characterization of the asset's signal — evidence, not advice."""
    X, names = io.read_window(asset_id, store=store)
    X = np.asarray(X, float)
    n, c = X.shape

    # seasonality (dominant spectral period) per channel — DETREND first so a strong
    # linear trend doesn't masquerade as a giant low-frequency "period".
    def _detrend(x):
        t = np.arange(len(x))
        a, b = np.polyfit(t, x, 1)
        return x - (a * t + b)

    periods = []
    for j in range(c):
        x = _detrend(X[:, j])
        sp = np.abs(np.fft.rfft(x))
        if len(sp) > 2 and sp[1:].max() > 1e-9:
            periods.append(int(round(n / (1 + int(np.argmax(sp[1:]))))))
    dominant_period = int(np.median(periods)) if periods else None
    seasonality_strength = None
    if dominant_period:
        sp = np.abs(np.fft.rfft(_detrend(X[:, 0])))
        seasonality_strength = round(float(sp[1:].max() / (sp[1:].sum() + 1e-9)), 3)

    # stationarity / trend on channel 0
    t = np.arange(n) - n / 2
    slope = float((t * (X[:, 0] - X[:, 0].mean())).sum() / ((t ** 2).sum() or 1.0))
    trend_strength = round(abs(slope) * n / (np.std(X[:, 0]) + 1e-9), 3)

    # inter-channel correlation summary
    corr = np.corrcoef(X.T) if c > 1 else np.array([[1.0]])
    off = corr[np.triu_indices(c, 1)] if c > 1 else np.array([])
    max_abs_corr = round(float(np.max(np.abs(off))), 3) if off.size else 0.0

    # missingness / gaps (synthetic data has none, but report the check)
    gaps = int(np.isnan(X).sum())

    return {"asset_id": asset_id, "n_observations": n, "n_channels": c,
            "channels": names, "dominant_period": dominant_period,
            "seasonality_strength": seasonality_strength,
            "trend_slope": round(slope, 5), "trend_strength": trend_strength,
            "non_stationary": trend_strength > 1.0,
            "max_abs_channel_corr": max_abs_corr, "n_missing": gaps,
            "value_range": [round(float(X.min()), 3), round(float(X.max()), 3)]}


def available_contexts(store, task_id: str = "tsfm_forecasting") -> dict:
    """What the model store offers for this task — so the agent can match context to lookback."""
    ms = model_store.list_models(store, task_id=task_id)
    return {"task_id": task_id, "models": [
        {"model_id": m["model_id"], "context_length": m.get("context_length"),
         "prediction_length": m.get("prediction_length"), "domain": m.get("domain"),
         "framework": m.get("framework"), "pipeline_type": m.get("pipeline_type"),
         "usage_modes": m.get("usage_modes")} for m in ms]}


def available_features(store, category: Optional[str] = None) -> dict:
    """Transforms + extractors the agent can choose to apply."""
    return {"transforms": [{"feature_id": f["feature_id"], "scenario_categories": f.get("scenario_categories"),
                            "invertible": f.get("invertible")} for f in feature_store.find_features(store, category=category)],
            "extractors": [e["extractor_name"] for e in feature_store.list_extractors(store, category=category)]}
