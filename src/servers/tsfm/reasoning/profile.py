"""Evidence tools: give the AGENT the facts to reason from. No decisions here.

These tools answer "what does the data look like?" and "what can the catalog offer?", the
raw signals an agent needs to choose lookback / context / horizon / channels / pipeline /
thresholding itself. Deliberately NO recommended values: the reasoning is the agent's job
(the server must not pre-decide, or the benchmark stops testing the agent).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..stores import model_store


def profile_ref(
    data_ref: str,
    *,
    timestamp_column: Optional[str] = None,
    channels: Optional[List[str]] = None,
) -> dict:
    """Profile a time series passed as a FILE POINTER (the IoT data model). Loads the ref into
    an sktime container and returns structured evidence for the agent to reason from."""
    from ..io import refs

    obj = refs.load_series(data_ref, time_col=timestamp_column, channels=channels)
    import pandas as pd

    if isinstance(obj, pd.Series):
        X, names = obj.to_numpy().reshape(-1, 1), [obj.name or "value"]
    else:
        X, names = obj.to_numpy(), list(obj.columns)
    return _profile_array(np.asarray(X, float), names, ident=data_ref)


def _profile_array(X: np.ndarray, names: List[str], *, ident: str) -> dict:
    """Core profiling on a loaded (n, c) array behind the file-pointer path."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, c = X.shape

    # seasonality (dominant spectral period) per channel: DETREND first so a strong
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
    slope = float((t * (X[:, 0] - X[:, 0].mean())).sum() / ((t**2).sum() or 1.0))
    trend_strength = round(float(abs(slope) * n / (np.std(X[:, 0]) + 1e-9)), 3)

    # inter-channel correlation summary
    corr = np.corrcoef(X.T) if c > 1 else np.array([[1.0]])
    off = corr[np.triu_indices(c, 1)] if c > 1 else np.array([])
    max_abs_corr = round(float(np.max(np.abs(off))), 3) if off.size else 0.0

    # missingness / gaps (synthetic data has none, but report the check)
    gaps = int(np.isnan(X).sum())

    return {
        "source": ident,
        "n_observations": n,
        "n_channels": c,
        "channels": names,
        "dominant_period": dominant_period,
        "seasonality_strength": seasonality_strength,
        "trend_slope": round(slope, 5),
        "trend_strength": trend_strength,
        "non_stationary": bool(trend_strength > 1.0),
        "max_abs_channel_corr": max_abs_corr,
        "n_missing": gaps,
        "value_range": [round(float(X.min()), 3), round(float(X.max()), 3)],
    }


def available_contexts(store, task_id: str = "tsfm_forecasting") -> dict:
    """What the model store offers for this task, so the agent can match context to lookback."""
    ms = model_store.list_models(store, task_id=task_id)
    return {
        "task_id": task_id,
        "models": [
            {
                "model_id": m["model_id"],
                "context_length": m.get("context_length"),
                "prediction_length": m.get("prediction_length"),
                "domain": m.get("domain"),
                "framework": m.get("framework"),
                "pipeline_type": m.get("pipeline_type"),
                "usage_modes": m.get("usage_modes"),
            }
            for m in ms
        ],
    }