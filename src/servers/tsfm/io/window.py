"""Data access — read a sensor window for (asset, channels, range).

In production this reads the `iot` / `vibration` CouchDB collections (or a dataset_path CSV).
For the deterministic demo/tests it synthesizes a realistic multivariate series per asset
(trend + seasonality + noise, with an injectable anomaly), so the whole pipeline runs with no
external data.
"""

from __future__ import annotations

import numpy as np

# per-asset profile: (n_channels, period, anomaly_at_fraction or None)
_PROFILE = {
    "chiller_6":   (3, 48, 0.85),
    "metro_pump_1": (4, 24, None),
    "motor_01":    (2, 16, 0.7),
}


def read_window(asset_id: str, *, channels=None, length: int = 600, seed: int = 0,
                anomalous: bool = True, store=None):
    """Return (X, channel_names). If `store` + iot data exist, could read real data; here
    synthesizes a deterministic series so the pipeline is runnable end-to-end."""
    n_ch, period, anom = _PROFILE.get(asset_id, (3, 24, None))
    rng = np.random.RandomState(seed + abs(hash(asset_id)) % 1000)
    t = np.arange(length)
    X = np.zeros((length, n_ch))
    for c in range(n_ch):
        X[:, c] = (0.01 * (c + 1) * t                       # trend
                   + (2 + c) * np.sin(2 * np.pi * t / period)  # seasonality
                   + rng.normal(0, 0.4, length))            # noise
    if anomalous and anom is not None:
        i = int(anom * length)
        X[i:i + 6, 0] += 8.0                                 # injected spike on channel 0
    names = (channels or [f"ch{c}" for c in range(n_ch)])[:n_ch]
    if channels:
        idx = list(range(min(len(channels), n_ch)))
        X = X[:, idx]
    return X, names
