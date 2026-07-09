"""End-to-end test: resolve the pyod_iforest catalog card and flag anomalies.

Placement: src/servers/tsfm/tests/test_pyod_iforest.py
Gated behind `pytest.importorskip("pyod")` so it skips cleanly when the optional
`pyod` dependency is not installed.
"""

import json
from pathlib import Path

import pytest

from servers.tsfm.substrate import resolver


def _load_catalog():
    # tests/ -> tsfm/ -> servers/ -> src/ -> repo root
    root = Path(__file__).resolve().parents[4]
    catalog = root / "src/couchdb/scenarios_data/shared/tsfm/model_catalog.json"
    if not catalog.exists():
        pytest.skip(f"catalog not found at {catalog}")
    return json.loads(catalog.read_text())


def test_pyod_iforest_resolves_and_flags_anomalies():
    pytest.importorskip("pyod")
    import numpy as np
    import pandas as pd

    card = next((c for c in _load_catalog() if c.get("model_id") == "pyod_iforest"), None)
    assert card is not None, "pyod_iforest card missing from model_catalog.json"

    # nested `_target_` estimator spec is built into a real PyOD IsolationForest here
    det = resolver.resolve(card)

    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=200))
    y.iloc[50] = 25.0     # obvious outliers
    y.iloc[150] = -25.0

    det.fit(y)
    pred = det.predict(y)

    # sktime detectors return either dense 0/1(-1) labels or anomaly index positions
    arr = np.asarray(getattr(pred, "values", pred)).ravel()
    if arr.size == len(y) and set(np.unique(arr).tolist()).issubset({0, 1, -1}):
        flagged = set(np.where(arr != 0)[0].tolist())
    else:
        flagged = {int(i) for i in arr if 0 <= i < len(y)}

    assert len(flagged) > 0, "expected pyod_iforest to flag at least one anomaly"