"""Exercise the standard image's scientific capabilities using artificial data.

No benchmark inputs, answers, catalog entries or model checkpoints are needed.
Run during both Docker builds and independently with networking disabled.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import tempfile


def main() -> None:
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pywt
    import seaborn as sns
    from scipy import signal, stats
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from threadpoolctl import threadpool_limits

    # Fixed artificial observations; these are not task or evaluator examples.
    t = np.arange(48, dtype=float)
    frame = pd.DataFrame({"t": t, "wave": np.sin(t / 4)})
    X = frame.to_numpy()
    y = 1_000_000.0 + 100 * t + frame["wave"].to_numpy()
    groups = np.repeat(np.arange(6), 8)
    with threadpool_limits(limits=1):
        for train, valid in GroupKFold(3).split(X, y, groups):
            assert set(groups[train]).isdisjoint(groups[valid])
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X[train], y[train])
            assert np.isfinite(model.predict(X[valid])).all()
        for estimator in [GradientBoostingRegressor(n_estimators=4, random_state=7),
                          KernelRidge(alpha=1.0, kernel="laplacian")]:
            estimator.fit(X, y)
            assert np.isfinite(estimator.predict(X[:3])).all()
        classifier = make_pipeline(StandardScaler(), LogisticRegression())
        classifier.fit(X, (t >= 24).astype(int))
        assert classifier.predict_proba(X[:3]).shape == (3, 2)
        forecast = ExponentialSmoothing(y, trend="add").fit().forecast(4)
        assert np.isfinite(forecast).all()

    frequencies, power = signal.welch(frame["wave"].to_numpy(), nperseg=16)
    assert len(frequencies) == len(power) and np.isfinite(power).all()
    assert np.isfinite(stats.skew(y))
    coefficients = pywt.wavedec(y, "haar", level=2)
    assert np.allclose(pywt.waverec(coefficients, "haar")[:len(y)], y)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        joblib.dump(model, root / "model.joblib")
        restored = joblib.load(root / "model.joblib")
        assert np.allclose(restored.predict(X[:3]), model.predict(X[:3]))
        fig, ax = plt.subplots()
        sns.lineplot(data=frame, x="t", y="wave", ax=ax)
        fig.savefig(root / "plot.png")
        plt.close(fig)
        assert (root / "plot.png").stat().st_size > 0
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
                "matplotlib", "seaborn", "joblib", "threadpoolctl", "PyWavelets"]
    print(json.dumps({"status": "passed", "packages": {
        name: importlib.metadata.version(name) for name in packages
    }}, sort_keys=True))


if __name__ == "__main__":
    main()
