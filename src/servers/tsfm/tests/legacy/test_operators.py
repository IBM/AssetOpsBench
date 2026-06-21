"""TS Operator Algebra — typed, leakage-safe, invertible, zero-model-comparable."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pytest
from tsfm.legacy.operators import (Operator, Role, Cardinality, Pipeline, quality_check,
                                    applicable, ValidityError, zero_model, beats_zero_model)


# ---- concrete operators ----
def robust_scaler():
    def fit(X, m):
        c = np.median(X, 0); q1, q3 = np.percentile(X, 25, 0), np.percentile(X, 75, 0)
        return {"c": c, "s": np.where(q3 - q1 > 1e-8, q3 - q1, 1.0)}
    return Operator("robust_scaler", Role.scaler, Cardinality.series, invertible=True,
                    _fit=fit, _apply=lambda X, st: (X - st["c"]) / st["s"],
                    _inverse=lambda Y, st: Y * st["s"] + st["c"])


def differencer():
    def ap(X, st):
        out = np.copy(X); out[1:] = X[1:] - X[:-1]; return out      # out[0]=X0, rest diffs
    return Operator("difference", Role.stationarizer, Cardinality.series, invertible=True,
                    _fit=lambda X, m: {}, _apply=ap, _inverse=lambda Y, st: np.cumsum(Y, axis=0))


def log_transform():   # requires positive data — precondition gate
    return Operator("log", Role.scaler, Cardinality.series, invertible=True,
                    preconditions=["positive_only"], _fit=lambda X, m: {},
                    _apply=lambda X, st: np.log(X), _inverse=lambda Y, st: np.exp(Y))


def persistence_estimator():
    def fit(X, m): return {"last": X[-1], "h": int(m.get("horizon", 8))}
    return Operator("persistence", Role.estimator, Cardinality.prediction,
                    _fit=fit, _apply=lambda X, st: np.repeat(st["last"][None, :], st["h"], 0))


def score_estimator():   # anomaly: robust z-score magnitude per timestep
    def ap(X, st):
        z = np.abs((X - np.median(X, 0)) / (np.median(np.abs(X - np.median(X, 0)), 0) * 1.4826 + 1e-8))
        return z.max(1)
    return Operator("zscore", Role.estimator, Cardinality.score, _fit=lambda X, m: {}, _apply=ap)


def otsu_thresholder():
    def ap(score, st):
        s = np.asarray(score, float); thr = np.percentile(s, 95)
        return np.where(s > thr, -1, 1)            # +1 normal / -1 anomalous (AnomalyKiTS)
    return Operator("otsu", Role.thresholder, Cardinality.label, _apply=ap)


# ---- tests ----
def test_operator_invertibility_roundtrip():
    rng = np.random.RandomState(0); X = rng.normal(5, 2, (200, 3))
    for op in (robust_scaler(), differencer()):
        st = op.fit(X, {}); Xt = op.apply(X, st); back = op.inverse(Xt, st)
        assert np.allclose(back, X, atol=1e-6), f"{op.name} round-trip failed"


def test_leakage_isolation_fit_on_Dfit_only():
    D_fit = np.full((100, 1), 10.0); D_fit[::2] = 0.0      # median 0..10
    D_in = np.full((20, 1), 999.0)
    sc = robust_scaler(); st = sc.fit(D_fit, {})
    # state derived from D_fit, NOT D_in (no leakage)
    assert abs(st["c"][0] - np.median(D_fit)) < 1e-9
    assert abs(st["c"][0] - np.median(D_in)) > 1.0


def test_validity_invariants():
    X = np.linspace(1, 10, 50)[:, None]
    bad_nan = Operator("nan", Role.scaler, _fit=lambda X, m: {}, _apply=lambda X, st: X * np.nan)
    from tsfm.legacy.operators import _check_invariants
    with pytest.raises(ValidityError):
        _check_invariants(bad_nan, X, bad_nan.apply(X, {}))
    collapse = Operator("collapse", Role.scaler, _fit=lambda X, m: {}, _apply=lambda X, st: np.zeros_like(X))
    with pytest.raises(ValidityError):
        _check_invariants(collapse, X, collapse.apply(X, {}))
    # differencer turning a ramp into a constant is allowed (stationarizer exempt)
    d = differencer(); _check_invariants(d, X, d.apply(X, {}))


def test_quality_check_disables_log_on_negatives():
    X = np.array([[-1.0], [2.0], [3.0]])
    qc = quality_check(X)
    assert qc["positive_only"] is False and qc["n_negative"] == 1
    assert not applicable(log_transform(), qc)              # log gated off
    assert applicable(robust_scaler(), qc)                  # scaler fine


def test_pipeline_forecasting_leakage_safe_and_inverse():
    rng = np.random.RandomState(1)
    series = (0.02 * np.arange(300) + np.sin(np.arange(300))).reshape(-1, 1) + 50
    D_fit, D_in = series[:250], series[250:]
    pipe = Pipeline([robust_scaler()], persistence_estimator(), task_id="tsfm_forecasting")
    pipe.fit(D_fit, meta={"horizon": 12})
    fc = pipe.predict(D_in, meta={"horizon": 12})
    assert fc.shape == (12, 1)
    # forecast returned in ORIGINAL units (inverse applied): near the last observed level
    assert abs(float(fc[0, 0]) - float(D_in[-1, 0])) < 5.0


def test_pipeline_anomaly_thresholds_to_labels():
    X = np.random.RandomState(2).normal(0, 1, (200, 2)); X[100:103] += 9
    pipe = Pipeline([differencer()], score_estimator(), post=otsu_thresholder(),
                    task_id="tsfm_anomaly_detection")
    pipe.fit(X[:150]); labels = pipe.predict(X)
    assert set(np.unique(labels)) <= {-1, 1} and (labels == -1).any()


def test_zero_model_and_gate():
    train = np.arange(100).reshape(-1, 1).astype(float)
    name, fn = zero_model("tsfm_forecasting", train)
    assert name == "persistence" and fn(5).shape == (5, 1)
    assert beats_zero_model(0.10, 0.25)            # pipeline mae 0.10 beats zero-model 0.25
    assert not beats_zero_model(0.30, 0.25)
