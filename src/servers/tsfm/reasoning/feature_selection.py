"""FLOps-style feature learning — dynamic, dataset-specific feature selection.

From Patel et al., "FLOps: On Learning Important Time Series Features for Real-Valued
Prediction" (IEEE BigData'20). FLOps takes a library of feature extractors, SCORES each on
the given input data, RANKS them against a Reference Feature, and FILTERS with a
Critical-Difference threshold — yielding a feature set tailored to *this* dataset+task.

This is the **selection / learning** layer of the feature store, complementing:
  - AnomalyKiTS Operators  = the extractors/transforms themselves,
  - EFE                    = generating/evolving new transforms,
  - FLOps (here)           = scoring + ranking + filtering which ones to use.

Faithful FLOps-lite (numpy only): tabulate with a look-back window, score each extractor by
|corr| with the target (a stand-in for the paper's multi-config performance score), rank,
then keep extractors that beat the Reference Feature by a Critical-Difference margin.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np

# ---- a small slice of the FLOps 130+ extractor library (scalar extractors) ----
EXTRACTORS: Dict[str, Callable[[np.ndarray], float]] = {
    # Data Profiling (value-based, order-independent)
    "mean": lambda w: float(np.mean(w)),
    "std": lambda w: float(np.std(w)),
    "min": lambda w: float(np.min(w)),
    "max": lambda w: float(np.max(w)),
    "range": lambda w: float(np.max(w) - np.min(w)),
    "q25": lambda w: float(np.percentile(w, 25)),
    "q75": lambda w: float(np.percentile(w, 75)),
    "kurtosis": lambda w: float(_kurtosis(w)),
    "skew": lambda w: float(_skew(w)),
    # Temporal / order-dependent
    "slope": lambda w: float(_slope(w)),
    "autocorr1": lambda w: float(_autocorr(w, 1)),
    "energy": lambda w: float(np.sum(np.asarray(w) ** 2) / len(w)),
    "abs_diff_mean": lambda w: float(np.mean(np.abs(np.diff(w)))) if len(w) > 1 else 0.0,
    # Frequency
    "spectral_centroid": lambda w: float(_spectral_centroid(w)),
    "dominant_freq_power": lambda w: float(_dominant_freq_power(w)),
}


def _kurtosis(w):
    w = np.asarray(w, float); s = w.std()
    return 0.0 if s < 1e-9 else float(np.mean(((w - w.mean()) / s) ** 4) - 3)


def _skew(w):
    w = np.asarray(w, float); s = w.std()
    return 0.0 if s < 1e-9 else float(np.mean(((w - w.mean()) / s) ** 3))


def _slope(w):
    w = np.asarray(w, float); t = np.arange(len(w)) - (len(w) - 1) / 2.0
    d = (t ** 2).sum() or 1.0
    return float((t * w).sum() / d)


def _autocorr(w, lag):
    w = np.asarray(w, float) - np.mean(w); v = (w ** 2).sum()
    return 0.0 if v < 1e-9 or len(w) <= lag else float((w[:-lag] * w[lag:]).sum() / v)


def _spectral_centroid(w):
    sp = np.abs(np.fft.rfft(np.asarray(w, float) - np.mean(w))); f = np.arange(len(sp))
    return 0.0 if sp.sum() < 1e-9 else float((f * sp).sum() / sp.sum())


def _dominant_freq_power(w):
    sp = np.abs(np.fft.rfft(np.asarray(w, float) - np.mean(w)))
    return 0.0 if len(sp) <= 1 else float(sp[1:].max())


# --------------------------------------------------------------------------- #
def discover_lookback(series: np.ndarray, max_lw: int = 128) -> int:
    """Dataset-specific look-back window via dominant spectral period (FLOps: lw from
    spectral/frequency analysis). Falls back to a sane default."""
    x = np.asarray(series, float).ravel(); x = x - x.mean()
    if len(x) < 8:
        return min(8, len(x))
    sp = np.abs(np.fft.rfft(x))
    if len(sp) <= 2 or sp[1:].max() < 1e-9:
        return min(32, len(x) // 2 or 8)
    k = 1 + int(np.argmax(sp[1:]))           # dominant non-DC bin
    period = int(round(len(x) / k))
    return int(max(8, min(max_lw, period)))


def _tabulate(series: np.ndarray, lw: int):
    """Slide a window; X = windows[:-1], y = next value (forecasting target)."""
    x = np.asarray(series, float).ravel()
    wins = np.stack([x[i:i + lw] for i in range(len(x) - lw)])   # (N, lw)
    y = x[lw:]                                                   # next value
    return wins, y


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


# --------------------------------------------------------------------------- #
# Multi-config scorers (FLOps: score under several criteria, aggregate by mean rank).
# Each returns a per-feature score vector aligned to `names`; higher = more relevant.
# --------------------------------------------------------------------------- #
def _norm(v):
    v = np.asarray(v, float); v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    m = v.max()
    return (v / m) if m > 1e-12 else v


def _score_corr(F, y, names):
    return _norm([_corr(F[:, j], y) for j in range(F.shape[1])])


def _score_ftest(F, y, names):
    try:
        from sklearn.feature_selection import f_regression
        f, _ = f_regression(F, y)
        return _norm(np.nan_to_num(f))
    except Exception:
        return _score_corr(F, y, names)


def _score_mutual_info(F, y, names):
    try:
        from sklearn.feature_selection import mutual_info_regression
        return _norm(mutual_info_regression(F, y, random_state=0))
    except Exception:
        return _score_corr(F, y, names)


def _score_model(F, y, names):
    """Multivariate model importance — captures interactions the univariate scorers miss."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=80, random_state=0, n_jobs=1)
        rf.fit(F, y)
        return _norm(rf.feature_importances_)
    except Exception:
        return _score_corr(F, y, names)


_SCORERS = {"corr": _score_corr, "f_test": _score_ftest,
            "mutual_info": _score_mutual_info, "model": _score_model}


def _feature_matrix(wins, ex):
    names = list(ex)
    F = np.column_stack([[ex[n](w) for w in wins] for n in names]).astype(float)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, names


def select_features(series: np.ndarray, *, reference_feature: str = "mean",
                    lookback: Optional[int] = None, cd_margin: float = 0.05,
                    extractors: Optional[Dict[str, Callable]] = None,
                    scorers: Optional[List[str]] = None) -> dict:
    """FLOps selection (multi-config): score each extractor on this series under several
    criteria (|corr|, F-test, mutual-info, model-importance), aggregate by MEAN RANK, rank,
    and keep those that beat the Reference Feature by `cd_margin` (Critical-Difference proxy).

    Aggregating across heterogeneous scorers is the FLOps robustness idea: a feature that ranks
    well under correlation, an F-test, mutual information AND a fitted model is trustworthy;
    one that only spikes under a single criterion is not. `scorers` defaults to all four;
    pass `["corr"]` for the fast univariate path.

    Returns {lookback, reference, scorers, scores{name:agg}, per_scorer{scorer:{name:score}},
             ranking[(name,agg)], selected[names], cd_margin}.
    """
    ex = extractors or EXTRACTORS
    lw = lookback or discover_lookback(series)
    wins, y = _tabulate(series, lw)
    F, names = _feature_matrix(wins, ex)
    use = scorers or ["corr", "f_test", "mutual_info", "model"]
    use = [s for s in use if s in _SCORERS] or ["corr"]

    # per-scorer normalized scores, then mean-rank aggregation across scorers
    per_scorer = {s: dict(zip(names, _SCORERS[s](F, y, names))) for s in use}
    ranks = np.zeros(len(names))
    for s in use:
        sv = np.array([per_scorer[s][n] for n in names])
        order = (-sv).argsort()                       # best→worst
        rk = np.empty(len(names)); rk[order] = np.arange(1, len(names) + 1)
        ranks += rk
    mean_rank = ranks / len(use)
    agg = 1.0 - (mean_rank - 1) / max(len(names) - 1, 1)   # best=1.0, worst→0
    scores = {n: float(agg[i]) for i, n in enumerate(names)}

    ref_score = scores.get(reference_feature, 0.0)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = [name for name, sc in ranking if sc >= ref_score + cd_margin]
    return {"lookback": lw, "reference": reference_feature, "reference_score": round(ref_score, 4),
            "scorers": use,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "per_scorer": {s: {k: round(v, 4) for k, v in d.items()} for s, d in per_scorer.items()},
            "ranking": [(k, round(v, 4)) for k, v in ranking],
            "selected": selected, "cd_margin": cd_margin}
