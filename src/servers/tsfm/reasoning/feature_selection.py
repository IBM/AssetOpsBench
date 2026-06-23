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
    "abs_diff_mean": lambda w: (
        float(np.mean(np.abs(np.diff(w)))) if len(w) > 1 else 0.0
    ),
    # Frequency
    "spectral_centroid": lambda w: float(_spectral_centroid(w)),
    "dominant_freq_power": lambda w: float(_dominant_freq_power(w)),
}


def _kurtosis(w):
    w = np.asarray(w, float)
    s = w.std()
    return 0.0 if s < 1e-9 else float(np.mean(((w - w.mean()) / s) ** 4) - 3)


def _skew(w):
    w = np.asarray(w, float)
    s = w.std()
    return 0.0 if s < 1e-9 else float(np.mean(((w - w.mean()) / s) ** 3))


def _slope(w):
    w = np.asarray(w, float)
    t = np.arange(len(w)) - (len(w) - 1) / 2.0
    d = (t**2).sum() or 1.0
    return float((t * w).sum() / d)


def _autocorr(w, lag):
    w = np.asarray(w, float) - np.mean(w)
    v = (w**2).sum()
    return 0.0 if v < 1e-9 or len(w) <= lag else float((w[:-lag] * w[lag:]).sum() / v)


def _spectral_centroid(w):
    sp = np.abs(np.fft.rfft(np.asarray(w, float) - np.mean(w)))
    f = np.arange(len(sp))
    return 0.0 if sp.sum() < 1e-9 else float((f * sp).sum() / sp.sum())


def _dominant_freq_power(w):
    sp = np.abs(np.fft.rfft(np.asarray(w, float) - np.mean(w)))
    return 0.0 if len(sp) <= 1 else float(sp[1:].max())


# ---- extended helpers for the full FLOps-style library (numpy only, robust) ----
def _safe(x):
    x = float(x)
    return x if np.isfinite(x) else 0.0


def _crossings(w, level):
    w = np.asarray(w, float) - level
    return int(np.sum(w[:-1] * w[1:] < 0)) if len(w) > 1 else 0


def _longest_strike(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def _rms(w):
    w = np.asarray(w, float)
    return float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0


def _binned_entropy(w, bins=10):
    w = np.asarray(w, float)
    if len(w) < 2 or w.max() - w.min() < 1e-12:
        return 0.0
    hist, _ = np.histogram(w, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum())


def _spectrum(w):
    w = np.asarray(w, float) - np.mean(w)
    ps = np.abs(np.fft.rfft(w)) ** 2
    return ps


def _spectral_entropy(w):
    ps = _spectrum(w)
    if ps.sum() < 1e-12:
        return 0.0
    p = ps / ps.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _spectral_rolloff(w, frac=0.85):
    ps = _spectrum(w)
    tot = ps.sum()
    if tot < 1e-12:
        return 0.0
    c = np.cumsum(ps)
    return float(np.searchsorted(c, frac * tot) / max(len(ps) - 1, 1))


def _spectral_flatness(w):
    ps = _spectrum(w) + 1e-12
    return float(np.exp(np.mean(np.log(ps))) / np.mean(ps))


def _band_energy(w, lo, hi):
    ps = _spectrum(w)
    n = len(ps)
    a, b = int(lo * n), max(int(hi * n), int(lo * n) + 1)
    tot = ps.sum()
    return float(ps[a:b].sum() / tot) if tot > 1e-12 else 0.0


def _hjorth(w):
    w = np.asarray(w, float)
    d1 = np.diff(w)
    d2 = np.diff(d1)
    v0, v1, v2 = np.var(w), np.var(d1) if len(d1) else 0.0, np.var(d2) if len(d2) else 0.0
    mob = np.sqrt(v1 / v0) if v0 > 1e-12 else 0.0
    comp = (np.sqrt(v2 / v1) / mob) if (v1 > 1e-12 and mob > 1e-12) else 0.0
    return float(mob), float(comp)


def _cid_ce(w):
    w = np.asarray(w, float)
    s = w.std()
    if s < 1e-12:
        return 0.0
    z = (w - w.mean()) / s
    return float(np.sqrt(np.sum(np.diff(z) ** 2)))


def _c3(w, lag=1):
    w = np.asarray(w, float)
    n = len(w)
    if n <= 2 * lag:
        return 0.0
    return float(np.mean(w[: n - 2 * lag] * w[lag : n - lag] * w[2 * lag :]))


def _tra(w, lag=1):                       # time-reversal asymmetry
    w = np.asarray(w, float)
    n = len(w)
    if n <= 2 * lag:
        return 0.0
    a, b, c = w[2 * lag :], w[lag : n - lag], w[: n - 2 * lag]
    return float(np.mean(a * a * b - b * c * c))


def _perm_entropy(w, order=3):
    w = np.asarray(w, float)
    n = len(w)
    if n < order + 1:
        return 0.0
    from itertools import permutations
    perms = list(permutations(range(order)))
    counts = {p: 0 for p in perms}
    for i in range(n - order + 1):
        counts[tuple(np.argsort(w[i : i + order]))] += 1
    c = np.array([v for v in counts.values() if v > 0], float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(perms)))


def _linregress_r(w):
    w = np.asarray(w, float)
    t = np.arange(len(w), dtype=float)
    if w.std() < 1e-12 or t.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(t, w)[0, 1])


def _half_mean_diff(w):
    w = np.asarray(w, float)
    h = len(w) // 2
    return float(np.mean(w[h:]) - np.mean(w[:h])) if h else 0.0


def _quantile(w, q):
    return float(np.percentile(np.asarray(w, float), q))


# ---- the full library: extend the curated 15 toward the FLOps 100+ ----
def _add(name, fn):
    EXTRACTORS[name] = fn


# distribution / profiling
EXTRACTORS.update({
    "var": lambda w: _safe(np.var(w)),
    "median": lambda w: _safe(np.median(w)),
    "q05": lambda w: _quantile(w, 5), "q10": lambda w: _quantile(w, 10),
    "q90": lambda w: _quantile(w, 90), "q95": lambda w: _quantile(w, 95),
    "iqr": lambda w: _quantile(w, 75) - _quantile(w, 25),
    "mad": lambda w: _safe(np.median(np.abs(np.asarray(w, float) - np.median(w)))),
    "mean_abs": lambda w: _safe(np.mean(np.abs(w))),
    "sum_abs": lambda w: _safe(np.sum(np.abs(w))),
    "rms": lambda w: _rms(w),
    "abs_energy": lambda w: _safe(np.sum(np.asarray(w, float) ** 2)),
    "cv": lambda w: _safe(np.std(w) / (abs(np.mean(w)) + 1e-12)),
    "std_to_mean": lambda w: _safe(np.std(w) / (np.mean(w) + 1e-12)),
    "count_above_mean": lambda w: float(np.sum(np.asarray(w, float) > np.mean(w))),
    "count_below_mean": lambda w: float(np.sum(np.asarray(w, float) < np.mean(w))),
    "ratio_above_mean": lambda w: _safe(np.mean(np.asarray(w, float) > np.mean(w))),
    "count_above_2std": lambda w: float(np.sum(np.abs(np.asarray(w, float) - np.mean(w)) > 2 * np.std(w))),
    "abs_max": lambda w: _safe(np.max(np.abs(w))),
    "abs_min": lambda w: _safe(np.min(np.abs(w))),
})

# temporal / order-dependent
EXTRACTORS.update({
    "intercept": lambda w: _safe(np.mean(w)),
    "autocorr2": lambda w: _autocorr(w, 2), "autocorr3": lambda w: _autocorr(w, 3),
    "autocorr5": lambda w: _autocorr(w, 5), "autocorr10": lambda w: _autocorr(w, 10),
    "mean_diff": lambda w: _safe(np.mean(np.diff(w))) if len(w) > 1 else 0.0,
    "mean_2nd_diff": lambda w: _safe(np.mean(np.diff(w, 2))) if len(w) > 2 else 0.0,
    "abs_2nd_diff_mean": lambda w: _safe(np.mean(np.abs(np.diff(w, 2)))) if len(w) > 2 else 0.0,
    "std_diff": lambda w: _safe(np.std(np.diff(w))) if len(w) > 1 else 0.0,
    "num_zero_crossings": lambda w: float(_crossings(w, 0.0)),
    "num_mean_crossings": lambda w: float(_crossings(w, float(np.mean(w)))),
    "longest_above_mean": lambda w: float(_longest_strike(np.asarray(w, float) > np.mean(w))),
    "longest_below_mean": lambda w: float(_longest_strike(np.asarray(w, float) < np.mean(w))),
    "first_loc_max": lambda w: _safe(int(np.argmax(w)) / len(w)),
    "last_loc_max": lambda w: _safe((len(w) - int(np.argmax(w[::-1])) - 1) / len(w)),
    "first_loc_min": lambda w: _safe(int(np.argmin(w)) / len(w)),
    "mean_change": lambda w: _safe((w[-1] - w[0]) / len(w)) if len(w) > 1 else 0.0,
    "cid_ce": lambda w: _cid_ce(w),
    "c3_lag1": lambda w: _c3(w, 1), "c3_lag2": lambda w: _c3(w, 2),
    "time_reversal_asym": lambda w: _tra(w, 1),
    "distinct_ratio": lambda w: _safe(len(np.unique(np.round(np.asarray(w, float), 6))) / len(w)),
})

# complexity / entropy
EXTRACTORS.update({
    "binned_entropy": lambda w: _binned_entropy(w, 10),
    "perm_entropy": lambda w: _perm_entropy(w, 3),
    "spectral_entropy": lambda w: _spectral_entropy(w),
    "hjorth_mobility": lambda w: _hjorth(w)[0],
    "hjorth_complexity": lambda w: _hjorth(w)[1],
})

# frequency
EXTRACTORS.update({
    "spectral_rolloff": lambda w: _spectral_rolloff(w),
    "spectral_flatness": lambda w: _spectral_flatness(w),
    "dc_power": lambda w: _safe(abs(np.mean(w))),
    "total_spectral_energy": lambda w: _safe(_spectrum(w).sum()),
    "band_low": lambda w: _band_energy(w, 0.0, 0.33),
    "band_mid": lambda w: _band_energy(w, 0.33, 0.66),
    "band_high": lambda w: _band_energy(w, 0.66, 1.0),
    "dominant_freq": lambda w: _safe(int(np.argmax(_spectrum(w)[1:]) + 1) if len(_spectrum(w)) > 1 else 0),
})

# shape / vibration diagnostics
EXTRACTORS.update({
    "peak_to_peak": lambda w: _safe(np.max(w) - np.min(w)),
    "crest_factor": lambda w: _safe(np.max(np.abs(w)) / (_rms(w) + 1e-12)),
    "shape_factor": lambda w: _safe(_rms(w) / (np.mean(np.abs(w)) + 1e-12)),
    "impulse_factor": lambda w: _safe(np.max(np.abs(w)) / (np.mean(np.abs(w)) + 1e-12)),
    "clearance_factor": lambda w: _safe(np.max(np.abs(w)) / ((np.mean(np.sqrt(np.abs(w)))) ** 2 + 1e-12)),
    "margin_factor": lambda w: _safe((np.max(w) - np.min(w)) / (_rms(w) + 1e-12)),
    "form_factor": lambda w: _safe(_rms(w) / (abs(np.mean(w)) + 1e-12)),
})

# trend / stationarity
EXTRACTORS.update({
    "linear_trend_r": lambda w: _linregress_r(w),
    "half_mean_diff": lambda w: _half_mean_diff(w),
    "half_std_ratio": lambda w: _safe(np.std(np.asarray(w, float)[len(w)//2:]) / (np.std(np.asarray(w, float)[:len(w)//2]) + 1e-12)),
    "energy_ratio_first_half": lambda w: _safe(np.sum(np.asarray(w, float)[:len(w)//2] ** 2) / (np.sum(np.asarray(w, float) ** 2) + 1e-12)),
    "trend_strength": lambda w: _safe(abs(_slope(w)) * len(w) / (np.std(w) + 1e-9)),
    "cumsum_argmax_ratio": lambda w: _safe(int(np.argmax(np.cumsum(np.asarray(w, float) - np.mean(w)))) / len(w)),
    "cumsum_max": lambda w: _safe(np.max(np.cumsum(np.asarray(w, float) - np.mean(w)))),
})

# additional profiling / temporal / frequency / shape to complete the FLOps-style library
EXTRACTORS.update({
    "q33": lambda w: _quantile(w, 33), "q66": lambda w: _quantile(w, 66),
    "range_to_std": lambda w: _safe((np.max(w) - np.min(w)) / (np.std(w) + 1e-12)),
    "mean_square": lambda w: _safe(np.mean(np.asarray(w, float) ** 2)),
    "abs_sum_changes": lambda w: _safe(np.sum(np.abs(np.diff(w)))) if len(w) > 1 else 0.0,
    "max_diff": lambda w: _safe(np.max(np.diff(w))) if len(w) > 1 else 0.0,
    "min_diff": lambda w: _safe(np.min(np.diff(w))) if len(w) > 1 else 0.0,
    "var_diff": lambda w: _safe(np.var(np.diff(w))) if len(w) > 1 else 0.0,
    "positive_diff_ratio": lambda w: _safe(np.mean(np.diff(w) > 0)) if len(w) > 1 else 0.0,
    "autocorr4": lambda w: _autocorr(w, 4), "autocorr7": lambda w: _autocorr(w, 7),
    "num_peaks": lambda w: float(np.sum((np.asarray(w, float)[1:-1] > np.asarray(w, float)[:-2]) &
                                        (np.asarray(w, float)[1:-1] > np.asarray(w, float)[2:]))) if len(w) > 2 else 0.0,
    "num_valleys": lambda w: float(np.sum((np.asarray(w, float)[1:-1] < np.asarray(w, float)[:-2]) &
                                          (np.asarray(w, float)[1:-1] < np.asarray(w, float)[2:]))) if len(w) > 2 else 0.0,
    "longest_above_2std": lambda w: float(_longest_strike(np.abs(np.asarray(w, float) - np.mean(w)) > 2 * np.std(w))),
    "zero_crossing_rate": lambda w: _safe(_crossings(w, float(np.mean(w))) / len(w)),
    "skew_abs": lambda w: _safe(abs(_skew(w))),
    "p2p_to_std": lambda w: _safe((np.max(w) - np.min(w)) / (np.std(w) + 1e-12)),
    "band_q1": lambda w: _band_energy(w, 0.0, 0.25), "band_q2": lambda w: _band_energy(w, 0.25, 0.5),
    "band_q3": lambda w: _band_energy(w, 0.5, 0.75), "band_q4": lambda w: _band_energy(w, 0.75, 1.0),
    "mean_psd": lambda w: _safe(np.mean(_spectrum(w))),
    "peak_psd_ratio": lambda w: _safe(_spectrum(w)[1:].max() / (_spectrum(w)[1:].sum() + 1e-12)) if len(_spectrum(w)) > 1 else 0.0,
    "diff_entropy": lambda w: _binned_entropy(np.diff(w), 10) if len(w) > 1 else 0.0,
    "quarter_mean_diff": lambda w: _safe(np.mean(np.asarray(w, float)[3 * (len(w)//4):]) - np.mean(np.asarray(w, float)[:len(w)//4])) if len(w) >= 4 else 0.0,
})


# one-line descriptions per extractor — surfaced on the catalog card so the agent can select
EXTRACTOR_DOC = {
    "mean": "Average value of the window.",
    "std": "Standard deviation (spread).",
    "min": "Minimum value.", "max": "Maximum value.",
    "range": "Max minus min (peak-to-peak spread).",
    "q25": "25th percentile (lower quartile).", "q75": "75th percentile (upper quartile).",
    "kurtosis": "Tailedness/peakedness of the distribution (excess kurtosis).",
    "skew": "Asymmetry of the distribution.",
    "slope": "Linear trend slope over the window.",
    "autocorr1": "Lag-1 autocorrelation (short-term persistence).",
    "energy": "Mean squared value (average power).",
    "abs_diff_mean": "Mean absolute first difference (roughness).",
    "spectral_centroid": "Center of mass of the frequency spectrum (brightness).",
    "dominant_freq_power": "Power of the strongest non-DC frequency.",
    "var": "Variance of the window.", "median": "Median value (robust center).",
    "q05": "5th percentile.", "q10": "10th percentile.", "q90": "90th percentile.",
    "q95": "95th percentile.", "iqr": "Inter-quartile range (robust spread).",
    "mad": "Median absolute deviation (robust spread).",
    "mean_abs": "Mean of absolute values.", "sum_abs": "Sum of absolute values.",
    "rms": "Root-mean-square amplitude.", "abs_energy": "Sum of squared values (total energy).",
    "cv": "Coefficient of variation (std/|mean|).",
    "std_to_mean": "Std divided by mean (relative dispersion).",
    "count_above_mean": "Number of points above the mean.",
    "count_below_mean": "Number of points below the mean.",
    "ratio_above_mean": "Fraction of points above the mean.",
    "count_above_2std": "Number of points beyond 2 std (outliers).",
    "abs_max": "Maximum absolute value.", "abs_min": "Minimum absolute value.",
    "intercept": "Mean level (regression intercept proxy).",
    "autocorr2": "Lag-2 autocorrelation.", "autocorr3": "Lag-3 autocorrelation.",
    "autocorr5": "Lag-5 autocorrelation.", "autocorr10": "Lag-10 autocorrelation (longer memory).",
    "mean_diff": "Mean of first differences (average step).",
    "mean_2nd_diff": "Mean of second differences (curvature).",
    "abs_2nd_diff_mean": "Mean absolute second difference.",
    "std_diff": "Std of first differences (volatility of change).",
    "num_zero_crossings": "Count of sign changes around zero.",
    "num_mean_crossings": "Count of crossings of the mean level.",
    "longest_above_mean": "Longest run of points above the mean.",
    "longest_below_mean": "Longest run of points below the mean.",
    "first_loc_max": "Relative position (0-1) of the first maximum.",
    "last_loc_max": "Relative position of the last maximum.",
    "first_loc_min": "Relative position of the first minimum.",
    "mean_change": "Average per-step change ((last-first)/n).",
    "cid_ce": "Complexity-invariant distance estimate (curve complexity).",
    "c3_lag1": "Non-linearity measure C3 at lag 1.",
    "c3_lag2": "Non-linearity measure C3 at lag 2.",
    "time_reversal_asym": "Time-reversal asymmetry statistic (lag 1).",
    "distinct_ratio": "Ratio of distinct values to length.",
    "binned_entropy": "Shannon entropy of a 10-bin value histogram.",
    "perm_entropy": "Permutation entropy (ordinal complexity, order 3).",
    "spectral_entropy": "Entropy of the normalized power spectrum.",
    "hjorth_mobility": "Hjorth mobility (mean frequency).",
    "hjorth_complexity": "Hjorth complexity (spectral bandwidth/shape change).",
    "spectral_rolloff": "Frequency below which 85% of spectral energy lies.",
    "spectral_flatness": "Geometric/arithmetic mean of spectrum (tonal vs noisy).",
    "dc_power": "Magnitude of the DC component (|mean|).",
    "total_spectral_energy": "Total power across the spectrum.",
    "band_low": "Spectral-energy fraction in the low band (0-1/3).",
    "band_mid": "Spectral-energy fraction in the mid band (1/3-2/3).",
    "band_high": "Spectral-energy fraction in the high band (2/3-1).",
    "dominant_freq": "Index of the dominant non-DC frequency.",
    "peak_to_peak": "Max minus min amplitude.",
    "crest_factor": "Peak/RMS — impulsiveness (vibration health).",
    "shape_factor": "RMS/mean-absolute — waveform shape (vibration).",
    "impulse_factor": "Peak/mean-absolute — impulsiveness (bearing faults).",
    "clearance_factor": "Peak/(mean sqrt|x|)^2 — early bearing wear (vibration).",
    "margin_factor": "Peak-to-peak / RMS.", "form_factor": "RMS / |mean|.",
    "linear_trend_r": "Pearson correlation of value vs time (trend strength).",
    "half_mean_diff": "Mean of 2nd half minus 1st half (drift).",
    "half_std_ratio": "Std of 2nd half / 1st half (variance shift).",
    "energy_ratio_first_half": "Fraction of total energy in the first half.",
    "trend_strength": "Normalized trend magnitude (|slope|*n/std).",
    "cumsum_argmax_ratio": "Relative position of the cumulative-sum peak.",
    "cumsum_max": "Maximum of the mean-centered cumulative sum.",
    "q33": "33rd percentile.", "q66": "66th percentile.",
    "range_to_std": "Range divided by std.", "mean_square": "Mean of squared values.",
    "abs_sum_changes": "Sum of absolute first differences (total variation).",
    "max_diff": "Largest single-step increase.", "min_diff": "Largest single-step decrease.",
    "var_diff": "Variance of first differences.",
    "positive_diff_ratio": "Fraction of upward steps.",
    "autocorr4": "Lag-4 autocorrelation.", "autocorr7": "Lag-7 autocorrelation.",
    "num_peaks": "Count of local maxima.", "num_valleys": "Count of local minima.",
    "longest_above_2std": "Longest run beyond 2 std (sustained anomaly).",
    "zero_crossing_rate": "Mean-crossings per sample.",
    "skew_abs": "Absolute skewness.", "p2p_to_std": "Peak-to-peak / std.",
    "band_q1": "Spectral-energy fraction, quartile band 0-25%.",
    "band_q2": "Spectral-energy fraction, quartile band 25-50%.",
    "band_q3": "Spectral-energy fraction, quartile band 50-75%.",
    "band_q4": "Spectral-energy fraction, quartile band 75-100%.",
    "mean_psd": "Mean power spectral density.",
    "peak_psd_ratio": "Peak PSD / total PSD (spectral peakedness).",
    "diff_entropy": "Binned entropy of first differences.",
    "quarter_mean_diff": "Mean of last quarter minus first quarter (drift).",
}


def describe(name: str) -> str:
    return EXTRACTOR_DOC.get(name, f"FLOps extractor '{name}'.")


# --------------------------------------------------------------------------- #
def discover_lookback(series: np.ndarray, max_lw: int = 128) -> int:
    """Dataset-specific look-back window via dominant spectral period (FLOps: lw from
    spectral/frequency analysis). Falls back to a sane default."""
    x = np.asarray(series, float).ravel()
    x = x - x.mean()
    if len(x) < 8:
        return min(8, len(x))
    sp = np.abs(np.fft.rfft(x))
    if len(sp) <= 2 or sp[1:].max() < 1e-9:
        return min(32, len(x) // 2 or 8)
    k = 1 + int(np.argmax(sp[1:]))  # dominant non-DC bin
    period = int(round(len(x) / k))
    return int(max(8, min(max_lw, period)))


def _tabulate(series: np.ndarray, lw: int):
    """Slide a window; X = windows[:-1], y = next value (forecasting target)."""
    x = np.asarray(series, float).ravel()
    wins = np.stack([x[i : i + lw] for i in range(len(x) - lw)])  # (N, lw)
    y = x[lw:]  # next value
    return wins, y


def _corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


# --------------------------------------------------------------------------- #
# Multi-config scorers (FLOps: score under several criteria, aggregate by mean rank).
# Each returns a per-feature score vector aligned to `names`; higher = more relevant.
# --------------------------------------------------------------------------- #
def _norm(v):
    v = np.asarray(v, float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
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


_SCORERS = {
    "corr": _score_corr,
    "f_test": _score_ftest,
    "mutual_info": _score_mutual_info,
    "model": _score_model,
}


def _feature_matrix(wins, ex):
    names = list(ex)
    F = np.column_stack([[ex[n](w) for w in wins] for n in names]).astype(float)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, names


def select_features(
    series: np.ndarray,
    *,
    reference_feature: str = "mean",
    lookback: Optional[int] = None,
    cd_margin: float = 0.05,
    extractors: Optional[Dict[str, Callable]] = None,
    scorers: Optional[List[str]] = None,
) -> dict:
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
        order = (-sv).argsort()  # best→worst
        rk = np.empty(len(names))
        rk[order] = np.arange(1, len(names) + 1)
        ranks += rk
    mean_rank = ranks / len(use)
    agg = 1.0 - (mean_rank - 1) / max(len(names) - 1, 1)  # best=1.0, worst→0
    scores = {n: float(agg[i]) for i, n in enumerate(names)}

    ref_score = scores.get(reference_feature, 0.0)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = [name for name, sc in ranking if sc >= ref_score + cd_margin]
    return {
        "lookback": lw,
        "reference": reference_feature,
        "reference_score": round(ref_score, 4),
        "scorers": use,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "per_scorer": {
            s: {k: round(v, 4) for k, v in d.items()} for s, d in per_scorer.items()
        },
        "ranking": [(k, round(v, 4)) for k, v in ranking],
        "selected": selected,
        "cd_margin": cd_margin,
    }
