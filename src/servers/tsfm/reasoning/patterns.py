"""patterns.py - the pattern-evidence engine (P1: univariate state + rate, grouped).

Given a (standardized) multivariate series, DESCRIBE its shape as structured evidence the LLM can
reason over - never name a fault. This is the server's half of the SenTSR-style split: the server
says "vibration shows a sharp rise; temperature stable"; the LLM says "alignment drift".

Design notes
------------
* Reference-free. SenTSR standardizes each channel by median/MAD, so the baseline is the series
  itself. We robust-standardize internally too (median/MAD) so the same logic works on any input
  and reads in robust-z units - no external "normal" window is needed.
* Channel grouping. The benchmark reasons about "vibration" (= Acceleration + Velocity) vs
  "temperature", so we aggregate member channels into a group before describing (configurable).
* P1 is single-phase (whole-series) state labeling. Phases (changepoints) + bivariate relations
  come in P2/P3; the output dict already nests under a single phase so it extends cleanly.

States: STABLE · RISE · DECLINE · SPIKE · LEVEL_SHIFT · CESSATION · OSCILLATION
Each carries a rate (gradual|sharp where meaningful) and SPIKE carries persistence
(transient|sustained). All purely descriptive.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# --- thresholds (robust-z units); tunable -----------------------------------
_FLAT_EPS = 1e-9  # |diff| below this == constant
_CESSATION_FRAC = 0.25  # a quarter of the series held at one value
_TREND_R = 0.55  # |corr(value, time)| (oscillation/level-shift guards)
_TREND_STRENGTH = 1.5  # Theil-Sen total change (slope*n) in robust-spread units => rise/decline
_SPIKE_Z = 5.0  # an extreme point this far out (robust-z)
_OSC_CROSS = 0.20  # mean-crossings per sample for oscillation
_LEVEL_HALF = 1.0  # half-to-half jump for a level shift (with a dominant single step)


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-12 else (x.std() or 1.0)
    return (x - med) / (scale + 1e-12)


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _pearson_t(x: np.ndarray) -> float:
    t = np.linspace(0.0, 1.0, len(x))
    if np.std(x) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, t)[0, 1])


def _autocorr1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 1e-12 else 0.0


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """De-spike: rolling median removes impulsive outliers (duty-cycle dips, lone spikes) while
    preserving trends, edges and smooth oscillations. The basis for low-frequency shape reading.
    """
    if w <= 1 or len(x) < w:
        return x
    half = w // 2
    xp = np.pad(x, (half, half), mode="edge")
    return np.array([np.median(xp[i : i + w]) for i in range(len(x))])


def _theil_sen(y: np.ndarray) -> float:
    """Theil-Sen slope: median of all pairwise slopes. Robust to outliers (duty-cycle dips), and
    it does NOT fabricate a trend on a flat/noisy signal - the general, principled trend estimator.
    """
    n = len(y)
    idx = np.arange(n, dtype=float)
    dy = np.subtract.outer(y, y)
    dx = np.subtract.outer(idx, idx)
    m = dx > 0
    return float(np.median(dy[m] / dx[m])) if m.any() else 0.0


def classify_state(x: np.ndarray) -> dict:
    """Label one 1-D series with a descriptive state + rate (+ persistence for spikes).

    Scale-robust without destroying scale: trend via correlation (scale-free), 'sharp vs gradual'
    via change CONCENTRATION (one step's share of total variation), magnitudes via the series'
    own robust spread (median/MAD). Assumes the input is already standardized-ish (as SenTSR is);
    grouping pre-standardizes members so this holds.
    """
    x = np.asarray(x, float).ravel()
    n = len(x)
    if n < 4:
        return {
            "state": "STABLE",
            "rate": None,
            "magnitude": 0.0,
            "persistence": None,
            "evidence": {"n": n},
        }
    # RAW: high-frequency features (spike / cessation) read on the original signal
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    spread = 1.4826 * mad if mad > 1e-12 else (float(x.std()) or 1.0)
    same = np.abs(np.diff(x)) <= _FLAT_EPS
    flat_frac = _longest_run(same) / n
    devs = np.abs(x - med) / spread
    ext_idx = np.where(devs >= _SPIKE_Z)[0]
    max_abs = float(devs.max())

    # DE-SPIKED: low-frequency shape (trend / stable / oscillation) - removes impulsive duty-cycle
    # dips & lone spikes so a rise buried under a recurring floor still reads as a rise.
    xs = _rolling_median(
        x, 5
    )  # removes ≤2-consecutive impulses; preserves trends & oscillations
    med_s = float(np.median(xs))
    mad_s = float(np.median(np.abs(xs - med_s)))
    spread_s = 1.4826 * mad_s if mad_s > 1e-12 else (float(xs.std()) or 1.0)
    rng_norm = float((xs.max() - xs.min()) / spread_s)
    r = _pearson_t(xs)
    half = float((np.mean(xs[n // 2 :]) - np.mean(xs[: n // 2])) / spread_s)
    sdiffs = np.abs(np.diff(xs))
    # concentration = one step's share of total variation (rate cue: a single dominant jump)
    concentration = float(sdiffs.max() / (sdiffs.sum() + 1e-12)) if len(sdiffs) else 0.0
    cross_rate = float(np.sum(np.diff(np.sign(xs - med_s)) != 0) / n)
    ac1 = _autocorr1(xs)
    # robust trend on the de-spiked signal (Theil-Sen): outlier-resistant, no fabricated trends
    ts_slope = _theil_sen(xs)
    ts_strength = float(
        ts_slope * n / spread_s
    )  # total rise/fall in robust-spread units

    ev = {
        "trend_r": round(r, 3),
        "trend_strength": round(ts_strength, 2),
        "half_diff": round(half, 3),
        "range_norm": round(rng_norm, 2),
        "concentration": round(concentration, 3),
        "mean_cross_rate": round(cross_rate, 3),
        "autocorr1": round(ac1, 3),
        "flatline_fraction": round(flat_frac, 3),
        "max_abs_dev": round(max_abs, 2),
        "n_extreme": int(len(ext_idx)),
    }

    def out(state, rate=None, persistence=None, onset=None):
        ev2 = dict(ev)
        if onset is not None:
            ev2["onset_frac"] = round(float(onset), 3)
        return {
            "state": state,
            "rate": rate,
            "magnitude": round(max(abs(half), max_abs), 2),
            "persistence": persistence,
            "evidence": ev2,
        }

    # truly constant → stable
    if rng_norm < 1e-6:
        return out("STABLE")

    # CESSATION - a long constant run at/below baseline while the rest is active
    if flat_frac >= _CESSATION_FRAC and rng_norm >= 2.0:
        cur = best = best_end = 0
        for k, v in enumerate(same):
            cur = cur + 1 if v else 0
            if cur > best:
                best, best_end = cur, k
        run_level = float(np.median(x[max(0, best_end - best) : best_end + 1]))
        if run_level <= med:
            return out("CESSATION", onset=(best_end - best) / n)

    # SPIKE - a few extreme points, not a sustained trend
    if (
        1 <= len(ext_idx) <= max(3, int(0.03 * n))
        and max_abs >= _SPIKE_Z
        and abs(r) < _TREND_R
    ):
        first, last = int(ext_idx[0]), int(ext_idx[-1])
        before = float(np.mean(x[:first])) if first > 0 else med
        after = float(np.mean(x[last + 1 :])) if last + 1 < n else med
        persistence = "sustained" if (after - before) / spread > 2.0 else "transient"
        return out("SPIKE", persistence=persistence, onset=first / n)

    # RISE / DECLINE: robust Theil-Sen trend over a meaningful magnitude
    if abs(ts_strength) >= _TREND_STRENGTH:
        state = "RISE" if ts_slope > 0 else "DECLINE"
        rate = "sharp" if concentration >= 0.30 else "gradual"  # one dominant jump => sharp
        return out(state, rate=rate)

    # LEVEL_SHIFT: a step between two regimes, big half-to-half change dominated by one jump
    if abs(half) >= _LEVEL_HALF and concentration >= 0.30:
        return out("LEVEL_SHIFT", onset=int(np.argmax(sdiffs)) / n)

    # OSCILLATION - periodic swings (frequent crossings + smooth, i.e. autocorrelated), no trend
    if cross_rate >= _OSC_CROSS and ac1 >= 0.5 and abs(r) < 0.4 and rng_norm >= 1.5:
        return out("OSCILLATION")

    return out("STABLE")


# --- channel grouping (generic; domain grouping is opt-in) -------------------
# The engine is channel-agnostic: any names, any count. By DEFAULT every channel is its own group
# (no domain assumptions). Grouping is optional and supplied by the caller/agent - either an
# explicit {group: [channels]} map, or by applying a name-rule preset via auto_group(). This keeps
# the SenTSR "vibration = accel+velocity" choice out of the engine and in the hands of whoever
# knows the domain.
GROUP_PRESETS: Dict[str, Dict[str, str]] = {  # preset name → {name-substring: group}
    "vibration_temperature": {
        "accel": "vibration",
        "veloc": "vibration",
        "vibrat": "vibration",
        "temp": "temperature",
    },
}


def identity_groups(channel_names: List[str]) -> Dict[str, List[str]]:
    """Default, domain-free grouping: each channel is its own single-member group."""
    return {name: [name] for name in channel_names}


def auto_group(channel_names: List[str], rules) -> Dict[str, List[str]]:
    """Optional name-based grouping. `rules` is a {substring: group} dict or the name of a preset in
    GROUP_PRESETS. Channels matching no rule become their own group. Caller opts in explicitly.
    """
    if isinstance(rules, str):
        rules = GROUP_PRESETS[rules]
    groups: Dict[str, List[str]] = {}
    for name in channel_names:
        low = name.lower()
        g = next((grp for key, grp in rules.items() if key in low), name)
        groups.setdefault(g, []).append(name)
    return groups


def _to_channels(data, channel_names: Optional[List[str]]) -> Dict[str, np.ndarray]:
    """Normalize input (DataFrame | dict | 2-D array) to {name: 1-D array}."""
    if hasattr(data, "columns"):  # pandas DataFrame
        return {str(c): np.asarray(data[c], float) for c in data.columns}
    if isinstance(data, dict):
        return {str(k): np.asarray(v, float).ravel() for k, v in data.items()}
    arr = np.asarray(data, float)
    if arr.ndim == 1:
        arr = arr[None, :]
    names = channel_names or [f"ch{i}" for i in range(arr.shape[0])]
    return {names[i]: arr[i] for i in range(arr.shape[0])}


def group_signal(channels: Dict[str, np.ndarray], members: List[str]) -> np.ndarray:
    """Aggregate member channels into one group signal: robust-standardize each, then average
    (so a multi-axis 'vibration' group reads on a common scale)."""
    stacks = [_robust_z(channels[m]) for m in members if m in channels]
    return np.mean(np.column_stack(stacks), axis=1) if stacks else np.zeros(1)


# --- the descriptive phrase + series description -----------------------------
_PHRASE = {  # (state, rate?) → shape words - strictly descriptive, no fault terms
    ("STABLE", None): "stable, near baseline",
    ("RISE", "gradual"): "a gradual rise",
    ("RISE", "sharp"): "a sharp rise",
    ("RISE", None): "a rise",
    ("DECLINE", "gradual"): "a gradual decline",
    ("DECLINE", "sharp"): "a sharp decline",
    ("DECLINE", None): "a decline",
    ("LEVEL_SHIFT", None): "a level shift to a new band",
    ("CESSATION", None): "goes quiet (flatline)",
    ("OSCILLATION", None): "oscillation around baseline",
}


def _phrase(s: dict) -> str:
    if s["state"] == "SPIKE":
        return (
            "a spike followed by sustained elevation"
            if s.get("persistence") == "sustained"
            else "a transient spike"
        )
    return _PHRASE.get((s["state"], s.get("rate"))) or _PHRASE.get(
        (s["state"], None), s["state"].lower()
    )


# --- P2: bivariate relation between two group signals -------------------------
_REL_CORR = 0.5  # |corr| for a real association
_REL_LAG_GAIN = 0.1  # lagged corr must beat lag-0 by this to call it lead-lag


def _corr_at(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """corr(a, b) with b shifted: lag>0 means a leads b by `lag` steps."""
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    if len(x) < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: Optional[int] = None):
    """Best lead-lag: scan lags, return (lag, corr) maximizing |corr|. lag>0 ⇒ a leads b."""
    n = min(len(a), len(b))
    max_lag = max_lag if max_lag is not None else min(n // 4, 24)
    best = (0, _corr_at(a, b, 0))
    for k in range(-max_lag, max_lag + 1):
        c = _corr_at(a, b, k)
        if abs(c) > abs(best[1]):
            best = (k, c)
    return best


def relate_pair(
    sig_a, sig_b, state_a: str, state_b: str, max_lag: Optional[int] = None
) -> dict:
    """Classify the temporal relation between two group signals (generic, any pair):
    DECOUPLED (one moves, other flat) · CO_MOVE (move together) · LEAD_LAG (one precedes the
    other) · INDEPENDENT (both move, uncorrelated) · NONE (both flat)."""
    a = np.asarray(sig_a, float)
    b = np.asarray(sig_b, float)
    r0 = _corr_at(a, b, 0)
    lag, clag = xcorr_lag(a, b, max_lag)
    act_a, act_b = state_a != "STABLE", state_b != "STABLE"
    base = {"r0": round(r0, 3), "lag": int(lag), "xcorr": round(clag, 3)}
    if not act_a and not act_b:
        return {"type": "NONE", **base}
    if act_a != act_b:
        return {"type": "DECOUPLED", **base}
    if (
        abs(lag) >= 2
        and abs(clag) - abs(r0) >= _REL_LAG_GAIN
        and abs(clag) >= _REL_CORR
    ):
        return {"type": "LEAD_LAG", "leader": "a" if lag > 0 else "b", **base}
    if abs(r0) >= _REL_CORR:
        return {
            "type": "CO_MOVE",
            "direction": "same" if r0 > 0 else "opposite",
            **base,
        }
    return {"type": "INDEPENDENT", **base}


# --- P3: changepoint segmentation (conservative CUSUM) ------------------------
def _cusum_cps(x: np.ndarray, min_seg: int, shift: float, depth: int = 0) -> List[int]:
    """Recursive single-CUSUM changepoints: split where the mean shifts by >= `shift` robust
    units and both sides are >= min_seg. Conservative - gradual drifts yield no split.
    """
    n = len(x)
    if n < 2 * min_seg or depth > 3:
        return []
    c = np.cumsum(x - x.mean())
    cp = int(np.argmax(np.abs(c)))
    if cp < min_seg or cp > n - min_seg:
        return []
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    spread = 1.4826 * mad if mad > 1e-12 else (x.std() or 1.0)
    if abs(np.mean(x[:cp]) - np.mean(x[cp:])) / spread < shift:
        return []
    left = _cusum_cps(x[:cp], min_seg, shift, depth + 1)
    right = [cp + i for i in _cusum_cps(x[cp:], min_seg, shift, depth + 1)]
    return left + [cp] + right


def segment(
    group_sigs: Dict[str, np.ndarray],
    *,
    min_seg: Optional[int] = None,
    shift: float = 3.0,
) -> List[int]:
    """Union of changepoints across all groups → phase boundaries (sorted, deduped, merged).
    Conservative: de-spikes each signal first and requires a large mean shift, so only clear
    regime changes split (gradual drift / noise stays one phase)."""
    n = len(next(iter(group_sigs.values()))) if group_sigs else 0
    min_seg = min_seg if min_seg is not None else max(12, n // 8)
    cps: set = set()
    for sig in group_sigs.values():
        cps.update(
            _cusum_cps(_rolling_median(np.asarray(sig, float), 5), min_seg, shift)
        )
    pts = sorted(cps)
    merged: List[int] = []
    for p in pts:  # drop near-duplicate boundaries
        if not merged or p - merged[-1] >= min_seg:
            merged.append(p)
    return [0] + merged + [n]


def describe_series(
    data,
    *,
    channel_names: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    group_rules=None,
    segment_phases: bool = True,
) -> dict:
    """P1 evidence: label each channel-group's whole-series state+rate, render a shape-only NL
    summary. Single-phase; the dict nests under one phase for forward-compat.

    Generic by default: with no `groups`/`group_rules`, every channel is its own group (works for
    any signals, any count, any names). Pass `groups={group:[channels]}` for explicit grouping, or
    `group_rules=<dict|preset-name>` (e.g. "vibration_temperature") to auto-group by channel name.
    """
    channels = _to_channels(data, channel_names)
    if groups is None:
        groups = (
            auto_group(list(channels), group_rules)
            if group_rules
            else identity_groups(list(channels))
        )
    sigs = {g: group_signal(channels, members) for g, members in groups.items()}
    n = len(next(iter(channels.values()))) if channels else 0
    group_names = list(groups)
    pairs = [
        (group_names[i], group_names[j])
        for i in range(len(group_names))
        for j in range(i + 1, len(group_names))
    ]

    bounds = segment(sigs) if (segment_phases and n) else [0, n]
    phases = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 4:  # skip degenerate slivers
            continue
        per_group = {g: classify_state(sig[s:e]) for g, sig in sigs.items()}
        relations = []
        for a, b in pairs:
            rel = relate_pair(
                sigs[a][s:e], sigs[b][s:e], per_group[a]["state"], per_group[b]["state"]
            )
            if rel["type"] != "NONE":
                relations.append({"a": a, "b": b, **rel})
        phases.append(
            {"span": [int(s), int(e)], "per_group": per_group, "relations": relations}
        )

    summary = _summarize(phases)
    return {
        "groups": {g: members for g, members in groups.items()},
        "phases": phases,
        "n_observations": n,
        "standardized": True,
        "summary": summary,
    }


_REL_PHRASE = {
    "DECOUPLED": lambda r: f"{r['a']} and {r['b']} are decoupled",
    "CO_MOVE": lambda r: f"{r['a']} and {r['b']} move together",
    "LEAD_LAG": lambda r: (
        f"{r['a'] if r.get('leader') == 'a' else r['b']} leads "
        f"{r['b'] if r.get('leader') == 'a' else r['a']} by {abs(r['lag'])} steps"
    ),
    "INDEPENDENT": lambda r: f"{r['a']} and {r['b']} both change, uncorrelated",
}


def _summarize(phases: List[dict]) -> str:
    """Shape-only NL rendering. Single phase → one clause; multi-phase → numbered phases."""

    def phase_text(ph):
        states = "; ".join(f"{g}: {_phrase(s)}" for g, s in ph["per_group"].items())
        rels = "; ".join(
            _REL_PHRASE[r["type"]](r)
            for r in ph.get("relations", [])
            if r["type"] in _REL_PHRASE
        )
        return states + (f" ({rels})" if rels else "")

    if not phases:
        return ""
    if len(phases) == 1:
        return phase_text(phases[0]) + "."
    return (
        " | ".join(f"Phase {i+1}: {phase_text(ph)}" for i, ph in enumerate(phases))
        + "."
    )
