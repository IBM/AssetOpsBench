"""P1 pattern engine: univariate state+rate labeling, channel grouping, shape-only NL summary."""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from ..reasoning import patterns as P

N = 168


def _noise(scale=0.15, seed=0):
    return scale * np.random.RandomState(seed).randn(N)   # deterministic, order-independent


def test_stable():
    s = P.classify_state(_noise())
    assert s["state"] == "STABLE"


def test_gradual_rise():
    s = P.classify_state(np.linspace(0, 6, N) + _noise())
    assert s["state"] == "RISE" and s["rate"] == "gradual"


def test_sharp_rise():
    x = np.concatenate([np.zeros(N // 2), np.full(N // 2, 6.0)]) + _noise()  # a step up = sharp
    s = P.classify_state(x)
    assert s["state"] in ("RISE", "LEVEL_SHIFT")          # step reads as a sharp transition
    if s["state"] == "RISE":
        assert s["rate"] == "sharp"


def test_decline():
    s = P.classify_state(np.linspace(5, -2, N) + _noise())
    assert s["state"] == "DECLINE"


def test_transient_spike():
    x = _noise(0.1).copy(); x[80] += 12.0
    s = P.classify_state(x)
    assert s["state"] == "SPIKE" and s["persistence"] == "transient"


def test_sustained_spike():
    x = _noise(0.1).copy(); x[60] += 12.0; x[61:] += 5.0   # spike then stays elevated
    s = P.classify_state(x)
    assert s["state"] in ("SPIKE", "LEVEL_SHIFT", "RISE")   # a spike that steps up is shift-like


def test_cessation():
    # active first half, then the machine goes quiet (flat at baseline) for the rest
    x = np.concatenate([1.5 + 0.5 * np.sin(np.arange(N // 2)), np.zeros(N // 2)])
    s = P.classify_state(x)
    assert s["state"] == "CESSATION"


def test_oscillation():
    s = P.classify_state(3.0 * np.sin(np.arange(N) * 0.8) + _noise(0.1))
    assert s["state"] == "OSCILLATION"


# ---- GENERIC by default: arbitrary signals, any names/count, no domain assumptions ----
def test_generic_default_is_per_channel():
    frame = {"sensor_A": np.linspace(0, 6, N), "flow_rate": _noise(),
             "pressure": 3.0 * np.sin(np.arange(N) * 0.8)}
    d = P.describe_series(frame)                       # no groups, no rules
    assert set(d["groups"]) == {"sensor_A", "flow_rate", "pressure"}   # each its own group
    pg = d["phases"][0]["per_group"]
    assert pg["sensor_A"]["state"] == "RISE"
    assert pg["flow_rate"]["state"] == "STABLE"
    assert pg["pressure"]["state"] == "OSCILLATION"


def test_single_channel():
    d = P.describe_series({"x": np.linspace(0, 6, N)})
    assert list(d["groups"]) == ["x"] and d["phases"][0]["per_group"]["x"]["state"] == "RISE"


# ---- OPTIONAL grouping: explicit map, or a name-rule preset (e.g. SenTSR vibration/temp) ----
def test_explicit_grouping():
    ramp = np.linspace(0, 6, N)
    frame = {"Acceleration": ramp + _noise(seed=1), "Velocity": ramp + _noise(seed=2),
             "Temperature": _noise(seed=3)}
    d = P.describe_series(frame, groups={"vibration": ["Acceleration", "Velocity"],
                                         "temperature": ["Temperature"]})
    pg = d["phases"][0]["per_group"]
    assert pg["vibration"]["state"] == "RISE" and pg["temperature"]["state"] == "STABLE"


def test_preset_grouping_by_name():
    ramp = np.linspace(0, 6, N)
    frame = {"Acceleration": ramp + _noise(seed=1), "Velocity": ramp + _noise(seed=2),
             "Temperature": _noise(seed=3)}
    d = P.describe_series(frame, group_rules="vibration_temperature")
    assert set(d["groups"]) == {"vibration", "temperature"}


# ---- P2: bivariate relations (generic over any group pair) ----
def _rel(frame, groups):
    d = P.describe_series(frame, groups=groups, segment_phases=False)
    return d["phases"][0]["relations"][0]


def test_relation_decoupled():
    ramp = np.linspace(0, 6, N)
    r = _rel({"a": ramp + _noise(seed=1), "b": _noise(seed=2)}, {"a": ["a"], "b": ["b"]})
    assert r["type"] == "DECOUPLED"


def test_relation_comove():
    ramp = np.linspace(0, 6, N)
    r = _rel({"a": ramp + _noise(seed=1), "b": ramp + _noise(seed=2)}, {"a": ["a"], "b": ["b"]})
    assert r["type"] == "CO_MOVE" and r["direction"] == "same"


def test_relation_lead_lag():
    # relate_pair directly: a localized bump in `a` precedes the same bump in `b` → a leads b
    t = np.arange(N)
    a = 8.0 * np.exp(-((t - 50) / 4.0) ** 2) + _noise(0.05, 1)
    b = 8.0 * np.exp(-((t - 70) / 4.0) ** 2) + _noise(0.05, 2)
    r = P.relate_pair(a, b, "SPIKE", "SPIKE")          # both active, offset in time
    assert r["type"] == "LEAD_LAG" and r["leader"] == "a" and r["lag"] > 0


# ---- P3: changepoint phases (multi-phase sequence) ----
def test_multiphase_cessation_then_corise():
    # phase 1: both quiet; phase 2: both ramp up together  ("ceases, then ramp up together")
    a = np.concatenate([np.zeros(N // 2), np.linspace(0, 8, N - N // 2)])
    b = np.concatenate([np.zeros(N // 2), np.linspace(0, 8, N - N // 2)])
    d = P.describe_series({"a": a + _noise(0.05, 1), "b": b + _noise(0.05, 2)},
                          groups={"a": ["a"], "b": ["b"]})
    assert len(d["phases"]) >= 2                                  # segmentation found the transition
    last = d["phases"][-1]
    assert last["per_group"]["a"]["state"] == "RISE"
    assert any(r["type"] == "CO_MOVE" for r in last["relations"])


def test_clean_ramp_stays_single_phase():
    d = P.describe_series({"x": np.linspace(0, 6, N) + _noise(0.05)})
    assert len(d["phases"]) == 1                                  # gradual drift → no spurious split


def test_summary_is_fault_free():
    """Discipline: the description names shapes, never faults."""
    ramp = np.linspace(0, 6, N)
    d = P.describe_series({"Acceleration": ramp, "Velocity": ramp, "Temperature": _noise()})
    banned = ["alignment", "bearing", "lubric", "wear", "imbalance", "friction",
              "fault", "looseness", "gear", "fail"]
    assert not any(b in d["summary"].lower() for b in banned), d["summary"]
