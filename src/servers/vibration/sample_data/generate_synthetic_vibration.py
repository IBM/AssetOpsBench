#!/usr/bin/env python3
"""Generate synthetic vibration data for a bearing outer-race fault.

Model
-----
Simplified analytical simulation based on the McFadden & Smith [1] impulsive
model.  Each ball pass over the outer-race defect excites a structural
resonance that decays exponentially (ring-down).  Impulse amplitudes are
modulated at shaft frequency to approximate load-zone effects.

Simplifications / known limitations
------------------------------------
* Motor slip is neglected (ω_shaft = RPM / 60 exactly).  For realistic
  induction motor simulations, replace f_shaft with f_shaft × (1 − s).
* The resonance is a single mode; real structures show multiple modes.
* Noise is Gaussian; real industrial noise may contain tonal components
  from gearboxes, electrical line frequency, etc.

References
----------
[1] McFadden, P. D. & Smith, J. D. (1984).  "Model for the vibration
    produced by a single point defect in a rolling element bearing."
    Journal of Sound and Vibration, 96(1), 69-82.

Output
------
bulk_docs_vibration.json — JSON array of CouchDB documents with fields:
  asset_id, timestamp (ISO-8601 µs), Vibration_X (g)

Usage
-----
    python generate_synthetic_vibration.py          # writes JSON to cwd
    python generate_synthetic_vibration.py --check  # writes JSON + prints stats
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Machine / bearing parameters
# ---------------------------------------------------------------------------
FS = 4096           # sampling rate [Hz]
DURATION = 1.0      # seconds
RPM = 1800          # shaft speed
F_SHAFT = RPM / 60  # shaft frequency [Hz]

# SKF 6205-2RS   (common small motor bearing)
N_BALLS = 9
BD = 7.94           # ball diameter [mm]
PD = 39.04          # pitch diameter [mm]
ALPHA = 0.0         # contact angle [rad]

# Derived characteristic frequencies
BPFO = N_BALLS / 2 * F_SHAFT * (1 - BD / PD * np.cos(ALPHA))  # ~107.5 Hz

# Resonance and damping
F_RESONANCE = 3200.0   # structural resonance [Hz]
DAMPING = 5000.0       # exponential decay rate [1/s]  (fast → sharp impulses)
IMPULSE_AMP = 2.0      # peak impulse amplitude [g]
LOAD_MOD = 0.5         # load-zone modulation depth (0 = none, 1 = full)

# Background
SHAFT_1X = 0.10        # 1× shaft amplitude [g]
SHAFT_2X = 0.04        # 2× shaft amplitude [g]
NOISE_STD = 0.02       # broadband noise σ [g]

# Time origin (arbitrary)
T0 = datetime(2024, 1, 15, 0, 0, 0)

SEED = 42


def generate() -> tuple[np.ndarray, np.ndarray]:
    """Return (time_vector, acceleration_signal)."""
    rng = np.random.default_rng(SEED)
    n_samples = int(FS * DURATION)
    t = np.arange(n_samples) / FS

    # Shaft harmonics (healthy background)
    shaft = SHAFT_1X * np.sin(2 * np.pi * F_SHAFT * t) + \
            SHAFT_2X * np.sin(2 * np.pi * 2 * F_SHAFT * t)

    # Bearing fault impulses at BPFO
    impulse_times = np.arange(0, DURATION, 1.0 / BPFO)
    bearing = np.zeros_like(t)
    for t_imp in impulse_times:
        dt = t - t_imp
        mask = dt >= 0
        # Load-zone amplitude modulation
        amp = 1.0 + LOAD_MOD * np.cos(2 * np.pi * F_SHAFT * t_imp)
        ring = amp * IMPULSE_AMP * np.exp(-DAMPING * dt[mask]) * \
               np.sin(2 * np.pi * F_RESONANCE * dt[mask])
        bearing[mask] += ring

    noise = NOISE_STD * rng.standard_normal(n_samples)
    return t, shaft + bearing + noise


def to_couchdb_docs(t: np.ndarray, signal: np.ndarray) -> list[dict]:
    dt_step = timedelta(seconds=1.0 / FS)
    return [
        {
            "asset_id": "Motor_01",
            "timestamp": (T0 + i * dt_step).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "Vibration_X": round(float(signal[i]), 6),
        }
        for i in range(len(t))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Print signal statistics after generation")
    args = parser.parse_args()

    t, signal = generate()
    docs = to_couchdb_docs(t, signal)

    out = os.path.join(os.path.dirname(__file__), "bulk_docs_vibration.json")
    with open(out, "w") as f:
        json.dump(docs, f, indent=2)

    print(f"Wrote {len(docs)} documents to {out}")

    if args.check:
        rms = float(np.sqrt(np.mean(signal ** 2)))
        peak = float(np.max(np.abs(signal)))
        # Excess kurtosis with sample std (ddof=1), consistent with main.py
        kurt = float(np.mean((signal - signal.mean()) ** 4) /
                      np.std(signal, ddof=1) ** 4 - 3)
        print(f"  BPFO:          {BPFO:.2f} Hz")
        print(f"  f_shaft:       {F_SHAFT:.1f} Hz")
        print(f"  f_resonance:   {F_RESONANCE:.1f} Hz")
        print(f"  RMS:           {rms:.4f} g")
        print(f"  Peak:          {peak:.4f} g")
        print(f"  Crest factor:  {peak / rms:.2f}")
        print(f"  Kurtosis (excess): {kurt:.2f}")


if __name__ == "__main__":
    main()
