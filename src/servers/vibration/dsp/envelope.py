# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp
"""
Envelope analysis module for bearing fault detection.

Envelope analysis (demodulation) uses the Hilbert transform to extract
the amplitude envelope of a band-pass filtered vibration signal.
This reveals the repetition rate of impulsive events (e.g., a defect
on a bearing race) even when the individual impacts are buried in noise.

Typical workflow:
    1. Band-pass filter the signal around a structural resonance
    2. Compute the analytic signal via the Hilbert transform
    3. Extract the envelope (magnitude of the analytic signal)
    4. Compute the FFT of the envelope -> the envelope spectrum
    5. Look for peaks at bearing fault frequencies (BPFO, BPFI, BSF, FTF)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, hilbert


def bandpass_filter(
    signal: NDArray[np.floating],
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> NDArray[np.floating]:
    """
    Apply a Butterworth band-pass filter.

    Args:
        signal: Input time-domain signal.
        fs: Sampling frequency in Hz.
        low_hz: Lower cutoff frequency.
        high_hz: Upper cutoff frequency.
        order: Filter order (default 4).

    Returns:
        Filtered signal.
    """
    nyq = fs / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)  # stay below Nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal).astype(signal.dtype)


def compute_envelope(
    signal: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute the amplitude envelope using the Hilbert transform.

    Args:
        signal: Input time-domain signal (ideally band-pass filtered).

    Returns:
        Envelope signal (same length as input).
    """
    analytic = hilbert(signal)
    return np.abs(analytic).astype(signal.dtype)


def envelope_spectrum(
    signal: NDArray[np.floating],
    fs: float,
    band_low: float | None = None,
    band_high: float | None = None,
    filter_order: int = 4,
    n_fft: int | None = None,
) -> dict:
    """
    Full envelope analysis pipeline: band-pass -> envelope -> FFT.

    Args:
        signal: Raw vibration time-domain signal.
        fs: Sampling frequency in Hz.
        band_low: Band-pass lower cutoff (Hz). If None, defaults to fs/20.
        band_high: Band-pass upper cutoff (Hz). If None, defaults to fs/2.5.
        filter_order: Butterworth filter order.
        n_fft: FFT length. Default = signal length.

    Returns:
        dict with keys: frequencies, envelope_spectrum, filter_band, n_samples, fs
    """
    n = len(signal)
    if band_low is None:
        band_low = fs / 20.0
    if band_high is None:
        band_high = fs / 2.5
    if n_fft is None:
        n_fft = n

    # Step 1: Band-pass filter
    filtered = bandpass_filter(signal, fs, band_low, band_high, order=filter_order)

    # Step 2: Hilbert envelope
    env = compute_envelope(filtered)

    # Remove DC from envelope before FFT
    env_zero_mean = env - np.mean(env)

    # Step 3: FFT of envelope
    window = np.hanning(n)
    fft_vals = np.fft.rfft(env_zero_mean * window, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    magnitudes = (2.0 / n) * np.abs(fft_vals)

    return {
        "frequencies": freqs.tolist(),
        "envelope_spectrum": magnitudes.tolist(),
        "filter_band": (band_low, band_high),
        "n_samples": n,
        "fs": fs,
    }


def check_bearing_peaks(
    freqs: NDArray[np.floating] | list[float],
    magnitudes: NDArray[np.floating] | list[float],
    target_freq: float,
    n_harmonics: int = 3,
    tolerance_pct: float = 3.0,
    noise_floor_multiplier: float = 3.0,
) -> dict:
    """
    Check whether a target bearing frequency (and its harmonics) is present
    in an envelope spectrum.

    Args:
        freqs: Frequency axis (Hz).
        magnitudes: Spectrum magnitudes.
        target_freq: Expected fault frequency (e.g., BPFO) in Hz.
        n_harmonics: Number of harmonics to check (1x, 2x, ... Nx).
        tolerance_pct: Frequency matching tolerance in % of target.
        noise_floor_multiplier: Peak must exceed median * this to be significant.

    Returns:
        dict with detection results per harmonic and overall verdict.
    """
    freqs = np.asarray(freqs)
    mags = np.asarray(magnitudes)

    noise_floor = np.median(mags) * noise_floor_multiplier

    results = []
    detected_count = 0

    for h in range(1, n_harmonics + 1):
        f_expected = h * target_freq
        tol = f_expected * tolerance_pct / 100.0
        mask = (freqs >= f_expected - tol) & (freqs <= f_expected + tol)

        if np.any(mask):
            peak_mag = float(np.max(mags[mask]))
            peak_freq = float(freqs[mask][np.argmax(mags[mask])])
            is_detected = bool(peak_mag > noise_floor)
            if is_detected:
                detected_count += 1
            results.append(
                {
                    "harmonic": h,
                    "expected_hz": round(f_expected, 2),
                    "found_hz": round(peak_freq, 2),
                    "amplitude": round(peak_mag, 6),
                    "noise_threshold": round(float(noise_floor), 6),
                    "detected": is_detected,
                }
            )
        else:
            results.append(
                {
                    "harmonic": h,
                    "expected_hz": round(f_expected, 2),
                    "found_hz": None,
                    "amplitude": 0.0,
                    "noise_threshold": round(float(noise_floor), 6),
                    "detected": False,
                }
            )

    return {
        "target_frequency_hz": target_freq,
        "harmonics_checked": n_harmonics,
        "harmonics_detected": detected_count,
        "confidence": (
            "high" if detected_count >= 2 else ("medium" if detected_count == 1 else "none")
        ),
        "details": results,
    }
