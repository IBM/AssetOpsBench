# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp
"""
FFT and spectral analysis module.

Provides functions for frequency-domain analysis of vibration signals:
- FFT magnitude spectrum
- Power Spectral Density (Welch method)
- Spectrogram (STFT)
- Peak detection
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig
from typing import Optional


def compute_fft(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    n_fft: Optional[int] = None,
) -> dict:
    """
    Compute single-sided FFT magnitude spectrum.

    Args:
        data: 1D time-domain signal
        fs: Sampling frequency in Hz
        window: Window function ('hann', 'hamming', 'blackman', 'rectangular')
        n_fft: FFT length (defaults to len(data), zero-padded if > len(data))

    Returns:
        Dictionary with 'frequencies' (Hz), 'magnitude' (linear),
        'magnitude_db' (dB), and 'resolution_hz'
    """
    n = len(data)
    if n_fft is None:
        n_fft = n

    # Apply window
    if window == "rectangular":
        w = np.ones(n)
    else:
        w = sig.get_window(window, n)

    windowed = data * w

    # Compute FFT
    fft_vals = np.fft.rfft(windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # Magnitude (single-sided, compensated for window energy loss)
    magnitude = (2.0 / n) * np.abs(fft_vals)
    magnitude[0] /= 2.0  # DC component not doubled

    # Convert to dB (reference: 1.0)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    return {
        "frequencies": freqs,
        "magnitude": magnitude,
        "magnitude_db": magnitude_db,
        "resolution_hz": fs / n_fft,
        "max_frequency_hz": fs / 2,
        "num_points": len(freqs),
    }


def compute_psd(
    data: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> dict:
    """
    Compute Power Spectral Density using Welch's method.

    Args:
        data: 1D time-domain signal
        fs: Sampling frequency in Hz
        nperseg: Segment length for Welch method
        noverlap: Overlap between segments (default: nperseg // 2)
        window: Window function

    Returns:
        Dictionary with 'frequencies' (Hz) and 'psd'
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = sig.welch(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
    )

    return {
        "frequencies": freqs,
        "psd": psd,
        "total_power": float(np.trapezoid(psd, freqs)),
        "resolution_hz": freqs[1] - freqs[0] if len(freqs) > 1 else 0,
    }


def compute_spectrogram(
    data: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> dict:
    """
    Compute spectrogram (Short-Time Fourier Transform).

    Args:
        data: 1D time-domain signal
        fs: Sampling frequency in Hz
        nperseg: Segment length
        noverlap: Overlap (default: nperseg // 2)
        window: Window function

    Returns:
        Dictionary with 'frequencies', 'times', 'spectrogram_db'
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = sig.spectrogram(
        data, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    return {
        "frequencies": freqs,
        "times": times,
        "spectrogram_db": Sxx_db,
        "num_time_frames": len(times),
        "num_freq_bins": len(freqs),
    }


def find_peaks_in_spectrum(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    num_peaks: int = 10,
    min_distance_hz: float = 5.0,
    threshold_db: float = -60.0,
) -> list[dict]:
    """
    Find dominant peaks in a frequency spectrum.

    Args:
        frequencies: Frequency axis in Hz
        magnitude: Magnitude spectrum (linear)
        num_peaks: Maximum number of peaks to return
        min_distance_hz: Minimum distance between peaks in Hz
        threshold_db: Minimum amplitude threshold in dB

    Returns:
        List of dicts with 'frequency_hz', 'magnitude', 'magnitude_db'
    """
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    # Convert min_distance to samples
    df = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    min_distance_samples = max(1, int(min_distance_hz / df))

    # Find peaks
    peak_indices, _properties = sig.find_peaks(
        magnitude_db,
        distance=min_distance_samples,
        height=threshold_db,
    )

    if len(peak_indices) == 0:
        return []

    # Sort by amplitude (descending)
    sorted_idx = np.argsort(magnitude_db[peak_indices])[::-1]
    top_peaks = peak_indices[sorted_idx[:num_peaks]]

    peaks = []
    for idx in top_peaks:
        peaks.append(
            {
                "frequency_hz": float(frequencies[idx]),
                "magnitude": float(magnitude[idx]),
                "magnitude_db": float(magnitude_db[idx]),
            }
        )

    return peaks
