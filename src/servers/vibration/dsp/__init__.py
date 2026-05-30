# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp

from .fft_analysis import compute_fft, compute_psd, compute_spectrogram, find_peaks_in_spectrum
from .envelope import envelope_spectrum, check_bearing_peaks
from .bearing_freqs import compute_bearing_frequencies, get_bearing, list_bearings, COMMON_BEARINGS
from .fault_detection import (
    assess_iso10816,
    extract_shaft_features,
    classify_faults,
    generate_diagnosis_summary,
)

__all__ = [
    "compute_fft",
    "compute_psd",
    "compute_spectrogram",
    "find_peaks_in_spectrum",
    "envelope_spectrum",
    "check_bearing_peaks",
    "compute_bearing_frequencies",
    "get_bearing",
    "list_bearings",
    "COMMON_BEARINGS",
    "assess_iso10816",
    "extract_shaft_features",
    "classify_faults",
    "generate_diagnosis_summary",
]
