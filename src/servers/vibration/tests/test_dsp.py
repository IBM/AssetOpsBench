"""Pure-function unit tests for the DSP layer — no MCP, no CouchDB."""

import math

import numpy as np
import pytest

from servers.vibration.dsp.fft_analysis import (
    compute_fft,
    compute_psd,
    compute_spectrogram,
    find_peaks_in_spectrum,
)
from servers.vibration.dsp.envelope import (
    bandpass_filter,
    compute_envelope,
    envelope_spectrum,
    check_bearing_peaks,
)
from servers.vibration.dsp.bearing_freqs import (
    compute_bearing_frequencies,
    get_bearing,
    list_bearings,
)
from servers.vibration.dsp.fault_detection import (
    assess_iso10816,
    extract_shaft_features,
    classify_faults,
    generate_diagnosis_summary,
)


# ---------------------------------------------------------------------------
# Synthetic signals
# ---------------------------------------------------------------------------

SR = 4096.0
DURATION = 2.0
T = np.arange(0, DURATION, 1.0 / SR)
SINE_50 = np.sin(2 * np.pi * 50 * T)
SINE_120 = np.sin(2 * np.pi * 120 * T)
COMPOSITE = SINE_50 + 0.5 * SINE_120


# ===================================================================
# fft_analysis
# ===================================================================


class TestComputeFFT:
    def test_pure_sine(self):
        result = compute_fft(SINE_50, SR)
        freqs = np.array(result["frequencies"])
        mags = np.array(result["magnitude"])
        peak_idx = np.argmax(mags)
        assert abs(freqs[peak_idx] - 50.0) < 2.0

    def test_window_types(self):
        for win in ("hann", "hamming", "blackman", "rectangular"):
            result = compute_fft(SINE_50, SR, window=win)
            assert len(result["frequencies"]) == len(result["magnitude"])

    def test_frequency_resolution(self):
        result = compute_fft(SINE_50, SR)
        freqs = np.array(result["frequencies"])
        expected_res = SR / len(SINE_50)
        assert abs(freqs[1] - expected_res) < 0.01


class TestComputePSD:
    def test_basic(self):
        result = compute_psd(SINE_50, SR)
        assert "frequencies" in result
        assert "psd" in result
        assert len(result["frequencies"]) == len(result["psd"])


class TestComputeSpectrogram:
    def test_basic(self):
        result = compute_spectrogram(SINE_50, SR)
        assert "frequencies" in result
        assert "times" in result
        assert "spectrogram_db" in result


class TestFindPeaks:
    def test_find_50hz_peak(self):
        fft = compute_fft(SINE_50, SR)
        freqs = np.array(fft["frequencies"])
        mags = np.array(fft["magnitude"])
        peaks = find_peaks_in_spectrum(freqs, mags, num_peaks=5)
        top_freq = peaks[0]["frequency_hz"]
        assert abs(top_freq - 50.0) < 2.0


# ===================================================================
# envelope
# ===================================================================


class TestBandpassFilter:
    def test_shape_preserved(self):
        filtered = bandpass_filter(SINE_50, SR, 30.0, 70.0)
        assert len(filtered) == len(SINE_50)


class TestComputeEnvelope:
    def test_shape_and_non_negative(self):
        env = compute_envelope(SINE_50)
        assert len(env) == len(SINE_50)
        assert np.all(env >= 0)


class TestEnvelopeSpectrum:
    def test_basic(self):
        result = envelope_spectrum(SINE_120, SR)
        assert "frequencies" in result
        assert "envelope_spectrum" in result
        assert "filter_band" in result

    def test_custom_band(self):
        result = envelope_spectrum(SINE_120, SR, band_low=50.0, band_high=200.0)
        lo, hi = result["filter_band"]
        assert lo == 50.0
        assert hi == 200.0


class TestCheckBearingPeaks:
    def test_finds_peak(self):
        # Signal with a clear 100 Hz component
        sig = np.sin(2 * np.pi * 100 * T)
        env = envelope_spectrum(sig, SR)
        freqs = np.array(env["frequencies"])
        mags = np.array(env["envelope_spectrum"])
        result = check_bearing_peaks(freqs, mags, 100.0)
        assert "details" in result
        assert result["harmonics_detected"] >= 1


# ===================================================================
# bearing_freqs
# ===================================================================


class TestComputeBearingFrequencies:
    def test_6205_at_1800rpm(self):
        bf = compute_bearing_frequencies(1800, 9, 7.94, 39.04, 0.0, "6205")
        assert bf.bpfo > 0
        assert bf.bpfi > bf.bpfo  # inner race always higher
        assert bf.bsf > 0
        assert bf.ftf > 0
        assert bf.ftf < bf.bpfo  # cage freq is lowest

    def test_zero_rpm(self):
        bf = compute_bearing_frequencies(0, 9, 7.94, 39.04, 0.0, "test")
        assert bf.bpfo == 0.0
        assert bf.bpfi == 0.0

    def test_to_dict(self):
        bf = compute_bearing_frequencies(3600, 9, 7.94, 39.04, 0.0, "test")
        d = bf.to_dict()
        for key in ("bpfo_hz", "bpfi_hz", "bsf_hz", "ftf_hz", "rpm", "bearing"):
            assert key in d


class TestGetBearing:
    def test_known_bearing(self):
        b = get_bearing("6205")
        assert b is not None
        assert b.n_balls == 9

    def test_unknown_bearing(self):
        b = get_bearing("NONEXISTENT-999")
        assert b is None


class TestListBearings:
    def test_returns_list(self):
        bearings = list_bearings()
        assert len(bearings) >= 5
        assert all("name" in b for b in bearings)


# ===================================================================
# fault_detection
# ===================================================================


class TestAssessISO10816:
    def test_zone_a(self):
        result = assess_iso10816(0.5, "group2")
        assert result["iso_zone"] == "A"

    def test_zone_b(self):
        result = assess_iso10816(2.0, "group2")
        assert result["iso_zone"] == "B"

    def test_zone_c(self):
        result = assess_iso10816(6.0, "group2")
        assert result["iso_zone"] == "C"

    def test_zone_d(self):
        result = assess_iso10816(15.0, "group2")
        assert result["iso_zone"] == "D"

    def test_all_groups(self):
        for grp in ("group1", "group2", "group3", "group4"):
            r = assess_iso10816(4.5, grp)
            assert r["iso_zone"] in ("A", "B", "C", "D")

    def test_unknown_group_defaults(self):
        r = assess_iso10816(4.5, "group_INVALID")
        assert r["iso_zone"] in ("A", "B", "C", "D")


class TestExtractShaftFeatures:
    def test_basic(self):
        fft = compute_fft(COMPOSITE, SR)
        freqs = np.array(fft["frequencies"])
        mags = np.array(fft["magnitude"])
        shaft_freq = 50.0  # as if rpm=3000
        features = extract_shaft_features(freqs, mags, shaft_freq,
                                          time_signal=COMPOSITE)
        assert features.f_shaft == 50.0
        assert features.amp_1x > 0


class TestClassifyFaults:
    def test_empty_features_returns_healthy(self):
        # All amplitudes zero → healthy or very low vibration
        fft = compute_fft(np.zeros(1000), SR)
        freqs = np.array(fft["frequencies"])
        mags = np.array(fft["magnitude"])
        features = extract_shaft_features(freqs, mags, 30.0)
        diagnoses = classify_faults(features)
        # Should not crash; may return healthy or empty
        assert isinstance(diagnoses, list)


class TestGenerateDiagnosisSummary:
    def test_renders_markdown(self):
        iso = assess_iso10816(4.5, "group2")
        md = generate_diagnosis_summary([], iso, "Test pump")
        assert "ISO 10816" in md or "10816" in md
        assert isinstance(md, str)
