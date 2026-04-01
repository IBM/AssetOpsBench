"""Tests for vibration MCP server tools.

Unit tests use synthetic signals; integration tests require a live CouchDB
(skipped unless COUCHDB_URL is set).
"""

import numpy as np
import pytest

from servers.vibration.data_store import store
from servers.vibration.main import mcp
from .conftest import call_tool, requires_couchdb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(freq_hz: float = 50.0, sr: float = 2048.0,
               duration: float = 1.0, amplitude: float = 1.0) -> tuple:
    """Generate a pure sine wave and store it; return (data_id, signal, sr)."""
    t = np.arange(0, duration, 1.0 / sr)
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t)
    data_id = f"test_sine_{freq_hz}Hz"
    store.put(data_id, sig, sr, {"test": True})
    return data_id, sig, sr


def _make_composite(freqs: list[float], sr: float = 4096.0,
                    duration: float = 2.0) -> str:
    """Composite signal with multiple sine components; returns data_id."""
    t = np.arange(0, duration, 1.0 / sr)
    sig = np.zeros_like(t)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t)
    data_id = "test_composite"
    store.put(data_id, sig, sr, {"freqs": freqs})
    return data_id


# ---------------------------------------------------------------------------
# compute_fft_spectrum
# ---------------------------------------------------------------------------


class TestComputeFFTSpectrum:
    @pytest.mark.anyio
    async def test_basic_50hz(self):
        data_id, _, _ = _make_sine(50.0)
        result = await call_tool(mcp, "compute_fft_spectrum", {"data_id": data_id})
        assert "error" not in result
        assert result["sample_rate_hz"] == 2048.0
        # Dominant peak should be near 50 Hz
        top = result["top_peaks"][0]
        assert abs(top["freq_hz"] - 50.0) < 3.0

    @pytest.mark.anyio
    async def test_missing_data_id(self):
        result = await call_tool(mcp, "compute_fft_spectrum",
                                 {"data_id": "nonexistent"})
        assert "error" in result

    @pytest.mark.anyio
    async def test_window_types(self):
        data_id, _, _ = _make_sine(100.0)
        for win in ("hann", "hamming", "blackman", "rectangular"):
            result = await call_tool(mcp, "compute_fft_spectrum",
                                     {"data_id": data_id, "window": win})
            assert "error" not in result
            assert result["window"] == win


# ---------------------------------------------------------------------------
# compute_envelope_spectrum
# ---------------------------------------------------------------------------


class TestComputeEnvelopeSpectrum:
    @pytest.mark.anyio
    async def test_basic_run(self):
        data_id, _, _ = _make_sine(120.0, sr=4096.0)
        result = await call_tool(mcp, "compute_envelope_spectrum",
                                 {"data_id": data_id})
        assert "error" not in result
        assert "filter_band_hz" in result
        assert result["sample_rate_hz"] == 4096.0

    @pytest.mark.anyio
    async def test_missing_data_id(self):
        result = await call_tool(mcp, "compute_envelope_spectrum",
                                 {"data_id": "nope"})
        assert "error" in result


# ---------------------------------------------------------------------------
# assess_vibration_severity (ISO 10816)
# ---------------------------------------------------------------------------


class TestAssessVibrationSeverity:
    @pytest.mark.anyio
    async def test_zone_a(self):
        result = await call_tool(mcp, "assess_vibration_severity",
                                 {"rms_velocity_mm_s": 0.5})
        assert result["iso_zone"] == "A"

    @pytest.mark.anyio
    async def test_zone_d(self):
        result = await call_tool(mcp, "assess_vibration_severity",
                                 {"rms_velocity_mm_s": 50.0})
        assert result["iso_zone"] == "D"

    @pytest.mark.anyio
    async def test_group_param(self):
        for grp in ("group1", "group2", "group3", "group4"):
            result = await call_tool(mcp, "assess_vibration_severity",
                                     {"rms_velocity_mm_s": 4.5,
                                      "machine_group": grp})
            assert result["iso_zone"] in ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# list_known_bearings
# ---------------------------------------------------------------------------


class TestListKnownBearings:
    @pytest.mark.anyio
    async def test_returns_bearings(self):
        result = await call_tool(mcp, "list_known_bearings", {})
        assert "bearings" in result
        assert len(result["bearings"]) >= 5
        names = [b["name"] for b in result["bearings"]]
        assert any("6205" in n for n in names)


# ---------------------------------------------------------------------------
# calculate_bearing_frequencies
# ---------------------------------------------------------------------------


class TestCalculateBearingFrequencies:
    @pytest.mark.anyio
    async def test_basic(self):
        result = await call_tool(mcp, "calculate_bearing_frequencies", {
            "rpm": 1800,
            "n_balls": 9,
            "ball_diameter_mm": 7.94,
            "pitch_diameter_mm": 39.04,
            "contact_angle_deg": 0.0,
        })
        assert "bpfo_hz" in result
        assert "bpfi_hz" in result
        assert "bsf_hz" in result
        assert "ftf_hz" in result
        assert result["bpfo_hz"] > 0

    @pytest.mark.anyio
    async def test_with_name(self):
        result = await call_tool(mcp, "calculate_bearing_frequencies", {
            "rpm": 3600,
            "n_balls": 8,
            "ball_diameter_mm": 10.0,
            "pitch_diameter_mm": 46.0,
            "bearing_name": "test-bearing",
        })
        assert "bearing" in result
        assert result["bearing"] == "test-bearing"


# ---------------------------------------------------------------------------
# diagnose_vibration
# ---------------------------------------------------------------------------


class TestDiagnoseVibration:
    @pytest.mark.anyio
    async def test_no_rpm(self):
        """Without RPM we expect a partial result with a warning."""
        data_id, _, _ = _make_sine(120.0, sr=4096.0, duration=2.0)
        result = await call_tool(mcp, "diagnose_vibration", {
            "data_id": data_id,
        })
        assert "error" not in result
        assert "warning" in result
        assert result["shaft_features"] is None

    @pytest.mark.anyio
    async def test_with_rpm(self):
        data_id = _make_composite([30, 60, 90], sr=4096.0, duration=2.0)
        result = await call_tool(mcp, "diagnose_vibration", {
            "data_id": data_id,
            "rpm": 1800.0,
        })
        assert "error" not in result
        assert result["shaft_features"] is not None
        assert result["iso_10816"] is not None
        assert "report_markdown" in result

    @pytest.mark.anyio
    async def test_with_bearing_designation(self):
        data_id = _make_composite([30, 60, 120], sr=4096.0, duration=2.0)
        result = await call_tool(mcp, "diagnose_vibration", {
            "data_id": data_id,
            "rpm": 1800.0,
            "bearing_designation": "6205",
        })
        assert "error" not in result
        assert result["bearing_info_source"] is not None
        assert "database" in result["bearing_info_source"]

    @pytest.mark.anyio
    async def test_with_custom_bearing_geometry(self):
        data_id = _make_composite([30, 60], sr=4096.0, duration=2.0)
        result = await call_tool(mcp, "diagnose_vibration", {
            "data_id": data_id,
            "rpm": 1800.0,
            "bearing_n_balls": 9,
            "bearing_ball_dia_mm": 7.94,
            "bearing_pitch_dia_mm": 39.04,
        })
        assert "error" not in result
        assert result["bearing_info_source"] == "custom geometry"

    @pytest.mark.anyio
    async def test_missing_data_id(self):
        result = await call_tool(mcp, "diagnose_vibration",
                                 {"data_id": "ghost"})
        assert "error" in result


# ---------------------------------------------------------------------------
# get_vibration_data (integration — requires CouchDB)
# ---------------------------------------------------------------------------


class TestGetVibrationData:
    @requires_couchdb
    @pytest.mark.anyio
    async def test_fetch_integration(self):
        result = await call_tool(mcp, "get_vibration_data", {
            "site_name": "MAIN",
            "asset_id": "Motor_01",
            "sensor_name": "Vibration_X",
            "start": "2024-01-15T00:00:00",
        })
        assert "error" not in result
        assert "data_id" in result


# ---------------------------------------------------------------------------
# list_vibration_sensors (integration — requires CouchDB)
# ---------------------------------------------------------------------------


class TestListVibrationSensors:
    @requires_couchdb
    @pytest.mark.anyio
    async def test_list_integration(self):
        result = await call_tool(mcp, "list_vibration_sensors", {
            "site_name": "MAIN",
            "asset_id": "Chiller 6",
        })
        assert "sensors" in result or "error" in result
