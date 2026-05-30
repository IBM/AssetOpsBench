"""vibration MCP Server.

Vibration signal analysis and rotating machinery fault detection
for AssetOpsBench.  Reads sensor data from CouchDB, performs FFT,
envelope analysis, bearing fault detection, and ISO 10816 assessment.

DSP core adapted from https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from .couchdb_client import fetch_vibration_timeseries, list_sensor_fields
from .data_store import store
from .dsp.bearing_freqs import (
    COMMON_BEARINGS,
    compute_bearing_frequencies,
    get_bearing,
    list_bearings,
)
from .dsp.envelope import check_bearing_peaks, envelope_spectrum
from .dsp.fault_detection import (
    assess_iso10816,
    classify_faults,
    extract_shaft_features,
    generate_diagnosis_summary,
)
from .dsp.fft_analysis import compute_fft

load_dotenv()
_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("vibration-mcp-server")

mcp = FastMCP("vibration", instructions="Vibration signal analysis: FFT, envelope spectrum, bearing fault detection, and ISO 10816 severity assessment.")


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class ErrorResult(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compact_spectrum(freqs: np.ndarray, mags: np.ndarray, top_n: int = 20) -> dict:
    """Summarise a spectrum: top-N peaks + global stats.  No full arrays."""
    order = np.argsort(mags)[::-1]
    n = min(top_n, len(order))
    top_idx = np.sort(order[:n])  # sort back by frequency
    peaks = [
        {"freq_hz": round(float(freqs[i]), 3), "amplitude": round(float(mags[i]), 6)}
        for i in top_idx
    ]
    return {
        "top_peaks": peaks,
        "max_amplitude": round(float(np.max(mags)), 6),
        "max_amplitude_freq_hz": round(float(freqs[np.argmax(mags)]), 3),
        "rms_spectral": round(float(np.sqrt(np.mean(mags**2))), 6),
        "total_bins": len(freqs),
        "freq_range_hz": [round(float(freqs[0]), 3), round(float(freqs[-1]), 3)],
    }


def _accel_g_to_velocity_rms_mms(
    signal_g: np.ndarray,
    sample_rate: float,
    f_low: float = 10.0,
    f_high: float = 1000.0,
) -> float:
    """
    Convert an acceleration signal (in g) to RMS velocity (in mm/s)
    within the ISO 10816 band (default 10-1000 Hz).

    Method: frequency-domain integration.
    """
    N = len(signal_g)
    if N < 2:
        return 0.0

    accel_ms2 = signal_g * 9.80665
    fft_vals = np.fft.rfft(accel_ms2)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    # Vectorised integration: v(f) = a(f) / (j·2π·f), skip DC bin
    vel_fft = np.zeros_like(fft_vals)
    mask = (freqs >= f_low) & (freqs <= f_high)
    vel_fft[mask] = fft_vals[mask] / (1j * 2.0 * np.pi * freqs[mask])

    velocity_ms = np.fft.irfft(vel_fft, n=N)
    velocity_mms = velocity_ms * 1000.0
    rms = float(np.sqrt(np.mean(velocity_mms**2)))
    return rms


def _resolve_signal(data_id: str) -> tuple[np.ndarray, float]:
    """Return (1-D numpy signal, sample_rate) from a data_id."""
    entry = store.get(data_id)
    if entry is None:
        available = store.list_ids()
        raise ValueError(
            f"data_id '{data_id}' not found in store. "
            f"Available: {available or '(empty - use get_vibration_data first)'}."
        )
    sig = entry.signal
    if sig.ndim > 1:
        sig = sig[:, 0]  # default to first channel
    return sig, entry.sample_rate


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool(title="Get Vibration Data")
def get_vibration_data(
    site_name: str,
    asset_id: str,
    sensor_name: str,
    start: str,
    final: Optional[str] = None,
) -> Union[dict, ErrorResult]:
    """Fetch vibration sensor time-series from CouchDB and load into the analysis store.

    Returns a data_id that can be passed to analysis tools (compute_fft_spectrum,
    compute_envelope_spectrum, diagnose_vibration, etc.).

    Args:
        site_name: Site identifier (e.g., 'MAIN').
        asset_id: Asset identifier (e.g., 'Chiller 6').
        sensor_name: Name of the sensor field in CouchDB documents.
        start: ISO 8601 start timestamp (e.g., '2020-06-01T00:00:00').
        final: Optional ISO 8601 end timestamp.
    """
    result = fetch_vibration_timeseries(asset_id, sensor_name, start, final)
    if result is None:
        return ErrorResult(
            error=f"No vibration data found for asset '{asset_id}', "
            f"sensor '{sensor_name}' in time range starting {start}."
        )
    signal, sample_rate = result
    data_id = f"{asset_id}_{sensor_name}".replace(" ", "_")
    store.put(
        data_id,
        signal,
        sample_rate,
        {
            "source": "couchdb",
            "asset_id": asset_id,
            "sensor": sensor_name,
            "start": start,
            "final": final,
        },
    )
    entry = store.get(data_id)
    return {"data_id": data_id, **entry.summary()}


@mcp.tool(title="List Vibration Sensors")
def list_vibration_sensors(
    site_name: str,
    asset_id: str,
) -> Union[dict, ErrorResult]:
    """List available sensor fields for an asset.

    Args:
        site_name: Site identifier (e.g., 'MAIN').
        asset_id: Asset identifier (e.g., 'Chiller 6').
    """
    sensors = list_sensor_fields(asset_id)
    if not sensors:
        return ErrorResult(
            error=f"No sensors found for asset '{asset_id}' at site '{site_name}'."
        )
    return {
        "site_name": site_name,
        "asset_id": asset_id,
        "total_sensors": len(sensors),
        "sensors": sensors,
    }


@mcp.tool(title="Compute FFT Spectrum")
def compute_fft_spectrum(
    data_id: str,
    window: str = "hann",
    top_n: int = 20,
) -> Union[dict, ErrorResult]:
    """Compute FFT amplitude spectrum of a stored vibration signal.

    Returns a compact summary (top N peaks + statistics), not the full array.

    Args:
        data_id: Reference to a stored signal (from get_vibration_data).
        window: Window function ('hann', 'hamming', 'blackman', 'rectangular').
        top_n: Number of highest peaks to include in the summary.
    """
    try:
        sig, sr = _resolve_signal(data_id)
    except ValueError as e:
        return ErrorResult(error=str(e))

    result = compute_fft(sig, sr, window=window)
    freqs = np.asarray(result["frequencies"])
    mags = np.asarray(result["magnitude"])
    summary = _compact_spectrum(freqs, mags, top_n=top_n)
    summary.update(
        {
            "data_id": data_id,
            "n_samples": len(sig),
            "sample_rate_hz": sr,
            "window": window,
            "frequency_resolution_hz": round(float(freqs[1]), 4)
            if len(freqs) > 1
            else 0,
        }
    )
    return summary


@mcp.tool(title="Compute Envelope Spectrum")
def compute_envelope_spectrum(
    data_id: str,
    band_low_hz: Optional[float] = None,
    band_high_hz: Optional[float] = None,
    top_n: int = 20,
) -> Union[dict, ErrorResult]:
    """Compute the envelope spectrum for bearing fault detection.

    Args:
        data_id: Reference to a stored signal (from get_vibration_data).
        band_low_hz: Band-pass lower cutoff. Auto if None.
        band_high_hz: Band-pass upper cutoff. Auto if None.
        top_n: Number of highest peaks to include.
    """
    try:
        sig, sr = _resolve_signal(data_id)
    except ValueError as e:
        return ErrorResult(error=str(e))

    result = envelope_spectrum(sig, sr, band_low=band_low_hz, band_high=band_high_hz)
    freqs = np.asarray(result["frequencies"])
    env_mags = np.asarray(result["envelope_spectrum"])
    summary = _compact_spectrum(freqs, env_mags, top_n=top_n)
    summary.update(
        {
            "data_id": data_id,
            "filter_band_hz": list(result["filter_band"]),
            "n_samples": result["n_samples"],
            "sample_rate_hz": sr,
        }
    )
    return summary


@mcp.tool(title="Assess Vibration Severity")
def assess_vibration_severity(
    rms_velocity_mm_s: float,
    machine_group: str = "group2",
) -> dict:
    """Classify vibration severity per ISO 10816.

    IMPORTANT: expects RMS velocity in mm/s (not acceleration in g).
    The diagnose_vibration tool performs the conversion automatically.

    Machine groups:
        group1 - Large machines (>300 kW) on rigid foundations
        group2 - Medium machines (15-300 kW) on rigid foundations
        group3 - Large machines on flexible foundations
        group4 - Small machines (<15 kW)

    Args:
        rms_velocity_mm_s: Overall RMS velocity in mm/s (10-1000 Hz band).
        machine_group: ISO 10816 machine group identifier.
    """
    return assess_iso10816(rms_velocity_mm_s, machine_group)


@mcp.tool(title="Calculate Bearing Frequencies")
def calculate_bearing_frequencies(
    rpm: float,
    n_balls: int,
    ball_diameter_mm: float,
    pitch_diameter_mm: float,
    contact_angle_deg: float = 0.0,
    bearing_name: str = "",
) -> dict:
    """Compute bearing characteristic frequencies (BPFO, BPFI, BSF, FTF) from geometry.

    Args:
        rpm: Shaft speed in revolutions per minute.
        n_balls: Number of rolling elements.
        ball_diameter_mm: Ball or roller diameter in mm.
        pitch_diameter_mm: Pitch (cage) diameter in mm.
        contact_angle_deg: Contact angle in degrees (0 for radial bearings).
        bearing_name: Optional bearing designation for labeling.
    """
    result = compute_bearing_frequencies(
        rpm,
        n_balls,
        ball_diameter_mm,
        pitch_diameter_mm,
        contact_angle_deg,
        bearing_name or "custom",
    )
    return result.to_dict()


@mcp.tool(title="List Known Bearings")
def list_known_bearings() -> dict:
    """List all bearings in the built-in database with their geometric parameters."""
    return {"bearings": list_bearings()}


@mcp.tool(title="Diagnose Vibration")
def diagnose_vibration(
    data_id: str,
    rpm: Optional[float] = None,
    bearing_designation: Optional[str] = None,
    bearing_n_balls: Optional[int] = None,
    bearing_ball_dia_mm: Optional[float] = None,
    bearing_pitch_dia_mm: Optional[float] = None,
    bearing_contact_angle_deg: float = 0.0,
    bpfo_hz: Optional[float] = None,
    bpfi_hz: Optional[float] = None,
    bsf_hz: Optional[float] = None,
    ftf_hz: Optional[float] = None,
    machine_group: str = "group2",
    machine_description: str = "",
) -> Union[dict, ErrorResult]:
    """Full automated vibration diagnosis pipeline.

    1. Compute FFT and extract shaft-frequency features (requires rpm)
    2. Optionally perform envelope analysis for bearing faults
    3. Classify faults (unbalance, misalignment, looseness, bearing)
    4. Assess ISO 10816 severity
    5. Generate human-readable report

    rpm is optional but strongly recommended.  Without it the tool cannot
    perform shaft-frequency analysis and will only report basic statistics.

    Signal input: provide data_id referencing a signal from get_vibration_data.

    Bearing information can be provided in three ways:
        a) Direct fault frequencies in Hz: bpfo_hz, bpfi_hz, bsf_hz, ftf_hz
        b) Custom geometry: bearing_n_balls, bearing_ball_dia_mm, bearing_pitch_dia_mm
        c) Database lookup: bearing_designation (e.g., '6205', 'NU206')

    Args:
        data_id: Reference to a stored signal (from get_vibration_data).
        rpm: Shaft speed in RPM.  Omit if unknown.
        bearing_designation: Bearing code from built-in database.
        bearing_n_balls: Number of rolling elements (custom geometry).
        bearing_ball_dia_mm: Ball diameter in mm (custom geometry).
        bearing_pitch_dia_mm: Pitch diameter in mm (custom geometry).
        bearing_contact_angle_deg: Contact angle in degrees (default 0).
        bpfo_hz: Known BPFO in Hz.
        bpfi_hz: Known BPFI in Hz.
        bsf_hz: Known BSF in Hz.
        ftf_hz: Known FTF in Hz.
        machine_group: ISO 10816 group ('group1'..'group4').
        machine_description: Free text describing the machine for the report.
    """
    try:
        sig, sr = _resolve_signal(data_id)
    except ValueError as e:
        return ErrorResult(error=str(e))

    # Step 1: FFT
    fft_result = compute_fft(sig, sr)
    freqs = np.array(fft_result["frequencies"])
    mags = np.array(fft_result["magnitude"])

    # Basic time-domain statistics
    ts_rms = float(np.sqrt(np.mean(sig**2)))
    ts_peak = float(np.max(np.abs(sig)))
    ts_crest = ts_peak / ts_rms if ts_rms > 0 else 0.0
    ts_mean = float(np.mean(sig))
    ts_std = float(np.std(sig, ddof=1))
    ts_kurtosis = (
        float(np.mean(((sig - ts_mean) / ts_std) ** 4) - 3.0)
        if ts_std > 0 and len(sig) >= 4
        else 0.0
    )

    # ISO 10816 — convert acceleration (g) -> velocity (mm/s)
    rms_vel_mms = _accel_g_to_velocity_rms_mms(sig, sr)

    # If RPM is not provided, skip shaft-frequency analysis
    if rpm is None or rpm <= 0:
        iso_no_rpm = assess_iso10816(rms_vel_mms, machine_group)
        return {
            "warning": (
                "RPM not provided — shaft-frequency analysis (1x, 2x, etc.) "
                "was skipped.  Only basic signal statistics are reported.  "
                "To enable full diagnosis, provide the shaft RPM."
            ),
            "signal_statistics": {
                "rms_g": round(ts_rms, 6),
                "peak_g": round(ts_peak, 6),
                "crest_factor": round(ts_crest, 2),
                "kurtosis": round(ts_kurtosis, 2),
                "duration_s": round(len(sig) / sr, 4),
                "sample_rate_hz": sr,
                "rms_velocity_mm_s": round(rms_vel_mms, 3),
            },
            "fft_summary": _compact_spectrum(freqs, mags, top_n=20),
            "iso_10816": iso_no_rpm,
            "diagnoses": [],
            "shaft_features": None,
            "bearing_analysis": None,
            "bearing_info_source": None,
            "report_markdown": (
                "## Vibration Analysis (RPM unknown)\n\n"
                f"| Metric | Value |\n|---|---|\n"
                f"| RMS (acceleration) | {ts_rms:.4f} g |\n"
                f"| Peak (acceleration) | {ts_peak:.4f} g |\n"
                f"| RMS (velocity, 10-1000 Hz) | {rms_vel_mms:.3f} mm/s |\n"
                f"| Crest Factor | {ts_crest:.1f} |\n"
                f"| Kurtosis | {ts_kurtosis:.1f} |\n\n"
                f"**ISO 10816 Severity:** Zone {iso_no_rpm['iso_zone']} — "
                f"{rms_vel_mms:.3f} mm/s RMS — {iso_no_rpm['description']}\n\n"
                "**Note:** RPM was not provided.  Fault classification "
                "(unbalance, misalignment, looseness) requires shaft frequency."
            ),
        }

    shaft_freq = rpm / 60.0

    # Step 2: Shaft features
    features = extract_shaft_features(freqs, mags, shaft_freq, time_signal=sig)

    # Step 3: Resolve bearing fault frequencies
    fault_freqs: dict[str, float] = {}
    bearing_info_source = None

    # Method A: Direct fault frequencies in Hz
    direct_hz = any(v is not None and v > 0 for v in [bpfo_hz, bpfi_hz, bsf_hz, ftf_hz])
    if direct_hz:
        bearing_info_source = "user-provided frequencies"
        for key, hz_val in [
            ("bpfo", bpfo_hz),
            ("bpfi", bpfi_hz),
            ("bsf", bsf_hz),
            ("ftf", ftf_hz),
        ]:
            if hz_val is not None and hz_val > 0:
                fault_freqs[key] = hz_val
    # Method B: Custom geometry
    elif bearing_n_balls and bearing_ball_dia_mm and bearing_pitch_dia_mm:
        bearing_info_source = "custom geometry"
        bf = compute_bearing_frequencies(
            rpm,
            bearing_n_balls,
            bearing_ball_dia_mm,
            bearing_pitch_dia_mm,
            bearing_contact_angle_deg,
            "custom",
        )
        fault_freqs = {"bpfo": bf.bpfo, "bpfi": bf.bpfi, "bsf": bf.bsf, "ftf": bf.ftf}
    # Method C: Database lookup
    elif bearing_designation:
        bearing = get_bearing(bearing_designation)
        if bearing:
            bearing_info_source = f"database ({bearing.name})"
            bf = compute_bearing_frequencies(
                rpm,
                bearing.n_balls,
                bearing.ball_dia,
                bearing.pitch_dia,
                bearing.contact_angle,
                bearing.name,
            )
            fault_freqs = {
                "bpfo": bf.bpfo,
                "bpfi": bf.bpfi,
                "bsf": bf.bsf,
                "ftf": bf.ftf,
            }

    # Envelope analysis if any fault frequencies are available
    bearing_results = None
    if fault_freqs:
        env = envelope_spectrum(sig, sr)
        env_freqs = np.array(env["frequencies"])
        env_mags = np.array(env["envelope_spectrum"])
        bearing_results = {}
        for key, freq_val in fault_freqs.items():
            bearing_results[key] = check_bearing_peaks(env_freqs, env_mags, freq_val)

    # Step 4: Classify
    diagnoses = classify_faults(features, bearing_results)

    # Step 5: ISO 10816
    iso = assess_iso10816(rms_vel_mms, machine_group)

    # Step 6: Report
    report = generate_diagnosis_summary(diagnoses, iso, machine_description)

    return {
        "diagnoses": [d.to_dict() for d in diagnoses],
        "iso_10816": iso,
        "shaft_features": {
            "shaft_freq_hz": features.f_shaft,
            "amp_1x": round(features.amp_1x, 6),
            "amp_2x": round(features.amp_2x, 6),
            "amp_3x": round(features.amp_3x, 6),
            "amp_half_x": round(features.amp_half_x, 6),
            "kurtosis": round(features.kurtosis, 2),
            "crest_factor": round(features.crest_factor, 2),
        },
        "signal_statistics": {
            "rms_g": round(ts_rms, 6),
            "peak_g": round(ts_peak, 6),
            "crest_factor": round(ts_crest, 2),
            "kurtosis": round(ts_kurtosis, 2),
            "duration_s": round(len(sig) / sr, 4),
            "sample_rate_hz": sr,
            "rms_velocity_mm_s": round(rms_vel_mms, 3),
        },
        "bearing_analysis": (
            {k: v for k, v in bearing_results.items()} if bearing_results else None
        ),
        "bearing_info_source": bearing_info_source,
        "report_markdown": report,
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
