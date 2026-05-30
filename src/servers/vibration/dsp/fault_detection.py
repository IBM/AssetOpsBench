# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp
"""
Automated fault detection and classification for rotating machinery.

Uses vibration spectral features to identify common mechanical faults:
    - Unbalance (1x shaft speed dominant)
    - Misalignment (2x shaft speed, axial component)
    - Mechanical looseness (many harmonics of shaft speed)
    - Bearing faults (BPFO / BPFI / BSF / FTF in envelope spectrum)

Also provides ISO 10816 vibration severity classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# ISO 10816 Vibration Severity (RMS velocity mm/s)
# ---------------------------------------------------------------------------

# Thresholds per machine group (mm/s RMS, 10-1000 Hz)
ISO_10816_THRESHOLDS: dict[str, dict[str, float]] = {
    # Group 1: Large machines > 300 kW on rigid foundations
    "group1": {"A_good": 2.8, "B_acceptable": 7.1, "C_alarm": 18.0},
    # Group 2: Medium machines 15-300 kW on rigid foundations
    "group2": {"A_good": 1.4, "B_acceptable": 2.8, "C_alarm": 7.1},
    # Group 3: Large machines on flexible foundations
    "group3": {"A_good": 3.5, "B_acceptable": 9.0, "C_alarm": 22.4},
    # Group 4: Small machines < 15 kW
    "group4": {"A_good": 0.71, "B_acceptable": 1.8, "C_alarm": 4.5},
}


def assess_iso10816(
    rms_velocity_mm_s: float,
    machine_group: str = "group2",
) -> dict:
    """
    Classify vibration severity per ISO 10816.

    Args:
        rms_velocity_mm_s: Overall RMS velocity in mm/s (10-1000 Hz band).
        machine_group: One of 'group1' .. 'group4'.

    Returns:
        dict with zone (A/B/C/D), description, and thresholds used.
    """
    thresholds = ISO_10816_THRESHOLDS.get(
        machine_group, ISO_10816_THRESHOLDS["group2"]
    )

    if rms_velocity_mm_s <= thresholds["A_good"]:
        zone, desc = "A", "Good - newly commissioned machines"
    elif rms_velocity_mm_s <= thresholds["B_acceptable"]:
        zone, desc = "B", "Acceptable - long-term operation permitted"
    elif rms_velocity_mm_s <= thresholds["C_alarm"]:
        zone, desc = "C", "Alarm - not suitable for long-term operation"
    else:
        zone, desc = "D", "Danger - risk of damage, stop machine"

    return {
        "rms_velocity_mm_s": round(rms_velocity_mm_s, 3),
        "iso_zone": zone,
        "description": desc,
        "machine_group": machine_group,
        "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
# Spectral fault feature extraction
# ---------------------------------------------------------------------------


@dataclass
class ShaftFeatures:
    """Spectral amplitude at shaft-frequency harmonics."""

    f_shaft: float  # Shaft frequency (Hz) = RPM/60
    amp_1x: float  # 1x amplitude
    amp_2x: float  # 2x amplitude
    amp_3x: float  # 3x amplitude
    amp_half_x: float  # 0.5x amplitude (sub-harmonic)
    rms_overall: float  # Broadband RMS
    crest_factor: float  # Peak / RMS
    kurtosis: float  # Excess kurtosis (Fisher: Gaussian = 0, impulsive > 1)


def extract_shaft_features(
    freqs: NDArray[np.floating] | list[float],
    magnitudes: NDArray[np.floating] | list[float],
    shaft_freq: float,
    time_signal: NDArray[np.floating] | list[float] | None = None,
    tolerance_pct: float = 3.0,
) -> ShaftFeatures:
    """
    Extract amplitudes at shaft-frequency harmonics from a spectrum.

    Args:
        freqs: Frequency axis in Hz.
        magnitudes: Spectrum magnitude array.
        shaft_freq: Shaft rotational frequency in Hz (RPM / 60).
        time_signal: Optional raw time signal for kurtosis / crest factor.
        tolerance_pct: Frequency tolerance percentage.

    Returns:
        ShaftFeatures dataclass.
    """
    freqs = np.asarray(freqs)
    mags = np.asarray(magnitudes)

    def _peak_at(f_target: float) -> float:
        tol = f_target * tolerance_pct / 100.0
        mask = (freqs >= f_target - tol) & (freqs <= f_target + tol)
        if np.any(mask):
            return float(np.max(mags[mask]))
        return 0.0

    amp_1x = _peak_at(shaft_freq)
    amp_2x = _peak_at(2.0 * shaft_freq)
    amp_3x = _peak_at(3.0 * shaft_freq)
    amp_half = _peak_at(0.5 * shaft_freq)
    rms_overall = float(np.sqrt(np.mean(mags**2)))

    # Crest factor & kurtosis from time signal if available
    if time_signal is not None:
        ts = np.asarray(time_signal)
        ts_rms = float(np.sqrt(np.mean(ts**2)))
        crest = float(np.max(np.abs(ts)) / ts_rms) if ts_rms > 0 else 0
        mean_ts = np.mean(ts)
        std_ts = np.std(ts, ddof=1)
        if std_ts > 0 and len(ts) >= 4:
            # Excess kurtosis (Fisher definition): Gaussian = 0
            kurt = float(np.mean(((ts - mean_ts) / std_ts) ** 4) - 3.0)
        else:
            kurt = 0.0
    else:
        crest = 0.0
        kurt = 0.0

    return ShaftFeatures(
        f_shaft=shaft_freq,
        amp_1x=amp_1x,
        amp_2x=amp_2x,
        amp_3x=amp_3x,
        amp_half_x=amp_half,
        rms_overall=rms_overall,
        crest_factor=crest,
        kurtosis=kurt,
    )


# ---------------------------------------------------------------------------
# Fault classification rules
# ---------------------------------------------------------------------------


@dataclass
class FaultDiagnosis:
    """Result of automated fault classification."""

    fault_type: str
    confidence: str  # "high", "medium", "low", "none"
    description: str
    evidence: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "fault_type": self.fault_type,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
        }


def classify_faults(
    features: ShaftFeatures,
    bearing_envelope_results: dict | None = None,
) -> list[FaultDiagnosis]:
    """
    Apply rule-based classification for common rotating-machinery faults.

    Args:
        features: Extracted shaft-frequency features.
        bearing_envelope_results: Optional dict from envelope analysis with
            keys like 'bpfo', 'bpfi', 'bsf', each containing
            {harmonics_detected, confidence}.

    Returns:
        List of FaultDiagnosis objects, sorted by confidence (high first).
    """
    diagnoses: list[FaultDiagnosis] = []
    rms = features.rms_overall if features.rms_overall > 0 else 1e-12

    # --- Unbalance: dominant 1x with low 2x ---
    ratio_1x = features.amp_1x / rms
    ratio_2x = features.amp_2x / rms
    if ratio_1x > 3.0 and features.amp_1x > 2.0 * features.amp_2x:
        conf = "high" if ratio_1x > 5.0 else "medium"
        diagnoses.append(
            FaultDiagnosis(
                fault_type="unbalance",
                confidence=conf,
                description="Mass unbalance detected - dominant 1x shaft speed component.",
                evidence=[
                    f"1x amplitude = {features.amp_1x:.4f} ({ratio_1x:.1f}x RMS)",
                    f"2x amplitude = {features.amp_2x:.4f} "
                    f"(ratio 1x/2x = {features.amp_1x / max(features.amp_2x, 1e-12):.1f})",
                ],
                recommendations=[
                    "Perform balancing of the rotor.",
                    "Check for material build-up or loss on rotating parts.",
                    "Verify coupling alignment (unbalance can be masked by misalignment).",
                ],
            )
        )

    # --- Misalignment: significant 2x (and sometimes 3x) ---
    if ratio_2x > 2.5 and features.amp_2x > 0.5 * features.amp_1x:
        conf = "high" if features.amp_2x > features.amp_1x else "medium"
        diagnoses.append(
            FaultDiagnosis(
                fault_type="misalignment",
                confidence=conf,
                description="Shaft misalignment suspected - elevated 2x component.",
                evidence=[
                    f"2x amplitude = {features.amp_2x:.4f} ({ratio_2x:.1f}x RMS)",
                    f"2x/1x ratio = {features.amp_2x / max(features.amp_1x, 1e-12):.2f}",
                    f"3x amplitude = {features.amp_3x:.4f}",
                ],
                recommendations=[
                    "Check shaft alignment with laser or dial indicator.",
                    "Inspect coupling condition and flexible element wear.",
                    "Verify thermal growth compensation.",
                ],
            )
        )

    # --- Mechanical looseness: many harmonics + sub-harmonics ---
    n_significant = sum(
        1
        for a in [features.amp_1x, features.amp_2x, features.amp_3x]
        if a / rms > 1.5
    )
    if n_significant >= 3 or (features.amp_half_x / rms > 1.5):
        evidence = [f"Harmonics above threshold: {n_significant}/3"]
        if features.amp_half_x / rms > 1.5:
            evidence.append(f"Sub-harmonic 0.5x = {features.amp_half_x:.4f}")
        diagnoses.append(
            FaultDiagnosis(
                fault_type="mechanical_looseness",
                confidence="medium",
                description="Mechanical looseness suggested - multiple shaft harmonics and/or sub-harmonics.",
                evidence=evidence,
                recommendations=[
                    "Inspect and tighten foundation bolts.",
                    "Check bearing housing fit and clearance.",
                    "Look for structural cracks or soft foot.",
                ],
            )
        )

    # --- Impulsiveness (excess kurtosis > 1.0 or crest factor > 5) ---
    if features.kurtosis > 1.0 or features.crest_factor > 5.0:
        diagnoses.append(
            FaultDiagnosis(
                fault_type="impulsive_signal",
                confidence="medium",
                description="Impulsive content detected - may indicate bearing defect or gear tooth damage.",
                evidence=[
                    f"Excess kurtosis = {features.kurtosis:.2f} (healthy ~ 0.0)",
                    f"Crest factor = {features.crest_factor:.2f} (healthy < 4)",
                ],
                recommendations=[
                    "Perform envelope analysis to isolate bearing fault frequencies.",
                    "Inspect gears for pitting or tooth breakage if applicable.",
                ],
            )
        )

    # --- Bearing faults from envelope results ---
    if bearing_envelope_results:
        for fault_key, label in [
            ("bpfo", "Outer race defect"),
            ("bpfi", "Inner race defect"),
            ("bsf", "Ball/roller defect"),
            ("ftf", "Cage defect"),
        ]:
            result = bearing_envelope_results.get(fault_key)
            if result and result.get("confidence", "none") != "none":
                conf = result["confidence"]
                diagnoses.append(
                    FaultDiagnosis(
                        fault_type=f"bearing_{fault_key}",
                        confidence=conf,
                        description=f"{label} detected via envelope analysis.",
                        evidence=[
                            f"Harmonics detected: {result.get('harmonics_detected', 0)}"
                            f"/{result.get('harmonics_checked', 3)}",
                            f"Fault frequency: {result.get('target_frequency_hz', 0):.2f} Hz",
                        ],
                        recommendations=[
                            "Schedule bearing replacement.",
                            "Monitor trend - frequency of data collection should increase.",
                            "Check lubrication condition.",
                        ],
                    )
                )

    # --- No faults ---
    if not diagnoses:
        diagnoses.append(
            FaultDiagnosis(
                fault_type="healthy",
                confidence="medium",
                description="No significant fault patterns detected in the vibration signature.",
                evidence=[
                    f"1x ratio = {ratio_1x:.1f}",
                    f"Kurtosis = {features.kurtosis:.2f}",
                    f"Crest factor = {features.crest_factor:.2f}",
                ],
                recommendations=[
                    "Continue routine monitoring.",
                    "Store this measurement as baseline reference.",
                ],
            )
        )

    # Sort: high > medium > low > none
    priority = {"high": 0, "medium": 1, "low": 2, "none": 3}
    diagnoses.sort(key=lambda d: priority.get(d.confidence, 3))
    return diagnoses


def generate_diagnosis_summary(
    diagnoses: list[FaultDiagnosis],
    iso_assessment: dict | None = None,
    machine_context: str = "",
) -> str:
    """
    Create a human-readable diagnosis summary.

    Args:
        diagnoses: List of FaultDiagnosis objects.
        iso_assessment: Optional ISO 10816 result dict.
        machine_context: Free text describing the machine.

    Returns:
        Formatted markdown string.
    """
    lines = ["## Vibration Diagnosis Summary\n"]

    if machine_context:
        lines.append(f"**Machine:** {machine_context}\n")

    if iso_assessment:
        zone = iso_assessment.get("iso_zone", "?")
        vel = iso_assessment.get("rms_velocity_mm_s", 0)
        desc = iso_assessment.get("description", "")
        lines.append(
            f"**Overall Severity (ISO 10816):** Zone {zone} "
            f"- {vel:.2f} mm/s RMS - {desc}\n"
        )

    lines.append("### Detected Conditions\n")
    for i, d in enumerate(diagnoses, 1):
        lines.append(
            f"#### {i}. {d.fault_type.replace('_', ' ').title()} "
            f"[{d.confidence.upper()}]\n"
        )
        lines.append(f"{d.description}\n")
        lines.append("**Evidence:**")
        for e in d.evidence:
            lines.append(f"- {e}")
        lines.append("\n**Recommendations:**")
        for r in d.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    return "\n".join(lines)
