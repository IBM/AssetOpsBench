# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/LGDiMaggio/claude-stwinbox-diagnostics/tree/main/mcp-servers/vibration-analysis-mcp
"""
Bearing characteristic frequency calculator.

Computes BPFI, BPFO, BSF, and FTF from bearing geometry and shaft RPM.
These are the fundamental frequencies used in envelope analysis to detect
bearing faults in rotating machinery.

Formulas based on standard rolling element bearing kinematics:
    FTF  = (RPM/60) * 0.5 * (1 - Bd/Pd * cos(alpha))
    BPFO = (RPM/60) * N_balls/2 * (1 - Bd/Pd * cos(alpha))
    BPFI = (RPM/60) * N_balls/2 * (1 + Bd/Pd * cos(alpha))
    BSF  = (RPM/60) * Pd/(2*Bd) * (1 - (Bd/Pd * cos(alpha))^2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class BearingGeometry:
    """Bearing physical dimensions."""

    name: str
    n_balls: int  # Number of rolling elements
    ball_dia: float  # Ball/roller diameter (mm)
    pitch_dia: float  # Pitch diameter (mm)
    contact_angle: float = 0.0  # Contact angle in degrees


@dataclass
class BearingFrequencies:
    """Calculated bearing characteristic frequencies at a given RPM."""

    rpm: float
    ftf: float  # Fundamental Train Frequency (cage)
    bpfo: float  # Ball Pass Frequency Outer race
    bpfi: float  # Ball Pass Frequency Inner race
    bsf: float  # Ball Spin Frequency
    bearing_name: str = ""

    def to_dict(self) -> dict:
        return {
            "bearing": self.bearing_name,
            "rpm": self.rpm,
            "shaft_frequency_hz": self.rpm / 60.0,
            "ftf_hz": round(self.ftf, 3),
            "bpfo_hz": round(self.bpfo, 3),
            "bpfi_hz": round(self.bpfi, 3),
            "bsf_hz": round(self.bsf, 3),
            "harmonics": {
                "bpfo_2x": round(2 * self.bpfo, 3),
                "bpfo_3x": round(3 * self.bpfo, 3),
                "bpfi_2x": round(2 * self.bpfi, 3),
                "bpfi_3x": round(3 * self.bpfi, 3),
                "bsf_2x": round(2 * self.bsf, 3),
            },
        }


def compute_bearing_frequencies(
    rpm: float,
    n_balls: int,
    ball_dia: float,
    pitch_dia: float,
    contact_angle: float = 0.0,
    bearing_name: str = "Unknown",
) -> BearingFrequencies:
    """
    Compute bearing characteristic frequencies.

    Args:
        rpm: Shaft rotational speed in RPM
        n_balls: Number of rolling elements (balls or rollers)
        ball_dia: Ball/roller diameter in mm
        pitch_dia: Pitch (cage) diameter in mm
        contact_angle: Contact angle in degrees (0 for radial bearings)
        bearing_name: Optional bearing designation

    Returns:
        BearingFrequencies with FTF, BPFO, BPFI, BSF values in Hz
    """
    f_shaft = rpm / 60.0
    alpha_rad = math.radians(contact_angle)
    ratio = ball_dia / pitch_dia

    ftf = f_shaft * 0.5 * (1.0 - ratio * math.cos(alpha_rad))
    bpfo = f_shaft * (n_balls / 2.0) * (1.0 - ratio * math.cos(alpha_rad))
    bpfi = f_shaft * (n_balls / 2.0) * (1.0 + ratio * math.cos(alpha_rad))
    bsf = f_shaft * (pitch_dia / (2.0 * ball_dia)) * (
        1.0 - (ratio * math.cos(alpha_rad)) ** 2
    )

    return BearingFrequencies(
        rpm=rpm,
        ftf=ftf,
        bpfo=bpfo,
        bpfi=bpfi,
        bsf=bsf,
        bearing_name=bearing_name,
    )


# ---------------------------------------------------------------------------
# Common bearing database
# ---------------------------------------------------------------------------

COMMON_BEARINGS: dict[str, BearingGeometry] = {
    "6205": BearingGeometry(
        name="6205 (Deep groove)",
        n_balls=9,
        ball_dia=7.938,
        pitch_dia=38.5,
        contact_angle=0,
    ),
    "6206": BearingGeometry(
        name="6206 (Deep groove)",
        n_balls=9,
        ball_dia=9.525,
        pitch_dia=46.0,
        contact_angle=0,
    ),
    "6207": BearingGeometry(
        name="6207 (Deep groove)",
        n_balls=9,
        ball_dia=11.112,
        pitch_dia=53.5,
        contact_angle=0,
    ),
    "6208": BearingGeometry(
        name="6208 (Deep groove)",
        n_balls=9,
        ball_dia=12.7,
        pitch_dia=60.0,
        contact_angle=0,
    ),
    "6305": BearingGeometry(
        name="6305 (Deep groove)",
        n_balls=8,
        ball_dia=10.319,
        pitch_dia=39.04,
        contact_angle=0,
    ),
    "6306": BearingGeometry(
        name="6306 (Deep groove)",
        n_balls=8,
        ball_dia=12.303,
        pitch_dia=46.36,
        contact_angle=0,
    ),
    "NU205": BearingGeometry(
        name="NU205 (Cylindrical roller)",
        n_balls=13,
        ball_dia=7.5,
        pitch_dia=38.5,
        contact_angle=0,
    ),
    "NU206": BearingGeometry(
        name="NU206 (Cylindrical roller)",
        n_balls=13,
        ball_dia=9.0,
        pitch_dia=46.0,
        contact_angle=0,
    ),
    "7205": BearingGeometry(
        name="7205 (Angular contact)",
        n_balls=12,
        ball_dia=7.144,
        pitch_dia=38.0,
        contact_angle=25,
    ),
}


def get_bearing(designation: str) -> Optional[BearingGeometry]:
    """Look up a bearing by its designation."""
    return COMMON_BEARINGS.get(designation.upper())


def list_bearings() -> list[dict]:
    """List all bearings in the database."""
    return [
        {
            "designation": k,
            "name": v.name,
            "n_balls": v.n_balls,
            "ball_dia_mm": v.ball_dia,
            "pitch_dia_mm": v.pitch_dia,
            "contact_angle_deg": v.contact_angle,
        }
        for k, v in COMMON_BEARINGS.items()
    ]
