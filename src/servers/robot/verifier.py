"""MultiReadingVerifier — CP-calibrated commit gate.

Formula (updated from research analysis):
    score = 0.35*C + 0.35*A + 0.30*H

    C: multi-read consistency  = 1 - std(readings) / gauge_range_span
    A: gauge-vs-IoT agreement  = 1 - |mean(readings) - iot_value| / span
    H: historical consistency  = 1 - |mean(readings) - hist_baseline| / span
       (H = 0.50 neutral when no history is available)

Hard gates (applied before scoring):
    N < 3                        → BLOCKED       (FM-7b)
    std(readings) < 0.1% range   → PANEL_RECHECK (FM-1: sensor freeze)

Thresholds (placeholder until field-visit calibration):
    score ≥ 0.82  → COMMIT
    score ≥ 0.65  → ESCALATE
    else          → OOD_FLAG

Note on Q (agent self-confidence):
    Q was removed from the formula because LLM self-confidence is poorly
    calibrated and creating a gaming surface. Q is recorded diagnostically
    in commit documents but does not influence the gate.
"""

import statistics
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VerifierResult:
    status: str          # COMMIT | BLOCKED | ESCALATE | OOD_FLAG | PANEL_RECHECK
    score: float
    C: float
    A: float
    H: float
    fm_flag: Optional[str]
    fm_annotations: list
    reason: str


class MultiReadingVerifier:
    TAU_COMMIT = 0.82
    TAU_ESCALATE = 0.65
    FREEZE_EPSILON = 0.001   # std < 0.1% of range → sensor freeze

    def verify(
        self,
        readings: list,
        iot_value: float,
        gauge_range: list,
        historical_baseline: Optional[float] = None,
    ) -> VerifierResult:
        range_span = float(gauge_range[1]) - float(gauge_range[0])
        if range_span <= 0:
            range_span = 1.0

        # Hard gate: insufficient readings
        if len(readings) < 3:
            return VerifierResult(
                status="BLOCKED",
                score=0.0,
                C=0.0, A=0.0, H=0.5,
                fm_flag="FM-7b",
                fm_annotations=["FM-7b: fewer than 3 readings before commit"],
                reason=(
                    f"N={len(readings)} readings supplied; "
                    "minimum 3 required before commit"
                ),
            )

        # Hard gate: sensor freeze (zero variance → stuck gauge or panel)
        std_val = statistics.stdev(readings) if len(readings) > 1 else 0.0
        if std_val < self.FREEZE_EPSILON * range_span:
            return VerifierResult(
                status="PANEL_RECHECK",
                score=0.0,
                C=0.0, A=0.0, H=0.5,
                fm_flag="FM-1",
                fm_annotations=[
                    "FM-1: sensor freeze — readings show no variance across attempts"
                ],
                reason="Zero variance across readings; panel may be stuck or gauge frozen",
            )

        mean_r = statistics.mean(readings)

        C = max(0.0, 1.0 - std_val / range_span)
        A = max(0.0, 1.0 - abs(mean_r - iot_value) / range_span)
        H = (
            max(0.0, 1.0 - abs(mean_r - historical_baseline) / range_span)
            if historical_baseline is not None
            else 0.5
        )

        score = round(0.35 * C + 0.35 * A + 0.30 * H, 4)

        fm_annotations = []
        if A < 0.15:
            fm_annotations.append(
                "FM-7: sensor-physical contradiction (reading vs IoT sensor)"
            )
        if historical_baseline is not None and H < 0.15:
            fm_annotations.append(
                "FM-7c: historical outlier (reading contradicts asset baseline)"
            )

        fm_flag = fm_annotations[0].split(":")[0] if fm_annotations else None

        if score >= self.TAU_COMMIT:
            status = "COMMIT"
        elif score >= self.TAU_ESCALATE:
            status = "ESCALATE"
        else:
            status = "OOD_FLAG"
            if not fm_flag:
                fm_flag = "FM-7"

        reason = f"score={score:.3f} (C={C:.2f}, A={A:.2f}, H={H:.2f})"
        if fm_annotations:
            reason += "; " + "; ".join(fm_annotations)

        return VerifierResult(
            status=status,
            score=score,
            C=round(C, 4),
            A=round(A, 4),
            H=round(H, 4),
            fm_flag=fm_flag,
            fm_annotations=fm_annotations,
            reason=reason,
        )
