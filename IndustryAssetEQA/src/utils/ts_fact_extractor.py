# src/utils/ts_fact_extractor.py

"""
Time-series Fact Extractor for PdM dataset.

Input CSVs:
  - PdM_telemetry.csv : hourly sensor readings
      columns: datetime, machineID, volt, rotate, pressure, vibration
  - PdM_failures.csv  : failure events
      columns: datetime, machineID, failure (e.g., comp1, comp2, ...)
  - PdM_errors.csv    : error events
      columns: datetime, machineID, errorID
  - PdM_maint.csv     : maintenance events
      columns: datetime, machineID, comp
  - PdM_machines.csv  : machine metadata
      columns: machineID, model, age

Output:
  - JSONL file of episode-level facts (e.g., pdm_facts.jsonl), where each fact is one
    time window (e.g., last 24h of telemetry) for one machine, labeled by:
      - failure component (for failure episodes), or
      - "healthy" (for healthy episodes).

Each fact has the structure:

{
  "fact_id": "pdm_m1_comp4_2015-01-05T06",
  "dataset": "pdm",
  "source_file": "PdM_telemetry.csv",
  "asset_id": "machine_1",
  "machineID": 1,

  "episode_type": "failure_window",
  "failure_component": "comp4",
  "failure_time": "2015-01-05 06:00:00",
  "start_time": "2015-01-04 06:00:00",
  "end_time": "2015-01-05 06:00:00",

  "label": "comp4",           # or "healthy"

  "features": [
    {"name": "volt_mean", "value": 170.2},
    {"name": "volt_std", "value": 8.3},
    {"name": "volt_min", "value": 150.0},
    {"name": "volt_max", "value": 190.0},
    {"name": "volt_trend", "value": 0.15},
    {"name": "vibration_max", "value": 70.5},
    {"name": "error_count_last_window", "value": 3},
    {"name": "hours_since_last_maint_comp4", "value": 450.0},
    {"name": "machine_age", "value": 18},
    {"name": "model_ML", "value": 1},
    ...
  ],

  "provenance": {
    "telemetry_source_file": "PdM_telemetry.csv",
    "telemetry_time_range": ["2015-01-04 06:00:00", "2015-01-05 06:00:00"],
    "failure_source_file": "PdM_failures.csv",
    "failure_index": 123,
    "errors_source_file": "PdM_errors.csv",
    "maint_source_file": "PdM_maint.csv",
    "machines_source_file": "PdM_machines.csv"
  },

  "row_index": 123  # we use the failure row index as a primary row_index for compatibility
}

You can ingest the resulting JSONL into EpisodicStore exactly like for static facts.
"""

from __future__ import annotations
import json
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np


def _parse_datetime(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def _compute_telemetry_features(window: pd.DataFrame) -> Dict[str, float]:
    """
    Compute simple aggregate features over a telemetry window.
    Assumes columns: volt, rotate, pressure, vibration.
    """
    feats: Dict[str, float] = {}
    if window.empty:
        return feats

    sensors = ["volt", "rotate", "pressure", "vibration"]
    for col in sensors:
        if col not in window.columns:
            continue
        series = window[col].astype(float)
        feats[f"{col}_mean"] = float(series.mean())
        feats[f"{col}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        feats[f"{col}_min"] = float(series.min())
        feats[f"{col}_max"] = float(series.max())

        # Simple linear trend via least-squares slope (if enough points)
        if len(series) > 1:
            # x as 0..n-1, y as values
            x = np.arange(len(series), dtype=float)
            y = series.values.astype(float)
            # slope only
            slope, _ = np.polyfit(x, y, 1)
            feats[f"{col}_trend"] = float(slope)
        else:
            feats[f"{col}_trend"] = 0.0

    return feats


def _compute_error_features(
    errors_window: pd.DataFrame
) -> Dict[str, float]:
    """
    Aggregate error events in the window.
    """
    feats: Dict[str, float] = {}
    if errors_window.empty:
        feats["error_count_last_window"] = 0.0
        feats["distinct_error_types_last_window"] = 0.0
        return feats

    feats["error_count_last_window"] = float(len(errors_window))
    if "errorID" in errors_window.columns:
        feats["distinct_error_types_last_window"] = float(errors_window["errorID"].nunique())
    else:
        feats["distinct_error_types_last_window"] = 0.0
    return feats


def _hours_since_last_maintenance(
    maint_df: pd.DataFrame,
    machine_id: int,
    ref_time: pd.Timestamp,
    component: Optional[str] = None,
) -> float:
    """
    Compute hours since last maintenance event for the given machine
    (and component, if provided) strictly before ref_time.
    Returns a float (hours); if no prior maintenance, returns -1.0.
    """
    mask = maint_df["machineID"] == machine_id
    if component is not None and "comp" in maint_df.columns:
        mask &= maint_df["comp"] == component

    past_events = maint_df[mask & (maint_df["datetime"] < ref_time)]
    if past_events.empty:
        return -1.0

    last_time = past_events["datetime"].max()
    delta = ref_time - last_time
    return delta.total_seconds() / 3600.0


def _one_hot_model(model: str) -> Dict[str, float]:
    """
    Simple one-hot encoding for machine model.
    If you have models like 'model1', 'model2', create features model_model1, etc.
    """
    feats: Dict[str, float] = {}
    if model:
        key = f"model_{model}"
        feats[key] = 1.0
    return feats


def _machine_static_features(
    machines_df: pd.DataFrame,
    machine_id: int
) -> Dict[str, float]:
    """
    Extract static features for a given machine from PdM_machines.csv
    (e.g., age, model one-hot).
    """
    feats: Dict[str, float] = {}
    row = machines_df[machines_df["machineID"] == machine_id]
    if row.empty:
        return feats
    row = row.iloc[0]
    if "age" in row:
        try:
            feats["machine_age"] = float(row["age"])
        except Exception:
            pass
    if "model" in row:
        feats.update(_one_hot_model(str(row["model"])))
    return feats


def _build_failure_facts(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
    window_hours: int = 24,
) -> List[Dict[str, Any]]:
    """
    Build failure-centered episode facts:
    For each row in failures, take [failure_time - window_hours, failure_time]
    window for that machineID.
    """
    facts: List[Dict[str, Any]] = []

    failures_sorted = failures.sort_values("datetime").reset_index(drop=True)

    for idx, row in failures_sorted.iterrows():
        machine_id = int(row["machineID"])
        failure_time = row["datetime"]
        failure_label = str(row["failure"])

        window_start = failure_time - pd.Timedelta(hours=window_hours)
        window_end = failure_time

        # Filter telemetry
        tele_win = telemetry[
            (telemetry["machineID"] == machine_id)
            & (telemetry["datetime"] > window_start)
            & (telemetry["datetime"] <= window_end)
        ]

        # Filter errors in same window
        errors_win = errors[
            (errors["machineID"] == machine_id)
            & (errors["datetime"] > window_start)
            & (errors["datetime"] <= window_end)
        ]

        # Maintenance events in same window (for context)
        maint_win = maint[
            (maint["machineID"] == machine_id)
            & (maint["datetime"] > window_start)
            & (maint["datetime"] <= window_end)
        ]

        # Aggregate features
        feat_tele = _compute_telemetry_features(tele_win)
        feat_err = _compute_error_features(errors_win)
        hours_since_last_maint_comp = _hours_since_last_maintenance(
            maint_df=maint,
            machine_id=machine_id,
            ref_time=failure_time,
            component=failure_label,
        )
        feat_maint = {
            f"hours_since_last_maint_{failure_label}": hours_since_last_maint_comp
        }

        feat_machine = _machine_static_features(machines_df=machines, machine_id=machine_id)

        # Merge all feature dicts
        features_dict = {}
        for d in [feat_tele, feat_err, feat_maint, feat_machine]:
            features_dict.update(d)

        # Convert to list-of-dicts format
        features_list = [
            {"name": k, "value": float(v) if v is not None else None}
            for k, v in features_dict.items()
        ]

        fact_id = f"pdm_m{machine_id}_{failure_label}_{failure_time.strftime('%Y-%m-%dT%H')}"
        fact = {
            "fact_id": fact_id,
            "dataset": "pdm",
            "source_file": "PdM_telemetry.csv",
            "asset_id": f"machine_{machine_id}",
            "machineID": machine_id,

            "episode_type": "failure_window",
            "failure_component": failure_label,
            "failure_time": failure_time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time": window_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": window_end.strftime("%Y-%m-%d %H:%M:%S"),

            "label": failure_label,  # this is what QA builder will use

            "features": features_list,

            "provenance": {
                "telemetry_source_file": "PdM_telemetry.csv",
                "telemetry_time_range": [
                    window_start.strftime("%Y-%m-%d %H:%M:%S"),
                    window_end.strftime("%Y-%m-%d %H:%M:%S"),
                ],
                "failure_source_file": "PdM_failures.csv",
                "failure_index": int(idx),
                "errors_source_file": "PdM_errors.csv",
                "maint_source_file": "PdM_maint.csv",
                "machines_source_file": "PdM_machines.csv",
            },

            # For compatibility with episodic_store that expects a row_index
            "row_index": int(idx),
        }

        facts.append(fact)

    return facts


def _build_healthy_facts(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
    window_hours: int = 24,
    horizon_hours: int = 24,
    max_per_machine: int = 50,
) -> List[Dict[str, Any]]:
    """
    Build "healthy" episode facts:
    Sample telemetry timestamps where there is NO failure for that machine
    within [t, t + horizon_hours]. We then take [t - window_hours, t] window.

    This is a heuristic; you can refine it later.
    """
    facts: List[Dict[str, Any]] = []
    failures_by_machine = failures.groupby("machineID")["datetime"].apply(list).to_dict()

    for machine_id, tele_m in telemetry.groupby("machineID"):
        tele_m = tele_m.sort_values("datetime")
        fail_times = failures_by_machine.get(machine_id, [])

        # Simple subsampling of candidate center times
        if len(tele_m) == 0:
            continue
        # Take every k-th timestamp to reduce number of episodes
        step = max(1, len(tele_m) // max_per_machine)
        candidate_times = tele_m["datetime"].iloc[::step]

        count_for_machine = 0

        for t_center in candidate_times:
            # Check there is no failure within [t_center, t_center + horizon_hours]
            future_fail = [
                ft for ft in fail_times
                if (ft >= t_center) and (ft <= t_center + pd.Timedelta(hours=horizon_hours))
            ]
            if future_fail:
                continue  # skip, not healthy

            window_start = t_center - pd.Timedelta(hours=window_hours)
            window_end = t_center

            tele_win = telemetry[
                (telemetry["machineID"] == machine_id)
                & (telemetry["datetime"] > window_start)
                & (telemetry["datetime"] <= window_end)
            ]
            if tele_win.empty:
                continue

            errors_win = errors[
                (errors["machineID"] == machine_id)
                & (errors["datetime"] > window_start)
                & (errors["datetime"] <= window_end)
            ]

            # We won't attach a specific component here, just generic maintenance stats
            feat_tele = _compute_telemetry_features(tele_win)
            feat_err = _compute_error_features(errors_win)
            hours_since_any_maint = _hours_since_last_maintenance(
                maint_df=maint,
                machine_id=machine_id,
                ref_time=t_center,
                component=None,
            )
            feat_maint = {"hours_since_last_maint_any": hours_since_any_maint}
            feat_machine = _machine_static_features(machines_df=machines, machine_id=machine_id)

            features_dict = {}
            for d in [feat_tele, feat_err, feat_maint, feat_machine]:
                features_dict.update(d)

            features_list = [
                {"name": k, "value": float(v) if v is not None else None}
                for k, v in features_dict.items()
            ]

            fact_id = f"pdm_m{machine_id}_healthy_{t_center.strftime('%Y-%m-%dT%H')}"
            fact = {
                "fact_id": fact_id,
                "dataset": "pdm",
                "source_file": "PdM_telemetry.csv",
                "asset_id": f"machine_{machine_id}",
                "machineID": machine_id,

                "episode_type": "healthy_window",
                "failure_component": None,
                "failure_time": None,
                "start_time": window_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": window_end.strftime("%Y-%m-%d %H:%M:%S"),

                "label": "healthy",

                "features": features_list,

                "provenance": {
                    "telemetry_source_file": "PdM_telemetry.csv",
                    "telemetry_time_range": [
                        window_start.strftime("%Y-%m-%d %H:%M:%S"),
                        window_end.strftime("%Y-%m-%d %H:%M:%S"),
                    ],
                    "failures_source_file": "PdM_failures.csv",
                    "errors_source_file": "PdM_errors.csv",
                    "maint_source_file": "PdM_maint.csv",
                    "machines_source_file": "PdM_machines.csv",
                },

                "row_index": -1,  # no specific single row; this is synthetic
            }

            facts.append(fact)
            count_for_machine += 1
            if count_for_machine >= max_per_machine:
                break

    return facts


def build_pdm_facts(
    telemetry_path: str,
    failures_path: str,
    errors_path: str,
    maint_path: str,
    machines_path: str,
    out_path: str,
    window_hours: int = 24,
    horizon_hours: int = 24,
    max_healthy_per_machine: int = 50,
    include_healthy: bool = True,
) -> int:
    """
    High-level entry point:
      - load PdM CSVs
      - build failure episodes
      - optionally build healthy episodes
      - write all facts to out_path (JSONL)
    """
    # Load CSVs
    telemetry = pd.read_csv(telemetry_path)
    failures = pd.read_csv(failures_path)
    errors = pd.read_csv(errors_path)
    maint = pd.read_csv(maint_path)
    machines = pd.read_csv(machines_path)

    # Parse datetimes
    telemetry = _parse_datetime(telemetry, "datetime")
    failures = _parse_datetime(failures, "datetime")
    errors = _parse_datetime(errors, "datetime")
    maint = _parse_datetime(maint, "datetime")

    # Build failure and healthy facts
    failure_facts = _build_failure_facts(
        telemetry=telemetry,
        failures=failures,
        errors=errors,
        maint=maint,
        machines=machines,
        window_hours=window_hours,
    )

    healthy_facts: List[Dict[str, Any]] = []
    if include_healthy:
        healthy_facts = _build_healthy_facts(
            telemetry=telemetry,
            failures=failures,
            errors=errors,
            maint=maint,
            machines=machines,
            window_hours=window_hours,
            horizon_hours=horizon_hours,
            max_per_machine=max_healthy_per_machine,
        )

    all_facts = failure_facts + healthy_facts

    # Write JSONL
    with open(out_path, "w") as f:
        for fact in all_facts:
            f.write(json.dumps(fact) + "\n")

    return len(all_facts)


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PdM time-series episode facts.")
    parser.add_argument("--telemetry", required=True, help="Path to PdM_telemetry.csv")
    parser.add_argument("--failures", required=True, help="Path to PdM_failures.csv")
    parser.add_argument("--errors", required=True, help="Path to PdM_errors.csv")
    parser.add_argument("--maint", required=True, help="Path to PdM_maint.csv")
    parser.add_argument("--machines", required=True, help="Path to PdM_machines.csv")
    parser.add_argument("--out", required=True, help="Output JSONL path (e.g. data/pdm_facts.jsonl)")
    parser.add_argument("--window-hours", type=int, default=24, help="Hours in the history window")
    parser.add_argument("--horizon-hours", type=int, default=24, help="No-failure horizon for healthy windows")
    parser.add_argument("--max-healthy-per-machine", type=int, default=50,
                        help="Max healthy episodes per machine")
    parser.add_argument("--no-healthy", action="store_true", help="If set, do not build healthy episodes")

    args = parser.parse_args()

    n = build_pdm_facts(
        telemetry_path=args.telemetry,
        failures_path=args.failures,
        errors_path=args.errors,
        maint_path=args.maint,
        machines_path=args.machines,
        out_path=args.out,
        window_hours=args.window_hours,
        horizon_hours=args.horizon_hours,
        max_healthy_per_machine=args.max_healthy_per_machine,
        include_healthy=not args.no_healthy,
    )
    print(f"Wrote {n} facts to {args.out}")
