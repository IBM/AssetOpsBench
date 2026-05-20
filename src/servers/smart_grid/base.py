"""Shared data-loading helpers for the Smart Grid MCP servers.

All four Smart Grid servers (iot, fmsr, tsfm, wo) read processed CSVs from a
common directory. The directory is configurable via the ``SG_DATA_DIR``
environment variable, with a default of ``./data/sg_processed/`` relative to
the current working directory.

Dataset → server mapping:
  Power Transformers FDD & RUL   →  IoT, TSFM
  DGA Fault Classification        →  FMSR
  Smart Grid Fault Records        →  WO
  Transformer Health Index        →  FMSR (supplemental)
  Current & Voltage Monitoring    →  IoT, TSFM (supplemental)

The expected CSV layout is documented in each loader's docstring below. The
source-project data pipeline that produces these CSVs lives under ``data/`` in
``HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _resolve_data_dir() -> Path:
    """Return the configured Smart Grid data directory.

    Resolution order:
    1. ``SG_DATA_DIR`` environment variable (absolute or cwd-relative).
    2. ``./data/sg_processed/`` relative to current working directory.

    The path is *not* required to exist at import time. Existence is enforced
    on first ``load_*`` call by :func:`_require`.
    """
    env_path = os.environ.get("SG_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path.cwd() / "data" / "sg_processed"


def _data_dir() -> Path:
    """Resolve ``SG_DATA_DIR`` lazily so env-var changes mid-process take effect."""
    return _resolve_data_dir()


# ---------------------------------------------------------------------------
# IoT domain
# ---------------------------------------------------------------------------


def load_asset_metadata() -> pd.DataFrame:
    """Load static asset metadata.

    Source CSV: ``$SG_DATA_DIR/asset_metadata.csv``
    Synthesized from: Power Transformers FDD & RUL dataset.
    """
    path = _data_dir() / "asset_metadata.csv"
    _require(path)
    return pd.read_csv(path)


def load_sensor_readings() -> pd.DataFrame:
    """Load time-series sensor readings indexed by (transformer_id, timestamp).

    Source CSV: ``$SG_DATA_DIR/sensor_readings.csv``
    Synthesized from: Power Transformers FDD & RUL + Current & Voltage
    Monitoring datasets.

    Expected columns:
        transformer_id, timestamp, sensor_id, value, unit, source
    """
    path = _data_dir() / "sensor_readings.csv"
    _require(path)
    return pd.read_csv(path, parse_dates=["timestamp"])


# ---------------------------------------------------------------------------
# FMSR domain
# ---------------------------------------------------------------------------


def load_failure_modes() -> pd.DataFrame:
    """Load failure mode descriptions and their associated sensor signatures.

    Source CSV: ``$SG_DATA_DIR/failure_modes.csv``
    Synthesized from: DGA Fault Classification + Transformer Health Index.

    Expected columns:
        failure_mode_id, name, dga_label, description, severity, iec_code,
        key_gases, recommended_action
    """
    path = _data_dir() / "failure_modes.csv"
    _require(path)
    return pd.read_csv(path)


def load_dga_records() -> pd.DataFrame:
    """Load dissolved gas analysis (DGA) records used for fault classification.

    Source CSV: ``$SG_DATA_DIR/dga_records.csv``
    Synthesized from: DGA Fault Classification dataset.

    Expected columns:
        transformer_id, sample_date, dissolved_h2_ppm, dissolved_ch4_ppm,
        dissolved_c2h2_ppm, dissolved_c2h4_ppm, dissolved_c2h6_ppm,
        dissolved_co_ppm, dissolved_co2_ppm, fault_label, source_dataset
    """
    path = _data_dir() / "dga_records.csv"
    _require(path)
    return pd.read_csv(path, parse_dates=["sample_date"])


# ---------------------------------------------------------------------------
# TSFM domain
# ---------------------------------------------------------------------------


def load_rul_labels() -> pd.DataFrame:
    """Load remaining-useful-life (RUL) ground-truth labels per transformer.

    Source CSV: ``$SG_DATA_DIR/rul_labels.csv``
    Synthesized from: Power Transformers FDD & RUL dataset.

    Expected columns:
        transformer_id, timestamp, rul_days, health_index, fdd_category
    """
    path = _data_dir() / "rul_labels.csv"
    _require(path)
    return pd.read_csv(path, parse_dates=["timestamp"])


# ---------------------------------------------------------------------------
# WO domain
# ---------------------------------------------------------------------------


def load_fault_records() -> pd.DataFrame:
    """Load historical fault / maintenance event records.

    Source CSV: ``$SG_DATA_DIR/fault_records.csv``
    Synthesized from: Smart Grid Fault Records dataset.

    Expected columns:
        transformer_id, fault_id, fault_type, location, voltage_v, current_a,
        power_load_mw, temperature_c, wind_speed_kmh, weather_condition,
        maintenance_status, component_health, duration_hrs, downtime_hrs
    """
    path = _data_dir() / "fault_records.csv"
    _require(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------


def json_safe_record(record: dict) -> dict:
    """Normalize a row dict so it serializes under ``json.dumps(allow_nan=False)``.

    Smart Grid tools commonly return a single CSV row via ``df.iloc[0].to_dict()``
    or a list of records via ``df.to_dict(orient="records")``. Both paths can
    leak ``pandas.Timestamp`` (from ``parse_dates``) and NaN floats through to
    the MCP JSON-RPC response, where the strict serializer rejects them.

    This helper is the canonical boundary normalizer: NaN-likes (including
    ``pd.NaT``) become ``None``, ``pandas.Timestamp`` becomes its ISO date
    string, all other values pass through unchanged.
    """
    normalized: dict = {}
    for key, value in record.items():
        if pd.isna(value):
            normalized[key] = None
        elif isinstance(value, pd.Timestamp):
            normalized[key] = value.date().isoformat()
        else:
            normalized[key] = value
    return normalized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require(path: Path) -> None:
    """Raise a clear error if a processed data file isn't present."""
    if not path.exists():
        raise FileNotFoundError(
            f"Smart Grid processed data file not found: {path}\n"
            "Set SG_DATA_DIR to the directory holding the processed CSVs, "
            "or run the team-side data pipeline (HPML Smart Grid MCP repo "
            "data/ tree) and copy the outputs to the configured location."
        )
