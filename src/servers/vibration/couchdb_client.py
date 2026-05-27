"""CouchDB client for fetching vibration sensor data.

Uses a dedicated database (VIBRATION_DBNAME, default 'vibration') to keep
vibration data isolated from the IoT asset database.  Connection
credentials are shared: COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import couchdb3
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("vibration-mcp-server")

COUCHDB_URL = os.environ.get("COUCHDB_URL")
VIBRATION_DBNAME = os.environ.get("VIBRATION_DBNAME", "vibration")
COUCHDB_USER = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")

# Static site as per IoT benchmark; documents are not filtered by a per-document site field.
SITES = ["MAIN"]

_ASSET_META_FIELDS = {"_id", "_rev", "asset_id", "timestamp", "site_name"}


def get_site_list() -> list[str]:
    """Sites accepted for vibration (aligned with IoT ``SITES``)."""
    return list(SITES)


def _get_db() -> Optional[couchdb3.Database]:
    """Lazy CouchDB connection with error handling."""
    if not COUCHDB_URL:
        logger.warning("COUCHDB_URL not set — vibration data from CouchDB unavailable")
        return None
    if not VIBRATION_DBNAME:
        logger.warning(
            "VIBRATION_DBNAME not set — vibration data from CouchDB unavailable"
        )
        return None
    try:
        return couchdb3.Database(
            VIBRATION_DBNAME,
            url=COUCHDB_URL,
            user=COUCHDB_USER,
            password=COUCHDB_PASSWORD,
        )
    except Exception as e:
        logger.error(f"CouchDB connection failed: {e}")
        return None


def fetch_vibration_timeseries(
    asset_id: str,
    sensor_name: str,
    start: str,
    final: Optional[str] = None,
    limit: int = 10000,
) -> Optional[tuple[np.ndarray, float]]:
    """
    Fetch sensor time-series from CouchDB and return as numpy array.

    Queries CouchDB for documents matching the given asset_id and time range,
    extracts values from the specified sensor column, and estimates the
    sample rate from the timestamp spacing.

    Args:
        asset_id: Asset identifier (e.g., 'Chiller 6').
        sensor_name: Name of the sensor field in CouchDB documents.
        start: ISO 8601 start timestamp.
        final: Optional ISO 8601 end timestamp.
        limit: Maximum number of documents to fetch.

    Returns:
        (signal_array, estimated_sample_rate) or None on error.
    """
    db = _get_db()
    if not db:
        return None

    try:
        selector: dict = {
            "asset_id": asset_id,
            "timestamp": {"$gte": datetime.fromisoformat(start).isoformat()},
        }
        if final:
            selector["timestamp"]["$lt"] = datetime.fromisoformat(final).isoformat()

        res = db.find(
            selector,
            limit=limit,
            sort=[{"asset_id": "asc"}, {"timestamp": "asc"}],
        )
    except Exception as e:
        logger.error(f"CouchDB query failed: {e}")
        return None

    docs = res.get("docs", [])
    if not docs:
        logger.info(f"No documents found for {asset_id}/{sensor_name}")
        return None

    # Extract single sensor column
    values: list[float] = []
    timestamps: list[str] = []
    for doc in docs:
        if sensor_name in doc and "timestamp" in doc:
            try:
                values.append(float(doc[sensor_name]))
                timestamps.append(doc["timestamp"])
            except (ValueError, TypeError):
                continue

    if len(values) < 2:
        logger.info(
            f"Insufficient data points ({len(values)}) for {asset_id}/{sensor_name}"
        )
        return None

    signal = np.array(values, dtype=np.float64)

    # Estimate sample rate from timestamp differences
    try:
        ts = [datetime.fromisoformat(t) for t in timestamps]
        diffs = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
        avg_dt = sum(diffs) / len(diffs)
        sample_rate = 1.0 / avg_dt if avg_dt > 0 else 1.0
    except Exception:
        sample_rate = 1.0  # fallback: 1 Hz

    return signal, sample_rate


def list_sensor_fields(asset_id: str) -> list[str]:
    """Return the sensor field names available for an asset in CouchDB."""
    db = _get_db()
    if not db:
        return []
    try:
        res = db.find({"asset_id": asset_id}, limit=1)
        if not res["docs"]:
            return []
        doc = res["docs"][0]
        exclude = {"_id", "_rev", "asset_id", "timestamp"}
        return sorted(k for k in doc.keys() if k not in exclude)
    except Exception as e:
        logger.error(f"Error listing sensors for {asset_id}: {e}")
        return []

def get_asset_time_range(asset_id: str) -> dict:
    """Return start/end timestamps and observation count for an asset.

    Uses the static ``SITES`` list (single-site benchmark); rows are not filtered by
    a per-document site field.
    """
    db = _get_db()
    if not db:
        return {"start": None, "end": None, "total_observations": 0}
    try:
        res = db.find(
            {"asset_id": asset_id, "timestamp": {"$exists": True}},
            fields=["timestamp"],
            limit=100000,
        )
        timestamps = sorted(
            doc["timestamp"]
            for doc in res.get("docs", [])
            if isinstance(doc, dict)
            and doc.get("timestamp")
        )
        if not timestamps:
            return {"start": None, "end": None, "total_observations": 0}
        return {
            "start": timestamps[0],
            "end": timestamps[-1],
            "total_observations": len(timestamps),
        }
    except Exception as e:
        logger.error(f"Error fetching vibration time range for {asset_id}: {e}")
        return {"start": None, "end": None, "total_observations": 0}


def list_asset_coverage() -> list[dict]:
    """Return vibration asset coverage, sensors, and time ranges.

    Uses ``SITES[0]`` as the site label (single static site); documents are not read
    for a site field.
    """
    db = _get_db()
    if not db:
        return []

    effective_site = SITES[0]

    try:
        res = db.find({"asset_id": {"$exists": True}}, limit=100000)
    except Exception as e:
        logger.error(f"Error fetching vibration asset coverage: {e}")
        return []

    grouped: dict[tuple[str, str], dict] = {}
    for doc in res.get("docs", []):
        if not isinstance(doc, dict):
            continue
        asset_id = str(doc.get("asset_id", "")).strip()
        if not asset_id:
            continue

        key = (effective_site, asset_id)
        group = grouped.setdefault(
            key,
            {
                "site_name": effective_site,
                "asset_id": asset_id,
                "sensors": set(),
                "timestamps": [],
            },
        )
        group["sensors"].update(k for k in doc.keys() if k not in _ASSET_META_FIELDS)
        timestamp = doc.get("timestamp")
        if isinstance(timestamp, str) and timestamp:
            group["timestamps"].append(timestamp)

    coverage: list[dict] = []
    for (_site_name, asset_id), group in grouped.items():
        timestamps = sorted(group["timestamps"])
        coverage.append(
            {
                "site_name": group["site_name"],
                "asset_id": asset_id,
                "sensors": sorted(group["sensors"]),
                "time_range": {
                    "start": timestamps[0] if timestamps else None,
                    "end": timestamps[-1] if timestamps else None,
                    "total_observations": len(timestamps),
                },
            }
        )

    return sorted(coverage, key=lambda item: (item["site_name"].lower(), item["asset_id"].lower()))
