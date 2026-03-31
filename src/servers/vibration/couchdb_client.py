"""CouchDB client for fetching vibration sensor data.

Uses a dedicated database (VIBRATION_DBNAME, default 'vibration') to keep
vibration data isolated from the IoT chiller database.  Connection
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
