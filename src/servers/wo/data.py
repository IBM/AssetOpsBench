"""Data access helpers for the Work Order MCP server.

Reads from a CouchDB ``workorder`` database populated by ``src/couchdb/init_wo.py``.
Each document carries a ``dataset`` field that acts as a collection discriminator.

Connection is established lazily on first use.  If CouchDB is unavailable the
helpers return ``None`` / empty results so the server can still start.
"""

import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import EventItem, WorkOrderItem

logger = logging.getLogger("wo-mcp-server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COUCHDB_URL: str = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME: str = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD: str = os.environ.get("COUCHDB_PASSWORD", "password")
WO_DBNAME: str = os.environ.get("WO_DBNAME", "workorder")

# ---------------------------------------------------------------------------
# Lazy connection
# ---------------------------------------------------------------------------

_db = None  # couchdb3.Database instance, initialised on first call to _get_db()


def _get_db():
    """Return a live couchdb3.Database, connecting on first call."""
    global _db
    if _db is not None:
        return _db
    try:
        import couchdb3  # lazy import so the server starts without couchdb3 installed

        _db = couchdb3.Database(
            WO_DBNAME,
            url=COUCHDB_URL,
            user=COUCHDB_USERNAME,
            password=COUCHDB_PASSWORD,
        )
        logger.info("Connected to CouchDB database '%s'", WO_DBNAME)
    except Exception as exc:
        logger.error("Failed to connect to CouchDB: %s", exc)
        _db = None
    return _db


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

# Date columns that must be converted from ISO strings after fetch
_DATE_COLS: Dict[str, List[str]] = {
    "wo_events": ["actual_finish"],
    "events": ["event_time"],
    "alert_events": ["start_time", "end_time"],
}


_dataset_cache: Dict[str, Optional[pd.DataFrame]] = {}


def load(dataset: str) -> Optional[pd.DataFrame]:
    """Fetch all documents with ``_dataset == dataset`` and return a DataFrame.

    Results are cached after the first successful load to avoid repeated
    full-collection scans against CouchDB on every tool call.

    Returns ``None`` when CouchDB is unavailable or the dataset is empty.
    """
    if dataset in _dataset_cache:
        cached = _dataset_cache[dataset]
        return cached.copy() if cached is not None else None

    db = _get_db()
    if db is None:
        return None
    try:
        result = db.find(
            selector={"dataset": {"$eq": dataset}},
            limit=100_000,
        )
        docs = result.get("docs", [])
        if not docs:
            logger.warning("No documents found for dataset '%s'", dataset)
            _dataset_cache[dataset] = None
            return None

        df = pd.DataFrame(docs)
        # Drop internal CouchDB fields
        df.drop(columns=[c for c in ("_id", "_rev", "dataset") if c in df.columns], inplace=True)

        # Parse date columns
        for col in _DATE_COLS.get(dataset, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        logger.info("Loaded %d rows for dataset '%s'", len(df), dataset)
        _dataset_cache[dataset] = df
        return df.copy()
    except Exception as exc:
        logger.error("Failed to load dataset '%s': %s", dataset, exc)
        return None


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def filter_df(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """Filter *df* by a dict of ``{column: callable}`` conditions."""
    filtered = df.copy()
    for col, cond in conditions.items():
        if callable(cond):
            filtered = filtered[filtered[col].apply(cond)]
        else:
            filtered = filtered.query(f"{col} {cond}")
    if not filtered.empty:
        filtered = filtered.reset_index(drop=True)
    return filtered


def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO date string (YYYY-MM-DD) or raise ValueError."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD, got '{value}'") from exc


def date_conditions(equipment_id: str, date_col: str, start: Optional[str], end: Optional[str]) -> dict:
    """Build a filter-conditions dict for equipment + optional date range."""
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    cond: dict = {
        "equipment_id": lambda x, eid=equipment_id: isinstance(x, str) and x.strip().lower() == eid.strip().lower()
    }
    if start_dt or end_dt:
        cond[date_col] = lambda x, s=start_dt, e=end_dt: (
            (s is None or x >= s) and (e is None or x <= e)
        )
    return cond


def get_transition_matrix(event_df: pd.DataFrame, event_type_col: str) -> pd.DataFrame:
    """Build a row-normalised Markov transition matrix from a sequence of event types."""
    event_types = event_df[event_type_col].tolist()
    counts: dict = defaultdict(lambda: defaultdict(int))
    for cur, nxt in zip(event_types[:-1], event_types[1:]):
        counts[cur][nxt] += 1
    matrix = pd.DataFrame(counts).fillna(0)
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix


# ---------------------------------------------------------------------------
# Row → model converters
# ---------------------------------------------------------------------------


def row_to_wo(row: Any) -> WorkOrderItem:
    return WorkOrderItem(
        wo_id=str(row.get("wo_id", "")),
        wo_description=str(row.get("wo_description", "")),
        collection=str(row.get("collection", "")),
        primary_code=str(row.get("primary_code", "")),
        primary_code_description=str(row.get("primary_code_description", "")),
        secondary_code=str(row.get("secondary_code", "")),
        secondary_code_description=str(row.get("secondary_code_description", "")),
        equipment_id=str(row.get("equipment_id", "")),
        equipment_name=str(row.get("equipment_name", "")),
        preventive=str(row.get("preventive", "")).upper() == "TRUE",
        work_priority=int(row["work_priority"]) if pd.notna(row.get("work_priority")) else None,
        actual_finish=row["actual_finish"].isoformat() if pd.notna(row.get("actual_finish")) else None,
        duration=str(row.get("duration", "")) if pd.notna(row.get("duration")) else None,
        actual_labor_hours=str(row.get("actual_labor_hours", "")) if pd.notna(row.get("actual_labor_hours")) else None,
    )


def row_to_event(row: Any) -> EventItem:
    return EventItem(
        event_id=str(row.get("event_id", "")),
        event_group=str(row.get("event_group", "")),
        event_category=str(row.get("event_category", "")),
        event_type=str(row["event_type"]) if pd.notna(row.get("event_type")) else None,
        description=str(row["description"]) if pd.notna(row.get("description")) else None,
        equipment_id=str(row.get("equipment_id", "")),
        equipment_name=str(row.get("equipment_name", "")),
        event_time=row["event_time"].isoformat() if pd.notna(row.get("event_time")) else "",
        note=str(row["note"]) if pd.notna(row.get("note")) else None,
    )


def fetch_work_orders(
    df: pd.DataFrame,
    equipment_id: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[WorkOrderItem]:
    """Filter *df* by equipment + date range and return ``WorkOrderItem`` list."""
    cond = date_conditions(equipment_id, "actual_finish", start_date, end_date)
    filtered = filter_df(df, cond)
    if filtered is None or filtered.empty:
        return []
    return [row_to_wo(row) for _, row in filtered.iterrows()]
