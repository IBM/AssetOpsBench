"""Initialize the CouchDB work-order database from CSV files.

Usage:
    python -m couchdb.init_wo [--data-dir <path>] [--db <name>] [--drop]

Environment variables (or .env):
    COUCHDB_URL        e.g. http://localhost:5984
    COUCHDB_USERNAME   admin user
    COUCHDB_PASSWORD   admin password
    WO_DBNAME  target database (default: workorder)
    WO_DATA_DIR        override CSV directory
"""

import argparse
import logging
import math
import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(__file__)
_DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "sample_data", "work_order")

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "password")
WO_DBNAME = os.environ.get("WO_DBNAME", "workorder")
WO_DATA_DIR = os.environ.get("WO_DATA_DIR", _DEFAULT_DATA_DIR)

# ---------------------------------------------------------------------------
# CSV → dataset mapping
# ---------------------------------------------------------------------------

# (csv_filename, _dataset key, date columns and their parse formats)
_DATASETS = [
    (
        "all_wo_with_code_component_events.csv",
        "wo_events",
        {"actual_finish": "%m/%d/%y %H:%M"},
    ),
    (
        "event.csv",
        "events",
        {"event_time": "%Y-%m-%d %H:%M:%S"},
    ),
    (
        "alert_events.csv",
        "alert_events",
        {"start_time": "%m/%d/%y %H:%M", "end_time": "%m/%d/%y %H:%M"},
    ),
    ("alert_rule.csv", "alert_rule", {}),
    ("alert_rule_failure_code_mapping.csv", "alert_rule_fc_mapping", {}),
    ("anomaly_to_failure_code_mapping.csv", "anomaly_fc_mapping", {}),
    ("failure_codes.csv", "failure_codes", {}),
    ("primary_failure_codes.csv", "primary_failure_codes", {}),
    ("component.csv", "component", {}),
]

# Mango indexes to create: list of field-lists
_INDEXES = [
    ["dataset", "equipment_id"],
    ["dataset", "actual_finish"],
    ["dataset", "event_time"],
    ["dataset", "rule_id"],
    ["dataset", "primary_code"],
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUTH = (COUCHDB_USERNAME, COUCHDB_PASSWORD)


def _db_url(db: str, *parts: str) -> str:
    return "/".join([COUCHDB_URL.rstrip("/"), db] + list(parts))


def _ensure_db(db_name: str, drop: bool) -> None:
    url = _db_url(db_name)
    resp = requests.head(url, auth=_AUTH, timeout=10)
    if resp.status_code == 200:
        if drop:
            logger.info("Dropping existing database '%s'…", db_name)
            requests.delete(url, auth=_AUTH, timeout=10).raise_for_status()
        else:
            logger.info("Database '%s' already exists — skipping creation.", db_name)
            return
    logger.info("Creating database '%s'…", db_name)
    requests.put(url, auth=_AUTH, timeout=10).raise_for_status()


def _create_indexes(db_name: str) -> None:
    url = _db_url(db_name, "_index")
    for fields in _INDEXES:
        payload = {"index": {"fields": fields}, "type": "json"}
        resp = requests.post(url, json=payload, auth=_AUTH, timeout=10)
        resp.raise_for_status()
        logger.info("Index on %s: %s", fields, resp.json().get("result", "?"))


def _bulk_insert(db_name: str, docs: list, batch_size: int = 500) -> None:
    url = _db_url(db_name, "_bulk_docs")
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        resp = requests.post(url, json={"docs": batch}, auth=_AUTH, timeout=60)
        resp.raise_for_status()
        errors = [r for r in resp.json() if r.get("error")]
        if errors:
            logger.warning("%d bulk-insert errors in batch %d", len(errors), i // batch_size)
        logger.info("Inserted batch %d/%d (%d docs)", i // batch_size + 1, math.ceil(total / batch_size), len(batch))


def _row_to_doc(row: dict, dataset: str, date_cols: dict) -> dict:
    """Convert a CSV row dict to a CouchDB document dict."""
    doc: dict = {"dataset": dataset}
    for k, v in row.items():
        if pd.isna(v):
            doc[k] = None
        elif k in date_cols and isinstance(v, pd.Timestamp):
            doc[k] = v.isoformat()
        else:
            doc[k] = v
    return doc


def load_dataset(data_dir: str, csv_file: str, dataset: str, date_cols: dict) -> list:
    path = os.path.join(data_dir, csv_file)
    if not os.path.exists(path):
        logger.warning("CSV not found, skipping: %s", path)
        return []

    df = pd.read_csv(path, dtype=str)
    for col, fmt in date_cols.items():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")

    docs = [_row_to_doc(row, dataset, date_cols) for row in df.to_dict(orient="records")]
    logger.info("Loaded %d rows from '%s' → dataset '%s'", len(docs), csv_file, dataset)
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize CouchDB work-order database from CSVs.")
    parser.add_argument("--data-dir", default=WO_DATA_DIR, help="Directory containing CSVs")
    parser.add_argument("--db", default=WO_DBNAME, help="CouchDB database name")
    parser.add_argument("--drop", action="store_true", help="Drop and recreate database if it exists")
    args = parser.parse_args()

    logger.info("CouchDB URL: %s", COUCHDB_URL)
    logger.info("Database: %s", args.db)
    logger.info("Data dir: %s", args.data_dir)

    _ensure_db(args.db, drop=args.drop)

    all_docs: list = []
    for csv_file, dataset, date_cols in _DATASETS:
        all_docs.extend(load_dataset(args.data_dir, csv_file, dataset, date_cols))

    if not all_docs:
        logger.error("No documents to insert — check --data-dir path.")
        sys.exit(1)

    logger.info("Inserting %d total documents…", len(all_docs))
    _bulk_insert(args.db, all_docs)

    _create_indexes(args.db)
    logger.info("Done. Database '%s' is ready.", args.db)


if __name__ == "__main__":
    main()
