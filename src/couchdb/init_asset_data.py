"""Initialize the CouchDB IoT asset database from a JSON sensor data file.

Usage:
    python -m couchdb.init_asset_data [--data-file <path>] [--db <name>] [--drop]

Environment variables (or .env):
    COUCHDB_URL        e.g. http://localhost:5984
    COUCHDB_USERNAME   admin user
    COUCHDB_PASSWORD   admin password
    IOT_DBNAME         target database (default: chiller)
    ASSET_DATA_FILE    override JSON file path
"""

import argparse
import json
import logging
import math
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(__file__)
_DEFAULT_DATA_FILE = os.path.join(
    _SCRIPT_DIR, "sample_data", "iot", "chiller6_june2020_sensordata_couchdb.json"
)

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "password")
IOT_DBNAME = os.environ.get("IOT_DBNAME", "chiller")
ASSET_DATA_FILE = os.environ.get("ASSET_DATA_FILE", _DEFAULT_DATA_FILE)

_AUTH = (COUCHDB_USERNAME, COUCHDB_PASSWORD)

# Mango index for typical IoT queries
_INDEXES = [
    ["asset_id", "timestamp"],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_url(db: str, *parts: str) -> str:
    return "/".join([COUCHDB_URL.rstrip("/"), db] + list(parts))


def _ensure_db(db_name: str, drop: bool) -> bool:
    """Return True if the database was freshly created (data should be loaded), False if it already existed."""
    url = _db_url(db_name)
    resp = requests.head(url, auth=_AUTH, timeout=10)
    if resp.status_code == 200:
        if drop:
            logger.info("Dropping existing database '%s'…", db_name)
            requests.delete(url, auth=_AUTH, timeout=10).raise_for_status()
        else:
            logger.info("Database '%s' already exists — skipping.", db_name)
            return False
    logger.info("Creating database '%s'…", db_name)
    requests.put(url, auth=_AUTH, timeout=10).raise_for_status()
    return True


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
        logger.info(
            "Inserted batch %d/%d (%d docs)",
            i // batch_size + 1,
            math.ceil(total / batch_size),
            len(batch),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize CouchDB IoT asset database from JSON.")
    parser.add_argument("--data-file", default=ASSET_DATA_FILE, help="Path to sensor data JSON file")
    parser.add_argument("--db", default=IOT_DBNAME, help="CouchDB database name")
    parser.add_argument("--drop", action="store_true", help="Drop and recreate database if it exists")
    args = parser.parse_args()

    logger.info("CouchDB URL: %s", COUCHDB_URL)
    logger.info("Database: %s", args.db)
    logger.info("Data file: %s", args.data_file)

    if not os.path.exists(args.data_file):
        logger.error("Data file not found: %s", args.data_file)
        sys.exit(1)

    with open(args.data_file) as f:
        docs = json.load(f)

    if not isinstance(docs, list) or not docs:
        logger.error("Expected a non-empty JSON array in %s", args.data_file)
        sys.exit(1)

    logger.info("Loaded %d documents from '%s'", len(docs), args.data_file)

    created = _ensure_db(args.db, drop=args.drop)
    if created:
        _bulk_insert(args.db, docs)
        _create_indexes(args.db)
    logger.info("Done. Database '%s' is ready.", args.db)


if __name__ == "__main__":
    main()
