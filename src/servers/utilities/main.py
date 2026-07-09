import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
from uuid import uuid4

import couchdb3
import pendulum
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

# Setup logging — default WARNING so stderr stays quiet when used as MCP server;
# set LOG_LEVEL=INFO (or DEBUG) in the environment to see verbose output.
_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("utilities-mcp-server")

# Configuration from environment
COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "password")
CATALOG_DBNAME = os.environ.get("CATALOG_DBNAME", "catalog")
CATALOG_QUERY_LIMIT = 1000

try:
    catalog_db = couchdb3.Database(
        CATALOG_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info("Connected to catalog database: %s", CATALOG_DBNAME)
except Exception as e:
    logger.error("Failed to connect to catalog database: %s", e)
    catalog_db = None

mcp = FastMCP(
    "utilities",
    instructions=(
        "General utilities: read JSON files, get current date/time, and query "
        "asset, sensor, and failure-mode catalog data."
    ),
)


class DateTimeResult(BaseModel):
    currentDateTime: str
    currentDateTimeDescription: str


class TimeEnglishResult(BaseModel):
    english: str
    iso: str


class ErrorResult(BaseModel):
    error: str


class CatalogResult(BaseModel):
    catalog_type: str
    query: Optional[str]
    total: int
    entries: list[dict[str, Any]]
    message: str


# --- Helper Functions ---


def get_temp_filename() -> str:
    tmpdir = tempfile.gettempdir()
    tmppath = Path(tmpdir)
    basepath = Path("cbmdir")
    filename = str(uuid4())

    tmpdir_path = tmppath / basepath
    tmpdir_path.mkdir(parents=True, exist_ok=True)

    filepath = tmpdir_path / (filename + ".json")
    return str(filepath)


def _clean_filter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _find_catalog(
    *,
    catalog_type: str,
    field: str,
    value: Optional[str],
    fields: list[str],
    category: Optional[str] = None,
) -> Union[CatalogResult, ErrorResult]:
    if catalog_db is None:
        return ErrorResult(error="catalog database is not available")

    query_value = _clean_filter(value)
    category_value = _clean_filter(category)
    selector: dict[str, Any] = {field: query_value or {"$exists": True}}
    if category_value is not None:
        selector["category"] = category_value

    try:
        res = catalog_db.find(
            selector,
            fields=fields,
            limit=CATALOG_QUERY_LIMIT,
        )
        docs = res.get("docs", [])
    except Exception as e:
        logger.error("Error querying %s catalog: %s", catalog_type, e)
        return ErrorResult(error=str(e))

    query_parts = []
    if query_value is not None:
        query_parts.append(query_value)
    if category_value is not None:
        query_parts.append(f"category={category_value}")
    query = ", ".join(query_parts) if query_parts else None

    return CatalogResult(
        catalog_type=catalog_type,
        query=query,
        total=len(docs),
        entries=docs,
        message=f"found {len(docs)} {catalog_type} catalog entries",
    )


# --- JSON Tools ---


@mcp.tool(title="Read JSON File")
def json_reader(file_name: str) -> str:
    """Reads a JSON file, parses its content, and returns the parsed data."""
    try:
        with open(file_name, "r") as fp:
            contents = json.load(fp)
        return json.dumps(contents)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_name}: {e}")
        return json.dumps({"error": str(e)})


# --- Catalog Tools ---


@mcp.tool(title="Get Sensor Catalog")
def get_sensor_catalog(
    sensor: Optional[str] = None,
) -> Union[CatalogResult, ErrorResult]:
    """Return cataloged sensor types.

    Entries contain `sensor` and `description`. Omit sensor to list all cataloged
    sensor types, or pass sensor for an exact sensor-name lookup.
    """
    return _find_catalog(
        catalog_type="sensor",
        field="sensor",
        value=sensor,
        fields=["sensor", "description"],
    )


@mcp.tool(title="Get Asset Catalog")
def get_asset_catalog(
    asset: Optional[str] = None,
    category: Optional[str] = None,
) -> Union[CatalogResult, ErrorResult]:
    """Return cataloged asset classes and categories.

    Entries contain `category`, `category_description`, `asset`, and
    `description`. Omit filters to list all cataloged asset classes, or pass
    asset and/or category for exact lookups.
    """
    return _find_catalog(
        catalog_type="asset",
        field="asset",
        value=asset,
        category=category,
        fields=["category", "category_description", "asset", "description"],
    )


@mcp.tool(title="Get Failure Mode Catalog")
def get_failure_mode_catalog(
    failure_mode: Optional[str] = None,
    category: Optional[str] = None,
) -> Union[CatalogResult, ErrorResult]:
    """Return cataloged failure modes by asset category.

    Entries contain `category`, `failure_mode`, and `description`. Omit filters
    to list all cataloged failure modes, or pass failure_mode and/or category
    for exact lookups.
    """
    return _find_catalog(
        catalog_type="failure_mode",
        field="failure_mode",
        value=failure_mode,
        category=category,
        fields=["category", "failure_mode", "description"],
    )


# --- Time Tools ---


@mcp.tool(title="Get Current Date and Time")
def current_date_time() -> DateTimeResult:
    """Provides the current date time as a JSON object."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat().replace("+00:00", "Z")

    date_part = now_iso.split("T")[0]
    time_part = now_iso.split("T")[1].split(".")[0]

    description = f"Today's date is {date_part} and time is {time_part}."

    return DateTimeResult(
        currentDateTime=now_iso, currentDateTimeDescription=description
    )


@mcp.tool(title="Get Current Time in English")
def current_time_english() -> TimeEnglishResult:
    """Returns the current time in English text."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat().replace("+00:00", "Z")

    dt = pendulum.parse(now_iso)
    eng = dt.to_datetime_string()

    return TimeEnglishResult(english=eng, iso=now_iso)


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
