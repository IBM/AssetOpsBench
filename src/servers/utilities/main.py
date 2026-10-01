import logging
import os
from typing import Any, Optional, Union

import couchdb3
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
    instructions="Query asset, sensor, and failure-mode catalog data.",
)


class ErrorResult(BaseModel):
    error: str


class CatalogResult(BaseModel):
    catalog_type: str
    query: Optional[str]
    total: int
    entries: list[dict[str, Any]]
    message: str


# --- Helper Functions ---


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


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
