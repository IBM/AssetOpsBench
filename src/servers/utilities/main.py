import logging
import os
from typing import Any, Optional, Union

import couchdb3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from servers.db_errors import DATA_UNAVAILABLE, db_failure_text

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


def get_temp_filename() -> str:
    tmpdir = tempfile.gettempdir()
    tmppath = Path(tmpdir)
    basepath = Path("cbmdir")
    filename = str(uuid4())

    tmpdir_path = tmppath / basepath
    tmpdir_path.mkdir(parents=True, exist_ok=True)

    filepath = tmpdir_path / (filename + ".json")
    return str(filepath)


def _missing_db_error() -> Optional[ErrorResult]:
    """Return an error when the catalog database is absent or unreachable, so it
    is not reported as a CouchDB error that names the database or host."""
    if catalog_db is not None and catalog_db.check():
        return None
    return ErrorResult(error=DATA_UNAVAILABLE)

def _clean_filter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _catalog_field_present(discriminator: str, name: str) -> bool:
    """True when at least one doc in this catalog carries `name`.

    The CSV loader drops empty cells rather than storing "", so a column that is
    blank for every row leaves no key behind. Filtering on it then matches nothing
    and is indistinguishable from a value that simply is not catalogued.
    """
    try:
        res = catalog_db.find(
            {discriminator: {"$exists": True}, name: {"$exists": True}},
            fields=[name],
            limit=1,
        )
        return bool(res.get("docs"))
    except Exception:
        return True


def _find_catalog(
    *,
    catalog_type: str,
    field: str,
    value: Optional[str],
    fields: list[str],
    category: Optional[str] = None,
) -> Union[CatalogResult, ErrorResult]:
    missing = _missing_db_error()
    if missing:
        return missing

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
        return _missing_db_error() or ErrorResult(error=db_failure_text(e))

    query_parts = []
    if query_value is not None:
        query_parts.append(query_value)
    if category_value is not None:
        query_parts.append(f"category={category_value}")
    query = ", ".join(query_parts) if query_parts else None

    message = f"found {len(docs)} {catalog_type} catalog entries"
    if not docs and (query_value is not None or category_value is not None):
        unserviceable = [
            name
            for name, applied in (
                (field, query_value is not None),
                ("category", category_value is not None),
            )
            if applied and not _catalog_field_present(field, name)
        ]
        if unserviceable:
            message += (
                f"; the loaded {catalog_type} catalog records no "
                + ", ".join(f"`{name}`" for name in unserviceable)
                + " value for any entry, so filtering on it cannot match"
            )

    return CatalogResult(
        catalog_type=catalog_type,
        query=query,
        total=len(docs),
        entries=docs,
        message=message,
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
    """Return cataloged failure modes.

    Entries contain `category`, `failure_mode`, and `description`. Omit filters
    to list all cataloged failure modes, or pass failure_mode and/or category
    for exact lookups. Whether entries are grouped by asset category depends on
    the loaded catalog: when it records no category, the category filter matches
    nothing and the result says so.
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
