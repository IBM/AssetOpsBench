import os
import logging
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import couchdb3
from dotenv import load_dotenv

load_dotenv()

# Setup logging — default WARNING so stderr stays quiet when used as MCP server;
# set LOG_LEVEL=INFO (or DEBUG) in the environment to see verbose output.
_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("iot-mcp-server")

# Configuration from environment
COUCHDB_URL = os.environ.get("COUCHDB_URL")
COUCHDB_DBNAME = os.environ.get("IOT_DBNAME")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")

# Initialize CouchDB
try:
    db = couchdb3.Database(
        COUCHDB_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info(f"Connected to CouchDB: {COUCHDB_DBNAME}")
except Exception as e:
    logger.error(f"Failed to connect to CouchDB: {e}")
    db = None

mcp = FastMCP("iot")

# Static site as per original requirement
SITES = ["MAIN"]


class ErrorResult(BaseModel):
    error: str


class SitesResult(BaseModel):
    sites: List[str]


class AssetsResult(BaseModel):
    site_name: str
    total_assets: int
    assets: List[str]
    message: str


class SensorsResult(BaseModel):
    site_name: str
    asset_id: str
    total_sensors: int
    sensors: List[str]
    message: str


class HistoryResult(BaseModel):
    site_name: str
    asset_id: str
    total_observations: int
    start: str
    final: Optional[str]
    observations: List[Dict[str, Any]]
    message: str


_asset_list_cache: Optional[List[str]] = None


def get_asset_list() -> List[str]:
    """Helper to fetch unique asset IDs from CouchDB.  Result is cached after
    the first successful call to avoid repeated full-table scans."""
    global _asset_list_cache
    if _asset_list_cache is not None:
        return _asset_list_cache

    if not db:
        return []

    try:
        # We limit the fields to just asset_id to minimize data transfer
        res = db.find(
            {"asset_id": {"$exists": True}}, fields=["asset_id"], limit=100000
        )
        assets = {doc["asset_id"] for doc in res["docs"] if "asset_id" in doc}
        _asset_list_cache = sorted(list(assets))
        return _asset_list_cache
    except Exception as e:
        logger.error(f"Error fetching assets: {e}")
        return []


_sensor_list_cache: Dict[str, List[str]] = {}


def get_sensor_list(asset_id: str) -> List[str]:
    """Helper to fetch sensor names for a given asset from CouchDB.
    Result is cached per asset_id after the first successful call."""
    if asset_id in _sensor_list_cache:
        return _sensor_list_cache[asset_id]

    if not db:
        return []

    try:
        # Get one document for the asset to inspect keys
        res = db.find({"asset_id": asset_id}, limit=1)
        if not res["docs"]:
            return []

        doc = res["docs"][0]
        # Exclude metadata and standard fields
        exclude = {"_id", "_rev", "asset_id", "timestamp"}
        sensors = sorted(key for key in doc.keys() if key not in exclude)
        _sensor_list_cache[asset_id] = sensors
        return sensors
    except Exception as e:
        logger.error(f"Error fetching sensors for {asset_id}: {e}")
        return []


@mcp.tool(title="List Sites")
def sites() -> SitesResult:
    """Retrieves a list of sites. Each site is represented by a name."""
    return SitesResult(sites=SITES)


@mcp.tool(title="List Assets")
def assets(site_name: str) -> Union[AssetsResult, ErrorResult]:
    """Returns a list of assets for a given site. Each asset includes an id and a name."""
    if site_name not in SITES:
        return ErrorResult(error=f"unknown site {site_name}")

    asset_list = get_asset_list()
    return AssetsResult(
        site_name=site_name,
        total_assets=len(asset_list),
        assets=asset_list,
        message=f"found {len(asset_list)} asset ids for site_name {site_name}: {', '.join(asset_list)}.",
    )


@mcp.tool(title="List Sensors")
def sensors(site_name: str, asset_id: str) -> Union[SensorsResult, ErrorResult]:
    """Lists the sensors available for a specified asset at a given site."""
    if site_name not in SITES:
        return ErrorResult(error=f"unknown site {site_name}")

    sensor_list = get_sensor_list(asset_id)
    if not sensor_list:
        return ErrorResult(error=f"unknown asset_id {asset_id} or no sensors found")

    return SensorsResult(
        site_name=site_name,
        asset_id=asset_id,
        total_sensors=len(sensor_list),
        sensors=sensor_list,
        message=f"found {len(sensor_list)} sensors for asset_id {asset_id} and site_name {site_name}: {', '.join(sensor_list)}.",
    )


@mcp.tool(title="Get Sensor History")
def history(
    site_name: str, asset_id: str, start: str, final: Optional[str] = None
) -> Union[HistoryResult, ErrorResult]:
    """Returns a list of historical sensor values for the specified asset(s) at a site within a given time range (start to final)."""
    try:
        start_iso = datetime.fromisoformat(start).isoformat()
        if final:
            datetime.fromisoformat(final)
            if start >= final:
                return ErrorResult(error="start >= final")
    except ValueError as e:
        return ErrorResult(error=f"Invalid date format: {e}")

    if not db:
        return ErrorResult(error="CouchDB not connected")

    selector = {
        "asset_id": asset_id,
        "timestamp": {"$gte": start_iso},
    }
    if final:
        selector["timestamp"]["$lt"] = datetime.fromisoformat(final).isoformat()

    logger.info(f"Querying CouchDB with selector: {selector}")
    try:
        res = db.find(
            selector, limit=1000, sort=[{"asset_id": "asc"}, {"timestamp": "asc"}]
        )
        docs = res["docs"]
        return HistoryResult(
            site_name=site_name,
            asset_id=asset_id,
            total_observations=len(docs),
            start=start,
            final=final,
            observations=docs,
            message=f"found {len(docs)} observations for asset_id {asset_id} from {start} to {final or 'now'}.",
        )
    except Exception as e:
        logger.error(f"CouchDB query failed: {e}")
        return ErrorResult(error=str(e))


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
