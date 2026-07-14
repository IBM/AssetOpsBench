import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Union

import couchdb3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

# Setup logging; default WARNING keeps stderr quiet when used as an MCP server.
_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("iot-mcp-server")

COUCHDB_URL = os.environ.get("COUCHDB_URL")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")
IOT_DBNAME = os.environ.get("IOT_DBNAME", "iot")
ASSET_DBNAME = os.environ.get("ASSET_DBNAME", "asset")

try:
    db = couchdb3.Database(
        IOT_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info(f"Connected to IoT records database: {IOT_DBNAME}")
except Exception as e:
    logger.error(f"Failed to connect to IoT records database: {e}")
    db = None

try:
    asset_db = couchdb3.Database(
        ASSET_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info(f"Connected to asset registry database: {ASSET_DBNAME}")
except Exception as e:
    logger.error(f"Failed to connect to asset registry database: {e}")
    asset_db = None

mcp = FastMCP(
    "iot",
    instructions=(
        "IoT asset registry and telemetry record tools. Use sites() to discover site names, "
        "asset_ids() for bare assetnum values at a site, assets() for registry metadata with "
        "optional assettype filtering, installed_sensors() for registry sensor inventory, "
        "and measured_sensors() for observed telemetry fields."
    ),
)

DEFAULT_SITES = ["MAIN"]
PAGE_SIZE = 1000
RESERVED_FIELDS = {"_id", "_rev", "asset_id", "timestamp", "dataset", "type", "doctype"}


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


class AssetSummary(BaseModel):
    asset_id: str
    description: Optional[str]
    assettype: Optional[str]
    vintage: Optional[str]
    n_sensors: int


class AssetsWithMetadataResult(BaseModel):
    site_name: str
    total_assets: int
    assets: List[AssetSummary]
    message: str


_registry_sites_cache: Optional[List[str]] = None
_sensor_list_cache: Dict[str, List[str]] = {}


def _iter_records(
    selector: Dict[str, Any],
    fields: Optional[List[str]] = None,
    sort: Optional[List[Dict[str, str]]] = None,
    page_size: int = PAGE_SIZE,
) -> Iterator[Dict[str, Any]]:
    """Yield telemetry records matching a selector across paged database results."""
    if not db:
        return
    if sort is None:
        sort = [{"asset_id": "asc"}, {"timestamp": "asc"}]
    bookmark: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"limit": page_size, "sort": sort}
        if fields is not None:
            kwargs["fields"] = fields
        if bookmark is not None:
            kwargs["bookmark"] = bookmark
        res = db.find(selector, **kwargs)
        docs = res.get("docs", [])
        if not docs:
            break
        for doc in docs:
            yield doc
        bookmark = res.get("bookmark")
        if bookmark is None or len(docs) < page_size:
            break


def get_sensor_list(asset_id: str) -> List[str]:
    """Return sorted telemetry field names observed across all records for an asset."""
    if asset_id in _sensor_list_cache:
        return _sensor_list_cache[asset_id]
    if not db:
        return []
    try:
        found = set()
        seen = False
        for doc in _iter_records({"asset_id": asset_id}):
            seen = True
            found.update(key for key in doc.keys() if key not in RESERVED_FIELDS)
        if not seen:
            return []
        sensors = sorted(found)
        _sensor_list_cache[asset_id] = sensors
        return sensors
    except Exception as e:
        logger.error(f"get_sensor_list failed for asset_id {asset_id}: {e}")
        return []


def get_registry_sites() -> List[str]:
    """Return distinct site ids present in the asset registry."""
    global _registry_sites_cache
    if _registry_sites_cache is not None:
        return _registry_sites_cache
    if not asset_db:
        return []
    try:
        res = asset_db.find(
            {"siteid": {"$exists": True}},
            fields=["siteid"],
            limit=100000,
        )
        _registry_sites_cache = sorted(
            {doc.get("siteid") for doc in res["docs"] if doc.get("siteid")}
        )
        return _registry_sites_cache
    except Exception as e:
        logger.error(f"get_registry_sites failed: {e}")
        return []


def known_sites() -> List[str]:
    """Return known registry sites, falling back to MAIN when the registry is unavailable."""
    return get_registry_sites() or DEFAULT_SITES


def _is_known_site(site_name: str) -> bool:
    return site_name in known_sites()


@mcp.tool(title="List Sites")
def sites() -> SitesResult:
    """List known site names from the asset registry.

    Returns:
        SitesResult: `sites` contains sorted distinct `siteid` values from
        asset profiles. If the registry database is unavailable or has no sites,
        `sites` falls back to `["MAIN"]`.
    """
    return SitesResult(sites=known_sites())


@mcp.tool(title="List Asset IDs")
def asset_ids(site_name: str) -> Union[AssetsResult, ErrorResult]:
    """List only asset identifiers for one site.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.

    Returns:
        AssetsResult: Contains `site_name`, `total_assets`, `assets`, and
        `message`. The `assets` field is a sorted list of asset registry
        `assetnum` values.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="asset registry database not connected")
    try:
        res = asset_db.find(
            {"siteid": site_name},
            fields=["assetnum"],
            limit=100000,
        )
        ids = sorted(doc["assetnum"] for doc in res["docs"])
        return AssetsResult(
            site_name=site_name,
            total_assets=len(ids),
            assets=ids,
            message=f"found {len(ids)} assets at site {site_name}: {', '.join(ids)}.",
        )
    except Exception as e:
        logger.error(f"asset_ids failed: {e}")
        return ErrorResult(error=str(e))


@mcp.tool(title="List Measured Sensors")
def measured_sensors(
    site_name: str, asset_id: str
) -> Union[SensorsResult, ErrorResult]:
    """List measured sensor fields for one asset at one site.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorsResult: Contains `site_name`, `asset_id`, `total_sensors`,
        `sensors`, and `message`. The `sensors` field is a sorted list of
        telemetry record fields observed for the asset, excluding structural
        fields such as `asset_id`, `timestamp`, and `dataset`.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not db:
        return ErrorResult(error="IoT records database not connected")

    sensor_list = get_sensor_list(asset_id)
    if not sensor_list:
        return ErrorResult(error=f"unknown asset_id {asset_id} or no sensors found")

    return SensorsResult(
        site_name=site_name,
        asset_id=asset_id,
        total_sensors=len(sensor_list),
        sensors=sensor_list,
        message=(
            f"found {len(sensor_list)} sensors for asset_id {asset_id} "
            f"and site_name {site_name}: {', '.join(sensor_list)}."
        ),
    )


@mcp.tool(title="List Installed Sensors")
def installed_sensors(
    site_name: str, asset_id: str
) -> Union[SensorsResult, ErrorResult]:
    """List installed sensor names for one asset from the asset registry.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorsResult: Contains `site_name`, `asset_id`, `total_sensors`,
        `sensors`, and `message`. The `sensors` field is the registry
        inventory for the asset, distinct from `measured_sensors()`, which
        lists fields observed in IoT telemetry records.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="asset registry database not connected")

    try:
        res = asset_db.find(
            {"siteid": site_name, "assetnum": asset_id},
            fields=["assetnum", "sensors"],
            limit=1,
        )
        docs = res.get("docs", [])
        if not docs:
            return ErrorResult(error=f"unknown asset_id {asset_id} at site {site_name}")
        names = list(docs[0].get("sensors", []))
        return SensorsResult(
            site_name=site_name,
            asset_id=asset_id,
            total_sensors=len(names),
            sensors=names,
            message=f"{len(names)} sensors installed on {asset_id}: {', '.join(names)}.",
        )
    except Exception as e:
        logger.error(f"installed_sensors failed: {e}")
        return ErrorResult(error=str(e))


@mcp.tool(title="List Assets")
def assets(
    site_name: str, assettype: Optional[str] = None
) -> Union[AssetsWithMetadataResult, ErrorResult]:
    """List asset registry records for one site with compact metadata.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        assettype: Optional exact asset type filter, such as `PUMP` or
            `COMPRESSOR`.

    Returns:
        AssetsWithMetadataResult: Contains `site_name`, `total_assets`,
        `assets`, and `message`. Each item in `assets` includes `asset_id`
        (the registry `assetnum`), `description`, `assettype`, `vintage`, and
        `n_sensors`.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="asset registry database not connected")
    try:
        selector: Dict[str, Any] = {"siteid": site_name}
        if assettype:
            selector["assettype"] = assettype
        res = asset_db.find(
            selector,
            fields=["assetnum", "assettype", "description", "vintage", "sensors"],
            limit=100000,
        )
        rows = sorted(
            (
                AssetSummary(
                    asset_id=doc["assetnum"],
                    assettype=doc.get("assettype"),
                    description=doc.get("description"),
                    vintage=doc.get("vintage"),
                    n_sensors=len(doc.get("sensors", [])),
                )
                for doc in res["docs"]
            ),
            key=lambda row: row.asset_id,
        )
        return AssetsWithMetadataResult(
            site_name=site_name,
            total_assets=len(rows),
            assets=rows,
            message=f"found {len(rows)} assets"
            + (f" of type '{assettype}'" if assettype else "")
            + ".",
        )
    except Exception as e:
        logger.error(f"assets failed: {e}")
        return ErrorResult(error=str(e))


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
