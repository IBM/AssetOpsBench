import logging
import os
from typing import Any, Dict, List, Optional, Union

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
ASSET_DBNAME = os.environ.get("ASSET_DBNAME", "asset")

try:
    asset_db = couchdb3.Database(
        ASSET_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info(f"Connected to CouchDB: {ASSET_DBNAME}")
except Exception as e:
    logger.error(f"Failed to connect to asset registry DB: {e}")
    asset_db = None

mcp = FastMCP(
    "iot",
    instructions=(
        "IoT asset registry tools. Use sites() to discover site names, asset_ids() for bare "
        "assetnum values at a site, and assets() for registry metadata with optional "
        "assettype filtering."
    ),
)

DEFAULT_SITES = ["MAIN"]


class ErrorResult(BaseModel):
    error: str


class SitesResult(BaseModel):
    sites: List[str]


class AssetsResult(BaseModel):
    site_name: str
    total_assets: int
    assets: List[str]
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

    Sites are discovered from distinct `siteid` values in asset profiles. If the registry is
    unavailable or empty, the tool returns the default site list so callers still have a valid
    starting point for `asset_ids()` and `assets()`.
    """
    return SitesResult(sites=known_sites())


@mcp.tool(title="List Asset IDs")
def asset_ids(site_name: str) -> Union[AssetsResult, ErrorResult]:
    """List only the asset identifiers for a site.

    Use this lightweight lookup when you only need valid `assetnum` values, for example to
    populate a selector, validate an asset reference, or choose an id for another workflow.
    The response contains no metadata beyond `site_name`, `total_assets`, the sorted `assets`
    id list, and a human-readable summary. Use `assets()` instead when you need descriptions,
    asset types, vintages, or installed sensor counts.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="CouchDB not connected")
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


@mcp.tool(title="List Assets")
def assets(
    site_name: str, assettype: Optional[str] = None
) -> Union[AssetsWithMetadataResult, ErrorResult]:
    """List asset registry records for a site with compact metadata.

    Use this when you need more than ids: each row includes `asset_id` (the registry
    `assetnum`), `description`, `assettype`, `vintage`, and `n_sensors` derived from the
    installed sensor list. Pass `assettype` to restrict results to one asset class, such as
    `PUMP` or `COMPRESSOR`. Use `asset_ids()` instead when a compact list of ids is enough.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="CouchDB not connected")
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
