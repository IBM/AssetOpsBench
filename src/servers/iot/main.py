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

# The asset registry is loaded as its own collection (manifest key "asset"), and the loader makes
# database name == collection key — so it lives in the "asset" database, NOT in IOT_DBNAME. Open a
# second handle for it. Telemetry (assets/sensors/history) keeps using `db` (the iot readings DB).
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
        "IoT sensor data + asset registry. Browse sites, assets, and sensors, read the asset "
        "nameplate (registry), see which installed sensors are actually measured (streaming), and "
        "query historical readings from CouchDB. NOTE: assets()/sensors() reflect TELEMETRY (what "
        "streams = measured); get_asset()/asset_sensors()/registry_assets() reflect the REGISTRY "
        "(what is installed, by name). Compare the two to find installed-but-not-streaming sensors."
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

# ── Asset-registry result models (identity / nameplate + installed sensor names) ──
class AssetDetail(BaseModel):
    site_name: str
    asset_id: str
    description: Optional[str]
    assettype: Optional[str]
    status: Optional[str]
    location: Optional[str]
    installdate: Optional[str]
    vintage: Optional[str]
    n_sensors: int
    message: str


class AssetSensorsResult(BaseModel):
    site_name: str
    asset_id: str
    total_sensors: int
    sensors: List[str]
    message: str


class RegistryAssetsResult(BaseModel):
    site_name: str
    total_assets: int
    assets: List[Dict[str, Any]]
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
    """The sensors an asset actually measures = the UNION of measurement keys across ALL of the
    asset's reading documents.
 
    IoT data may be sparse / non-uniform: different sensors are recorded at different timestamps
    (a timestamp may carry one sensor, several, or all), so a single document does NOT reveal the
    full measured set. We therefore scan every reading doc for the asset and union the non-metadata
    keys. Result is cached per asset_id after the first successful call."""
    if asset_id in _sensor_list_cache:
        return _sensor_list_cache[asset_id]
 
    if not db:
        return []
 
    try:
        res = db.find({"asset_id": asset_id}, limit=100000)
        docs = res["docs"]
        if not docs:
            return []
 
        # Exclude metadata; union the measurement keys across every reading document.
        exclude = {"_id", "_rev", "asset_id", "timestamp"}
        sensors = sorted({key for doc in docs for key in doc.keys() if key not in exclude})
        _sensor_list_cache[asset_id] = sensors
        return sensors
    except Exception as e:
        logger.error(f"Error fetching sensors for {asset_id}: {e}")
        return []

_asset_doc_cache: Dict[str, Dict[str, Any]] = {}


def get_asset_doc(asset_id: str) -> Optional[Dict[str, Any]]:
    """Helper to fetch one asset-registry document (doctype 'asset', _id 'asset:<assetnum>')
    by assetnum. Cached per asset_id. The registry holds identity/nameplate + the INSTALLED
    sensor inventory, separate from the telemetry reading docs."""
    if asset_id in _asset_doc_cache:
        return _asset_doc_cache[asset_id]
    if not asset_db:
        return None
    try:
        res = asset_db.find({"doctype": "asset", "assetnum": asset_id}, limit=1)
        docs = res["docs"]
        if not docs:
            return None
        _asset_doc_cache[asset_id] = docs[0]
        return docs[0]
    except Exception as e:
        logger.error(f"Error fetching asset doc {asset_id}: {e}")
        return None

_registry_sites_cache: Optional[List[str]] = None
 
 
def get_registry_sites() -> List[str]:
    """Distinct site ids present in the asset registry (from each asset profile's `siteid`). Cached."""
    global _registry_sites_cache
    if _registry_sites_cache is not None:
        return _registry_sites_cache
    if not asset_db:
        return []
    try:
        res = asset_db.find({"doctype": "asset"}, fields=["siteid"], limit=100000)
        found = sorted({d.get("siteid") for d in res["docs"] if d.get("siteid")})
        _registry_sites_cache = found
        return found
    except Exception as e:
        logger.error(f"get_registry_sites failed: {e}")
        return []
 
 
def known_sites() -> List[str]:
    """The server's site list — discovered DYNAMICALLY from the asset registry (each asset profile's
    `siteid`). Falls back to DEFAULT_SITES only if the registry is empty / unavailable."""
    return get_registry_sites() or DEFAULT_SITES
 
 
def _is_known_site(site_name: str) -> bool:
    return site_name in known_sites()
 

@mcp.tool(title="List Sites")
def sites() -> SitesResult:
    """Retrieves the list of sites, discovered dynamically from the asset registry (the distinct
    `siteid` across asset profiles). Falls back to the default only if the registry has no assets."""
    return SitesResult(sites=known_sites())


@mcp.tool(title="List Assets")
def assets(site_name: str) -> Union[AssetsResult, ErrorResult]:
    """Returns a list of assets for a given site. Each asset includes an id and a name."""
    if not _is_known_site(site_name):
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
    """Lists the sensors available for a specified asset at a given site.
    These are the MEASURED sensors — names discovered from the asset's telemetry documents,
    i.e. points that actually stream to the historian. For the full INSTALLED inventory
    (including sensors fitted but not streaming), use asset_sensors()."""
    if not _is_known_site(site_name):
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

@mcp.tool(title="Get Asset")
def get_asset(site_name: str, asset_id: str) -> Union[AssetDetail, ErrorResult]:
    """Return registry/nameplate details for one asset (Maximo MXASSET-aligned: description,
    assettype, status, location, installdate, vintage) plus installed sensor count.
    This is asset IDENTITY — distinct from the telemetry-derived assets() list."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    doc = get_asset_doc(asset_id)
    if not doc:
        return ErrorResult(error=f"unknown asset_id {asset_id} in registry")
    n = len(doc.get("sensors", []))
    return AssetDetail(
        site_name=site_name,
        asset_id=doc.get("assetnum", asset_id),
        description=doc.get("description"),
        assettype=doc.get("assettype"),
        status=doc.get("status"),
        location=doc.get("location"),
        installdate=doc.get("installdate"),
        vintage=doc.get("vintage"),
        n_sensors=n,
        message=(
            f"asset {asset_id} is a {doc.get('assettype')} "
            f"({doc.get('vintage')} vintage) at {doc.get('location')} with {n} installed sensors."
        ),
    )


@mcp.tool(title="List Asset Sensors")
def asset_sensors(site_name: str, asset_id: str) -> Union[AssetSensorsResult, ErrorResult]:
    """List the INSTALLED sensors for an asset, by name (installed is assumed). This is the registry
    inventory — distinct from sensors(), which lists only what actually streams (the MEASURED set).
    Compare the two to find installed-but-not-streaming sensors."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    doc = get_asset_doc(asset_id)
    if not doc:
        return ErrorResult(error=f"unknown asset_id {asset_id} in registry")
    names = list(doc.get("sensors", []))
    return AssetSensorsResult(
        site_name=site_name,
        asset_id=asset_id,
        total_sensors=len(names),
        sensors=names,
        message=f"{len(names)} sensors installed on {asset_id}: {', '.join(names)}.",
    )


@mcp.tool(title="List Registry Assets")
def registry_assets(
    site_name: str, assettype: Optional[str] = None
) -> Union[RegistryAssetsResult, ErrorResult]:
    """List assets from the registry with metadata (assettype, vintage, sensor count), optionally
    filtered by assettype (e.g. 'PUMP', 'COMPRESSOR'). Complements assets(), which returns bare ids derived from
    telemetry."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="CouchDB not connected")
    try:
        selector: Dict[str, Any] = {"doctype": "asset"}
        if assettype:
            selector["assettype"] = assettype
        res = asset_db.find(
            selector,
            fields=["assetnum", "assettype", "vintage", "sensors"],
            limit=100000,
        )
        rows = sorted(
            (
                {
                    "asset_id": d["assetnum"],
                    "assettype": d.get("assettype"),
                    "vintage": d.get("vintage"),
                    "n_sensors": len(d.get("sensors", [])),
                }
                for d in res["docs"]
            ),
            key=lambda r: r["asset_id"],
        )
        return RegistryAssetsResult(
            site_name=site_name,
            total_assets=len(rows),
            assets=rows,
            message=f"found {len(rows)} registry assets"
            + (f" of type '{assettype}'" if assettype else "")
            + ".",
        )
    except Exception as e:
        logger.error(f"registry_assets failed: {e}")
        return ErrorResult(error=str(e))



def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
