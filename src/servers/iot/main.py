import os
import logging
import statistics
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
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
        "query historical readings from CouchDB. NOTE: asset_ids()/measured_sensors() reflect TELEMETRY "
        "(what streams = measured); get_asset_detail()/installed_sensors()/assets() reflect the REGISTRY "
        "(what is installed, by name). Compare the two to find installed-but-not-streaming sensors."
    ),
)

DEFAULT_SITES = ["MAIN"]

# Fields on a reading document that are structural, not sensor channels.
RESERVED_FIELDS = {"_id", "_rev", "asset_id", "timestamp"}

# CouchDB per-request page cap used by history(); new paging tools respect it too.
PAGE_SIZE = 1000


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


# ── New tool result models ──
class AssetSensorMatch(BaseModel):
    asset_id: str
    matched_sensors: List[str]


class FindAssetsResult(BaseModel):
    site_name: str
    query_sensors: List[str]
    match: str
    source: str
    total_assets: int
    assets: List[AssetSensorMatch]
    message: str


class SensorCoverage(BaseModel):
    sensor: str
    non_null_count: int
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]


class SensorCoverageResult(BaseModel):
    site_name: str
    asset_id: str
    docs_scanned: int
    sensors: List[SensorCoverage]
    message: str


class StreamExtentResult(BaseModel):
    site_name: str
    asset_id: str
    sensor: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    total_records: int
    exceeds_page_limit: bool
    approx_interval_seconds: Optional[float]
    message: str


class PagedHistoryResult(BaseModel):
    site_name: str
    asset_id: str
    total_in_page: int
    observations: List[Dict[str, Any]]
    next_bookmark: Optional[str]
    has_more: bool
    message: str


class SensorStatsResult(BaseModel):
    site_name: str
    asset_id: str
    sensor: str
    count: int
    null_count: int
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    stddev: Optional[float]
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    message: str


class LatestReadingResult(BaseModel):
    site_name: str
    asset_id: str
    timestamp: Optional[str]
    values: Dict[str, Any]
    age_seconds: Optional[float]
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
        sensors = sorted(
            {key for doc in docs for key in doc.keys() if key not in exclude}
        )
        _sensor_list_cache[asset_id] = sensors
        return sensors
    except Exception as e:
        logger.error(f"Error fetching sensors for {asset_id}: {e}")
        return []


_asset_doc_cache: Dict[str, Dict[str, Any]] = {}


def get_asset_doc(asset_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one asset-registry document, resolving ANY of the asset's id spaces — the registry
    `assetnum` (Maximo id), the telemetry `iot_asset_id`, or the work-order `wo_assetnum`. This lets
    the same profile be found whether the caller holds the IoT id (e.g. 'Chiller 6') or the WO id
    ('CHILLER6'). Cached per asset_id."""
    if asset_id in _asset_doc_cache:
        return _asset_doc_cache[asset_id]
    if not asset_db:
        return None
    try:
        for field in ("assetnum", "iot_asset_id", "wo_assetnum"):
            res = asset_db.find({"doctype": "asset", field: asset_id}, limit=1)
            docs = res["docs"]
            if docs:
                _asset_doc_cache[asset_id] = docs[0]
                return docs[0]
        return None
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
    `siteid`). Falls back to DEFAULT_SITES only if the registry is empty / unavailable.
    """
    return get_registry_sites() or DEFAULT_SITES


def _is_known_site(site_name: str) -> bool:
    return site_name in known_sites()


# ---------------------------------------------------------------------------
# Shared helpers for the new tools
# ---------------------------------------------------------------------------
def _validate_dates(start: Optional[str], final: Optional[str]) -> Optional[str]:
    """Return None if ok, else an error message. None inputs are allowed."""
    try:
        if start is not None:
            datetime.fromisoformat(start)
        if final is not None:
            datetime.fromisoformat(final)
    except ValueError as e:
        return f"Invalid date format: {e}"
    if start is not None and final is not None and start >= final:
        return "start >= final"
    return None


def _time_selector(
    asset_id: str, start: Optional[str], final: Optional[str]
) -> Dict[str, Any]:
    selector: Dict[str, Any] = {"asset_id": asset_id}
    ts: Dict[str, Any] = {}
    if start is not None:
        ts["$gte"] = datetime.fromisoformat(start).isoformat()
    if final is not None:
        ts["$lt"] = datetime.fromisoformat(final).isoformat()
    if ts:
        selector["timestamp"] = ts
    return selector


def _iter_docs(
    selector: Dict[str, Any],
    fields: Optional[List[str]] = None,
    sort: Optional[List[Dict[str, str]]] = None,
    page_size: int = PAGE_SIZE,
    max_docs: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield all docs matching `selector`, transparently walking Mango bookmarks so callers are not
    limited to a single 1000-row page. Used by the new count/extent/stats tools."""
    if not db:
        return
    if sort is None:
        sort = [{"asset_id": "asc"}, {"timestamp": "asc"}]
    bookmark: Optional[str] = None
    yielded = 0
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
            yielded += 1
            if max_docs is not None and yielded >= max_docs:
                return
        bookmark = res.get("bookmark")
        if bookmark is None or len(docs) < page_size:
            break


def _latest_doc(
    asset_id: str, sensor: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the newest reading doc for an asset. Fast path uses a descending sort (needs a
    descending index); if that index does not exist, falls back to an ascending scan keeping the
    last doc — correct either way."""
    if not db:
        return None
    selector: Dict[str, Any] = {"asset_id": asset_id}
    if sensor:
        selector[sensor] = {"$exists": True, "$ne": None}
    try:
        res = db.find(
            selector, limit=1, sort=[{"asset_id": "desc"}, {"timestamp": "desc"}]
        )
        docs = res.get("docs", [])
        if docs:
            return docs[0]
        # empty result with a working sort => genuinely no matching docs
        return None
    except Exception as e:
        logger.info(f"descending sort unavailable, scanning ascending: {e}")
    last: Optional[Dict[str, Any]] = None
    for doc in _iter_docs(selector):
        last = doc
    return last


def _span_seconds(a_iso: str, b_iso: str) -> Optional[float]:
    try:
        return (
            datetime.fromisoformat(b_iso) - datetime.fromisoformat(a_iso)
        ).total_seconds()
    except ValueError:
        return None


def _age_seconds(ts_iso: str) -> Optional[float]:
    try:
        ts = datetime.fromisoformat(ts_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except ValueError:
        return None


def _site_asset_ids(site_name: str) -> List[str]:
    """Asset ids registered at a site (iot_asset_id where present, else assetnum) — mirrors asset_ids()."""
    if not asset_db:
        return []
    try:
        res = asset_db.find(
            {"doctype": "asset", "siteid": site_name},
            fields=["assetnum", "iot_asset_id"],
            limit=100000,
        )
        return sorted((d.get("iot_asset_id") or d.get("assetnum")) for d in res["docs"])
    except Exception as e:
        logger.error(f"_site_asset_ids failed: {e}")
        return []


def _installed_sensors(asset_id: str) -> List[str]:
    doc = get_asset_doc(asset_id)
    return list(doc.get("sensors", [])) if doc else []


# ---------------------------------------------------------------------------
# Original tools (unchanged)
# ---------------------------------------------------------------------------
@mcp.tool(title="List Sites")
def sites() -> SitesResult:
    """Retrieves the list of sites, discovered dynamically from the asset registry (the distinct
    `siteid` across asset profiles). Falls back to the default only if the registry has no assets.
    """
    return SitesResult(sites=known_sites())


@mcp.tool(title="List Asset IDs")
def asset_ids(site_name: str) -> Union[AssetsResult, ErrorResult]:
    """Returns the asset IDs registered at a given site, from the asset registry filtered by `siteid`.
    Each returned id is the asset's telemetry id (`iot_asset_id`) where it has one, otherwise its
    registry `assetnum` — so the id works with measured_sensors()/history() when telemetry exists.
    For assets with metadata (type, vintage, sensor count), use assets().
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="CouchDB not connected")
    try:
        res = asset_db.find(
            {"doctype": "asset", "siteid": site_name},
            fields=["assetnum", "iot_asset_id"],
            limit=100000,
        )
        ids = sorted((d.get("iot_asset_id") or d["assetnum"]) for d in res["docs"])
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
def measured_sensors(site_name: str, asset_id: str) -> Union[SensorsResult, ErrorResult]:
    """Lists the MEASURED sensors for a specified asset at a given site — names discovered from the
    asset's telemetry documents, i.e. points that actually stream to the historian. For the full
    INSTALLED inventory (including sensors fitted but not streaming), use installed_sensors()."""
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


@mcp.tool(title="Get Asset Detail")
def get_asset_detail(site_name: str, asset_id: str) -> Union[AssetDetail, ErrorResult]:
    """Return registry/nameplate details for one asset (Maximo MXASSET-aligned: description,
    assettype, status, location, installdate, vintage) plus installed sensor count.
    This is asset IDENTITY — distinct from the telemetry-derived asset_ids() list."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    doc = get_asset_doc(asset_id)
    if not doc:
        return ErrorResult(error=f"unknown asset_id {asset_id} in registry")
    n = len(doc.get("sensors", []))
    assettype = doc.get("assettype")
    vintage = doc.get("vintage")
    location = doc.get("location")

    # Build a human message that omits clauses whose fields are null.
    parts = [f"asset {asset_id} is a {assettype or 'asset'}"]
    if vintage:
        parts.append(f" ({vintage} vintage)")
    if location:
        parts.append(f" at {location}")
    parts.append(f" with {n} installed sensors.")
    message = "".join(parts)

    return AssetDetail(
        site_name=site_name,
        asset_id=doc.get("assetnum", asset_id),
        description=doc.get("description"),
        assettype=assettype,
        status=doc.get("status"),
        location=location,
        installdate=doc.get("installdate"),
        vintage=vintage,
        n_sensors=n,
        message=message,
    )


@mcp.tool(title="List Installed Sensors")
def installed_sensors(
    site_name: str, asset_id: str
) -> Union[AssetSensorsResult, ErrorResult]:
    """List the INSTALLED sensors for an asset, by name (installed is assumed). This is the registry
    inventory — distinct from measured_sensors(), which lists only what actually streams (the MEASURED
    set). Compare the two to find installed-but-not-streaming sensors."""
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


@mcp.tool(title="List Assets")
def assets(
    site_name: str, assettype: Optional[str] = None
) -> Union[RegistryAssetsResult, ErrorResult]:
    """List assets from the registry with metadata (assettype, vintage, sensor count), optionally
    filtered by assettype (e.g. 'PUMP', 'COMPRESSOR'). Complements asset_ids(), which returns bare
    ids derived from telemetry."""

    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="CouchDB not connected")
    try:
        selector: Dict[str, Any] = {"doctype": "asset", "siteid": site_name}
        if assettype:
            selector["assettype"] = assettype
        res = asset_db.find(
            selector,
            fields=["assetnum", "assettype", "description", "vintage", "sensors"],
            limit=100000,
        )
        rows = sorted(
            (
                {
                    "asset_id": d["assetnum"],
                    "assettype": d.get("assettype"),
                    "description": d.get("description"),
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
        logger.error(f"assets failed: {e}")
        return ErrorResult(error=str(e))


# ===========================================================================
# NEW TOOLS
# ===========================================================================
@mcp.tool(title="Find Assets By Sensors")
def find_assets_by_sensors(
    site_name: str,
    sensors: List[str],
    match: str = "all",
    substring: bool = False,
    source: str = "measured",
) -> Union[FindAssetsResult, ErrorResult]:
    """Return only the assets at a site that have the given sensor(s) — not all assets.
    match='all' requires every listed sensor; match='any' requires at least one.
    substring=True matches sensor names case-insensitively by substring.
    source='measured' checks telemetry (what streams, via measured_sensors()); source='installed'
    checks the registry nameplate (via installed_sensors())."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if match not in ("all", "any"):
        return ErrorResult(error="match must be 'all' or 'any'")
    if source not in ("measured", "installed"):
        return ErrorResult(error="source must be 'measured' or 'installed'")
    if not sensors:
        return ErrorResult(error="provide at least one sensor name")

    matches: List[AssetSensorMatch] = []
    for asset_id in _site_asset_ids(site_name):
        available = (
            get_sensor_list(asset_id)
            if source == "measured"
            else _installed_sensors(asset_id)
        )

        def _hits(w: str) -> List[str]:
            if substring:
                wl = w.lower()
                return [s for s in available if wl in s.lower()]
            return [s for s in available if s == w]

        per_query = {w: _hits(w) for w in sensors}
        satisfied = [w for w, hits in per_query.items() if hits]
        ok = len(satisfied) == len(sensors) if match == "all" else len(satisfied) > 0
        if ok:
            matched = sorted({s for hits in per_query.values() for s in hits})
            matches.append(AssetSensorMatch(asset_id=asset_id, matched_sensors=matched))

    return FindAssetsResult(
        site_name=site_name,
        query_sensors=sensors,
        match=match,
        source=source,
        total_assets=len(matches),
        assets=matches,
        message=f"{len(matches)} asset(s) at {site_name} match {sensors} "
        f"(match={match}, substring={substring}, source={source}).",
    )


@mcp.tool(title="Stream Extent")
def stream_extent(
    site_name: str,
    asset_id: str,
    sensor: Optional[str] = None,
    start: Optional[str] = None,
    final: Optional[str] = None,
) -> Union[StreamExtentResult, ErrorResult]:
    """Time bounds + record count for an asset's stream (optionally one sensor and/or a window),
    so callers can size a history() request and know if it exceeds the 1000-row page limit."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    err = _validate_dates(start, final)
    if err:
        return ErrorResult(error=err)
    if not db:
        return ErrorResult(error="CouchDB not connected")

    selector = _time_selector(asset_id, start, final)
    if sensor:
        selector[sensor] = {"$exists": True, "$ne": None}

    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    total = 0
    for doc in _iter_docs(selector, fields=["timestamp"]):
        ts = doc.get("timestamp")
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        total += 1

    if total == 0:
        return ErrorResult(
            error=f"no records for asset_id {asset_id}"
            + (f", sensor {sensor}" if sensor else "")
        )

    approx_interval: Optional[float] = None
    if first_ts and last_ts and total > 1:
        span = _span_seconds(first_ts, last_ts)
        if span is not None:
            approx_interval = span / (total - 1)

    return StreamExtentResult(
        site_name=site_name,
        asset_id=asset_id,
        sensor=sensor,
        start_time=first_ts,
        end_time=last_ts,
        total_records=total,
        exceeds_page_limit=total > PAGE_SIZE,
        approx_interval_seconds=approx_interval,
        message=f"{total} record(s) for asset_id {asset_id}"
        + (f" (sensor {sensor})" if sensor else "")
        + f" from {first_ts} to {last_ts}.",
    )


@mcp.tool(title="Get Sensor History (Paged)")
def history_paged(
    site_name: str,
    asset_id: str,
    start: str,
    final: Optional[str] = None,
    sensors: Optional[List[str]] = None,
    page_size: int = PAGE_SIZE,
    bookmark: Optional[str] = None,
) -> Union[PagedHistoryResult, ErrorResult]:
    """Like history() but paginated past the 1000-row limit via a CouchDB bookmark cursor.
    Pass `sensors` to project only those columns. Call repeatedly, feeding back next_bookmark,
    until has_more is false."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    err = _validate_dates(start, final)
    if err:
        return ErrorResult(error=err)
    if not db:
        return ErrorResult(error="CouchDB not connected")
    page_size = max(1, min(page_size, PAGE_SIZE))

    selector = _time_selector(asset_id, start, final)
    fields: Optional[List[str]] = None
    if sensors:
        fields = ["_id", "asset_id", "timestamp"] + list(sensors)

    kwargs: Dict[str, Any] = {
        "limit": page_size,
        "sort": [{"asset_id": "asc"}, {"timestamp": "asc"}],
    }
    if fields is not None:
        kwargs["fields"] = fields
    if bookmark is not None:
        kwargs["bookmark"] = bookmark

    try:
        res = db.find(selector, **kwargs)
    except Exception as e:
        logger.error(f"CouchDB query failed: {e}")
        return ErrorResult(error=str(e))

    docs = res.get("docs", [])
    next_bookmark = res.get("bookmark")
    has_more = len(docs) == page_size
    return PagedHistoryResult(
        site_name=site_name,
        asset_id=asset_id,
        total_in_page=len(docs),
        observations=docs,
        next_bookmark=next_bookmark if has_more else None,
        has_more=has_more,
        message=f"page of {len(docs)} observation(s) for asset_id {asset_id}; has_more={has_more}.",
    )


@mcp.tool(title="Sensor Coverage")
def sensor_coverage(
    site_name: str,
    asset_id: str,
    sample_limit: int = 5000,
) -> Union[SensorCoverageResult, ErrorResult]:
    """Per-measured-sensor record counts and time coverage (non-null count, first/last timestamp)
    for an asset. Complements measured_sensors(), which lists names but not how much data each channel has.
    sample_limit=0 scans all docs."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not db:
        return ErrorResult(error="CouchDB not connected")

    counts: Dict[str, int] = {}
    first_ts: Dict[str, str] = {}
    last_ts: Dict[str, str] = {}
    scanned = 0

    for doc in _iter_docs(
        {"asset_id": asset_id},
        sort=[{"asset_id": "asc"}, {"timestamp": "asc"}],
        max_docs=sample_limit if sample_limit > 0 else None,
    ):
        scanned += 1
        ts = doc.get("timestamp")
        for k, v in doc.items():
            if k in RESERVED_FIELDS or v is None:
                continue
            counts[k] = counts.get(k, 0) + 1
            if k not in first_ts and ts is not None:
                first_ts[k] = ts
            if ts is not None:
                last_ts[k] = ts

    if scanned == 0:
        return ErrorResult(error=f"unknown asset_id {asset_id} or no records found")

    cov = [
        SensorCoverage(
            sensor=s,
            non_null_count=counts[s],
            first_timestamp=first_ts.get(s),
            last_timestamp=last_ts.get(s),
        )
        for s in sorted(counts)
    ]
    return SensorCoverageResult(
        site_name=site_name,
        asset_id=asset_id,
        docs_scanned=scanned,
        sensors=cov,
        message=f"{len(cov)} sensor(s) with data for asset_id {asset_id} across {scanned} scanned docs.",
    )


@mcp.tool(title="Sensor Statistics")
def sensor_stats(
    site_name: str,
    asset_id: str,
    sensor: str,
    start: Optional[str] = None,
    final: Optional[str] = None,
) -> Union[SensorStatsResult, ErrorResult]:
    """Numeric summary (count/min/max/mean/stddev) for one sensor over an optional time window,
    without returning the raw rows."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    err = _validate_dates(start, final)
    if err:
        return ErrorResult(error=err)
    if not db:
        return ErrorResult(error="CouchDB not connected")

    selector = _time_selector(asset_id, start, final)
    selector[sensor] = {"$exists": True}

    values: List[float] = []
    null_count = 0
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    for doc in _iter_docs(selector, fields=["timestamp", sensor]):
        ts = doc.get("timestamp")
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        v = doc.get(sensor)
        if v is None:
            null_count += 1
            continue
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            null_count += 1

    if not values and null_count == 0:
        return ErrorResult(error=f"no records for asset_id {asset_id} sensor {sensor}")

    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return SensorStatsResult(
        site_name=site_name,
        asset_id=asset_id,
        sensor=sensor,
        count=len(values),
        null_count=null_count,
        min=min(values) if values else None,
        max=max(values) if values else None,
        mean=statistics.fmean(values) if values else None,
        stddev=stddev if values else None,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        message=f"{len(values)} value(s) for sensor {sensor} on asset_id {asset_id} "
        f"({null_count} null/non-numeric).",
    )


@mcp.tool(title="Latest Reading")
def latest_reading(
    site_name: str,
    asset_id: str,
    sensor: Optional[str] = None,
) -> Union[LatestReadingResult, ErrorResult]:
    """Most recent observation for an asset (all measured sensors, or just `sensor`), plus its age
    in seconds for staleness / offline checks."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not db:
        return ErrorResult(error="CouchDB not connected")

    doc = _latest_doc(asset_id, sensor=sensor)
    if not doc:
        return ErrorResult(error=f"unknown asset_id {asset_id} or no records found")

    ts = doc.get("timestamp")
    if sensor:
        values = {sensor: doc.get(sensor)}
    else:
        values = {k: v for k, v in doc.items() if k not in RESERVED_FIELDS}

    return LatestReadingResult(
        site_name=site_name,
        asset_id=asset_id,
        timestamp=ts,
        values=values,
        age_seconds=_age_seconds(ts) if ts else None,
        message=f"latest reading for asset_id {asset_id} at {ts}.",
    )


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
