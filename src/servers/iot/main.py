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
        "query historical readings."
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
    observations: List[Dict[str, Any]]
    returned: int
    next_cursor: Optional[str]
    has_more: bool
    start: Optional[str]
    end: Optional[str]
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


class AssetSummary(BaseModel):
    asset_id: str
    description: Optional[str]
    assettype: Optional[str]
    vintage: Optional[str]
    n_sensors: int  # installed sensor count


class AssetsWithMetadataResult(BaseModel):
    site_name: str
    total_assets: int
    assets: List[AssetSummary]
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


class SensorStat(BaseModel):
    sensor: str
    count: int
    null_count: int
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    stddev: Optional[float]
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]


class SensorStatsResult(BaseModel):
    site_name: str
    asset_id: str
    stats: List[SensorStat]  # one entry if `sensor` given, else every measured sensor
    message: str


class LatestReadingResult(BaseModel):
    site_name: str
    asset_id: str
    timestamp: Optional[str]
    values: Dict[str, Any]
    age_seconds: Optional[float]
    message: str


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
        found: set = set()
        seen = False
        for doc in _iter_docs(
            {"asset_id": asset_id}
        ):  # walks all pages via Mango bookmark
            seen = True
            for key in doc.keys():
                if key not in RESERVED_FIELDS:
                    found.add(key)
        if not seen:
            return []
        sensors = sorted(found)
        _sensor_list_cache[asset_id] = sensors
        return sensors
    except Exception as e:
        logger.error(f"Error fetching sensors for {asset_id}: {e}")
        return []


_asset_doc_cache: Dict[str, Dict[str, Any]] = {}


def get_asset_doc(asset_id: str, site_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch one asset-registry document by assetnum, optionally constrained to a site so both
    conditions must hold. Cached per (asset_id, site_name)."""
    # impl: registry docs are keyed by `assetnum`, whose value equals the telemetry `asset_id`.
    cache_key = (asset_id, site_name)
    if cache_key in _asset_doc_cache:
        return _asset_doc_cache[cache_key]
    if not asset_db:
        return None
    try:
        selector: Dict[str, Any] = {"doctype": "asset", "assetnum": asset_id}
        if site_name is not None:
            selector["siteid"] = site_name
        res = asset_db.find(selector, limit=1)
        docs = res["docs"]
        doc = docs[0] if docs else None
        _asset_doc_cache[cache_key] = doc          # cache hit AND miss (asset is static per run)
        return doc
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
        res = asset_db.find({"doctype": "asset"}, fields=["siteid"], limit=10**12)
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
def _validate_dates(start: Optional[str], end: Optional[str]) -> Optional[str]:
    """Return None if ok, else an error message. None inputs are allowed."""
    try:
        if start is not None:
            datetime.fromisoformat(start)
        if end is not None:
            datetime.fromisoformat(end)
    except ValueError as e:
        return f"Invalid date format (expected ISO 8601, e.g. 2024-01-15T00:00:00): {e}"
    if start is not None and end is not None and start >= end:
        return "start >= end"
    return None


def _time_selector(
    asset_id: str, start: Optional[str], end: Optional[str]
) -> Dict[str, Any]:
    selector: Dict[str, Any] = {"asset_id": asset_id}
    ts: Dict[str, Any] = {}
    if start is not None:
        ts["$gte"] = datetime.fromisoformat(start).isoformat()
    if end is not None:
        ts["$lt"] = datetime.fromisoformat(end).isoformat()
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
    """Asset ids registered at a site — mirrors asset_ids()."""
    if not asset_db:
        return []
    try:
        res = asset_db.find(
            {"doctype": "asset", "siteid": site_name},
            fields=["assetnum"],
            limit=100000,
        )
        return sorted(d["assetnum"] for d in res["docs"])
    except Exception as e:
        logger.error(f"_site_asset_ids failed: {e}")
        return []

def _installed_sensors(asset_id: str, site_name: Optional[str] = None) -> List[str]:
    doc = get_asset_doc(asset_id, site_name)
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
    """List the asset IDs at a site. Use an ID from this list as the `asset_id` argument in the
    other tools (measured_sensors, history, stream_extent, ...).
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="CouchDB not connected")
    try:
        res = asset_db.find(
            {"doctype": "asset", "siteid": site_name},
            fields=["assetnum"],
            limit=100000,
        )
        ids = sorted(d["assetnum"] for d in res["docs"])
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
    """Lists the MEASURED sensors for a specified asset at a given site — names discovered from the
    asset's telemetry documents, i.e. points that actually stream to the historian.
    """
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
    site_name: str,
    asset_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sensors: Optional[List[str]] = None,
    limit: int = PAGE_SIZE,
    cursor: Optional[str] = None,
) -> Union[HistoryResult, ErrorResult]:
    """Return historical sensor readings for an asset, one page at a time.

    Timestamps are ISO 8601 strings, e.g. "2024-01-15T00:00:00" (offset allowed,
    "2024-01-15T00:00:00+00:00"). The range is half-open [start, end). Omit `start`/`end` to
    use the full available range; call stream_extent(site_name, asset_id) first to learn the
    exact bounds and total record count.

    Paging: `limit` = rows per page (<=1000). Leave `cursor` empty on the first call; each
    response returns `next_cursor` and `has_more` — repeat with cursor=next_cursor until
    `has_more` is false. Pass `sensors` to return only those columns.

    On a malformed timestamp or start >= end, returns {"error": ...}.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    err = _validate_dates(start, end)
    if err:
        return ErrorResult(error=err)
    if not db:
        return ErrorResult(error="CouchDB not connected")
    limit = max(1, min(limit, PAGE_SIZE))

    selector = _time_selector(asset_id, start, end)
    fields: Optional[List[str]] = None
    if sensors:
        fields = ["_id", "asset_id", "timestamp"] + list(sensors)

    kwargs: Dict[str, Any] = {
        "limit": limit,
        "sort": [{"asset_id": "asc"}, {"timestamp": "asc"}],
    }
    if fields is not None:
        kwargs["fields"] = fields
    if cursor is not None:
        kwargs["bookmark"] = cursor

    try:
        res = db.find(selector, **kwargs)
    except Exception as e:
        logger.error(f"CouchDB query failed: {e}")
        return ErrorResult(error=str(e))

    docs = res.get("docs", [])
    next_cursor = res.get("bookmark")
    has_more = len(docs) == limit
    return HistoryResult(
        site_name=site_name,
        asset_id=asset_id,
        observations=docs,
        returned=len(docs),
        next_cursor=next_cursor if has_more else None,
        has_more=has_more,
        start=start,
        end=end,
        message=f"{len(docs)} observation(s) for asset_id {asset_id}; has_more={has_more}.",
    )


@mcp.tool(title="Get Asset Detail")
def get_asset_detail(site_name: str, asset_id: str) -> Union[AssetDetail, ErrorResult]:
    """Return registry/nameplate details for one asset."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    doc = get_asset_doc(asset_id, site_name)
    if not doc:
        return ErrorResult(error=f"unknown asset_id {asset_id} at site {site_name}")
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
    doc = get_asset_doc(asset_id, site_name)
    if not doc:
        return ErrorResult(error=f"unknown asset_id {asset_id} at site {site_name}")
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
) -> Union[AssetsWithMetadataResult, ErrorResult]:
    """List assets at a site with metadata: description, assettype, vintage, and installed sensor
    count (one AssetSummary per asset). Optionally filter by `assettype` (e.g. 'PUMP',
    'COMPRESSOR'). For just the bare IDs to pass to other tools, use `asset_ids`."""
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
                AssetSummary(
                    asset_id=d["assetnum"],
                    description=d.get("description"),
                    assettype=d.get("assettype"),
                    vintage=d.get("vintage"),
                    n_sensors=len(d.get("sensors", [])),
                )
                for d in res["docs"]
            ),
            key=lambda r: r.asset_id,
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
            else _installed_sensors(asset_id, site_name)
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
    end: Optional[str] = None,
) -> Union[StreamExtentResult, ErrorResult]:
    """Time bounds + record count for an asset's stream (optionally one sensor and/or a window).
    Use this before history() to learn start_time / end_time / total_records and whether the
    result will page (exceeds_page_limit). Timestamps are ISO 8601, half-open [start, end);
    omit both for the full range."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    err = _validate_dates(start, end)
    if err:
        return ErrorResult(error=err)
    if not db:
        return ErrorResult(error="CouchDB not connected")

    selector = _time_selector(asset_id, start, end)
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


@mcp.tool(title="Sensor Coverage")
def sensor_coverage(
    site_name: str,
    asset_id: str,
    max_scan_docs: int = 5000,
) -> Union[SensorCoverageResult, ErrorResult]:
    """Per-measured-sensor record counts and time coverage (non-null count, first/last timestamp)
    for an asset. Complements measured_sensors(), which lists names but not how much data each
    channel has. `max_scan_docs` = maximum reading documents to scan (0 = scan all; larger =
    more complete, slower)."""
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
        max_docs=max_scan_docs if max_scan_docs > 0 else None,
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
    sensor: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Union[SensorStatsResult, ErrorResult]:
    """Numeric summary (count/min/max/mean/stddev) per sensor over an optional time window,
    without returning raw rows. Omit `sensor` to summarize EVERY measured sensor; give one to
    summarize just that channel. Timestamps are ISO 8601, half-open [start, end); omit both for
    the full range."""
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    err = _validate_dates(start, end)
    if err:
        return ErrorResult(error=err)
    if not db:
        return ErrorResult(error="CouchDB not connected")

    targets = [sensor] if sensor else get_sensor_list(asset_id)
    if not targets:
        return ErrorResult(error=f"unknown asset_id {asset_id} or no sensors found")

    acc = {s: {"vals": [], "nulls": 0, "first": None, "last": None} for s in targets}
    tset = set(targets)
    for doc in _iter_docs(_time_selector(asset_id, start, end)):
        ts = doc.get("timestamp")
        for s in tset:
            if s not in doc:
                continue
            a = acc[s]
            if ts is not None:
                if a["first"] is None:
                    a["first"] = ts
                a["last"] = ts
            v = doc.get(s)
            if v is None:
                a["nulls"] += 1
                continue
            try:
                a["vals"].append(float(v))
            except (TypeError, ValueError):
                a["nulls"] += 1

    stats: List[SensorStat] = []
    for s in targets:
        a = acc[s]
        vals = a["vals"]
        stats.append(
            SensorStat(
                sensor=s,
                count=len(vals),
                null_count=a["nulls"],
                min=min(vals) if vals else None,
                max=max(vals) if vals else None,
                mean=statistics.fmean(vals) if vals else None,
                stddev=(
                    statistics.pstdev(vals)
                    if len(vals) > 1
                    else (0.0 if vals else None)
                ),
                first_timestamp=a["first"],
                last_timestamp=a["last"],
            )
        )
    return SensorStatsResult(
        site_name=site_name,
        asset_id=asset_id,
        stats=stats,
        message=f"stats for {len(stats)} sensor(s) on asset_id {asset_id}.",
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
