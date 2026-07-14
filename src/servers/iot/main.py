import logging
import os
from datetime import datetime
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
    iot_db = couchdb3.Database(
        IOT_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info(f"Connected to IoT records database: {IOT_DBNAME}")
except Exception as e:
    logger.error(f"Failed to connect to IoT records database: {e}")
    iot_db = None

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
        "asset_ids() for bare assetnum values at a site, asset_detail() for one asset's "
        "registry details, assets() for registry metadata with optional assettype filtering, "
        "find_assets_by_sensors() to find assets by installed or measured sensors, "
        "installed_sensors() for registry sensor inventory, and measured_sensors() for "
        "observed telemetry fields. Use stream_extent() to inspect telemetry time bounds "
        "and record counts before planning larger telemetry reads."
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


class AssetDetail(BaseModel):
    site_name: str
    asset_id: str
    description: Optional[str]
    assettype: Optional[str]
    status: Optional[str]
    location: Optional[str]
    installdate: Optional[str]
    vintage: Optional[str]
    n_installed_sensors: int
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


_registry_sites_cache: Optional[List[str]] = None
_sensor_list_cache: Dict[str, List[str]] = {}


def _iter_records(
    selector: Dict[str, Any],
    fields: Optional[List[str]] = None,
    sort: Optional[List[Dict[str, str]]] = None,
    page_size: int = PAGE_SIZE,
) -> Iterator[Dict[str, Any]]:
    """Yield telemetry records matching a selector across paged database results."""
    if not iot_db:
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
        res = iot_db.find(selector, **kwargs)
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
    if not iot_db:
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


def _site_asset_ids(site_name: str) -> List[str]:
    """Return asset ids registered at a site."""
    if not asset_db:
        return []
    try:
        res = asset_db.find(
            {"siteid": site_name},
            fields=["assetnum"],
            limit=100000,
        )
        return sorted(doc["assetnum"] for doc in res["docs"])
    except Exception as e:
        logger.error(f"_site_asset_ids failed: {e}")
        return []


def _installed_sensors(asset_id: str, site_name: Optional[str] = None) -> List[str]:
    """Return registry sensor names installed on an asset."""
    if not asset_db:
        return []
    try:
        selector: Dict[str, Any] = {"assetnum": asset_id}
        if site_name is not None:
            selector["siteid"] = site_name
        res = asset_db.find(selector, fields=["sensors"], limit=1)
        docs = res.get("docs", [])
        return list(docs[0].get("sensors") or []) if docs else []
    except Exception as e:
        logger.error(f"_installed_sensors failed for asset_id {asset_id}: {e}")
        return []


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse one ISO 8601 timestamp, returning None for invalid values."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _is_timezone_aware(value: datetime) -> bool:
    return value.utcoffset() is not None


def _validate_dates(start: Optional[str], end: Optional[str]) -> Optional[str]:
    """Return None when the optional ISO 8601 bounds are valid."""
    start_dt = _parse_iso_timestamp(start) if start is not None else None
    end_dt = _parse_iso_timestamp(end) if end is not None else None
    if start is not None and start_dt is None:
        return "Invalid date format for start (expected ISO 8601)"
    if end is not None and end_dt is None:
        return "Invalid date format for end (expected ISO 8601)"
    if start_dt is not None and end_dt is not None:
        if _is_timezone_aware(start_dt) != _is_timezone_aware(end_dt):
            return "start and end must use matching timezone awareness"
        if start_dt >= end_dt:
            return "start >= end"
    return None


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


@mcp.tool(title="Get Asset Detail")
def asset_detail(site_name: str, asset_id: str) -> Union[AssetDetail, ErrorResult]:
    """Return registry details for one asset.

    This tool reads the asset registry from `asset_db` (`ASSET_DBNAME`, default
    `asset`) and returns nameplate fields for one asset record. Use
    `installed_sensors()` when you only need the registry sensor inventory.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        AssetDetail: Contains `site_name`, `asset_id`, `description`,
        `assettype`, `status`, `location`, `installdate`, `vintage`,
        `n_installed_sensors`, and `message`.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not asset_db:
        return ErrorResult(error="asset registry database not connected")

    try:
        res = asset_db.find(
            {"siteid": site_name, "assetnum": asset_id},
            fields=[
                "assetnum",
                "description",
                "assettype",
                "status",
                "location",
                "installdate",
                "vintage",
                "sensors",
            ],
            limit=1,
        )
        docs = res.get("docs", [])
        if not docs:
            return ErrorResult(error=f"unknown asset_id {asset_id} at site {site_name}")

        doc = docs[0]
        sensors = list(doc.get("sensors") or [])
        n_installed_sensors = len(sensors)
        assettype = doc.get("assettype")
        vintage = doc.get("vintage")
        location = doc.get("location")

        parts = [f"asset {asset_id} is a {assettype or 'asset'}"]
        if vintage:
            parts.append(f" ({vintage} vintage)")
        if location:
            parts.append(f" at {location}")
        parts.append(f" with {n_installed_sensors} installed sensors.")

        return AssetDetail(
            site_name=site_name,
            asset_id=doc.get("assetnum", asset_id),
            description=doc.get("description"),
            assettype=assettype,
            status=doc.get("status"),
            location=location,
            installdate=doc.get("installdate"),
            vintage=vintage,
            n_installed_sensors=n_installed_sensors,
            message="".join(parts),
        )
    except Exception as e:
        logger.error(f"asset_detail failed: {e}")
        return ErrorResult(error=str(e))


@mcp.tool(title="List Measured Sensors")
def measured_sensors(
    site_name: str, asset_id: str
) -> Union[SensorsResult, ErrorResult]:
    """List telemetry fields actually measured for one asset.

    This tool reads IoT records from `iot_db` (`IOT_DBNAME`, default `iot`).
    It scans records matching `asset_id` and returns the union of observed
    measurement fields. Use `installed_sensors()` instead when you need the
    asset registry inventory.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorsResult: Contains `site_name`, `asset_id`, `total_sensors`,
        `sensors`, and `message`. The `sensors` field is a sorted list of
        observed telemetry fields, excluding record metadata such as `_id`,
        `_rev`, `asset_id`, `timestamp`, `dataset`, `type`, and `doctype`.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not iot_db:
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
    """List sensor names installed on one asset according to the registry.

    This tool reads the asset registry from `asset_db` (`ASSET_DBNAME`, default
    `asset`) and returns the asset record's `sensors` inventory. Use
    `measured_sensors()` instead when you need fields actually observed in IoT
    telemetry records.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorsResult: Contains `site_name`, `asset_id`, `total_sensors`,
        `sensors`, and `message`. The `sensors` field preserves the registry
        sensor inventory order from the asset record.
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
        names = list(docs[0].get("sensors") or [])
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
                    n_sensors=len(doc.get("sensors") or []),
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


@mcp.tool(title="Find Assets By Sensors")
def find_assets_by_sensors(
    site_name: str,
    sensors: List[str],
    match: str = "all",
    substring: bool = False,
    source: str = "measured",
) -> Union[FindAssetsResult, ErrorResult]:
    """Return assets at a site that match the requested sensor names.

    The search is limited to assets registered at `site_name`. Duplicate query
    sensors are ignored while preserving the first occurrence order.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        sensors: One or more sensor names or, when `substring` is true, sensor
            name fragments to search for.
        match: `all` requires every listed sensor; `any` requires at least one.
        substring: If true, match sensor names case-insensitively by substring.
        source: `measured` checks telemetry fields; `installed` checks the
            asset registry inventory.

    Returns:
        FindAssetsResult: Contains the deduplicated query sensors, matching
        asset ids, and the concrete sensor names that matched each asset.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if match not in ("all", "any"):
        return ErrorResult(error="match must be 'all' or 'any'")
    if source not in ("measured", "installed"):
        return ErrorResult(error="source must be 'measured' or 'installed'")
    if not sensors:
        return ErrorResult(error="provide at least one sensor name")
    if not asset_db:
        return ErrorResult(error="asset registry database not connected")
    if source == "measured" and not iot_db:
        return ErrorResult(error="IoT records database not connected")

    query_sensors = list(dict.fromkeys(sensors))
    matches: List[AssetSensorMatch] = []
    for asset_id in _site_asset_ids(site_name):
        available = (
            get_sensor_list(asset_id)
            if source == "measured"
            else _installed_sensors(asset_id, site_name)
        )

        def _hits(sensor_name: str) -> List[str]:
            if substring:
                sensor_name_lower = sensor_name.lower()
                return [
                    available_sensor
                    for available_sensor in available
                    if sensor_name_lower in available_sensor.lower()
                ]
            return [
                available_sensor
                for available_sensor in available
                if available_sensor == sensor_name
            ]

        per_query = {sensor_name: _hits(sensor_name) for sensor_name in query_sensors}
        satisfied = [
            sensor_name for sensor_name, hits in per_query.items() if hits
        ]
        ok = (
            len(satisfied) == len(query_sensors)
            if match == "all"
            else len(satisfied) > 0
        )
        if ok:
            matched = sorted(
                {
                    matched_sensor
                    for hits in per_query.values()
                    for matched_sensor in hits
                }
            )
            matches.append(
                AssetSensorMatch(asset_id=asset_id, matched_sensors=matched)
            )

    return FindAssetsResult(
        site_name=site_name,
        query_sensors=query_sensors,
        match=match,
        source=source,
        total_assets=len(matches),
        assets=matches,
        message=f"{len(matches)} asset(s) at {site_name} match {query_sensors} "
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
    """Return count and observed time bounds for one asset's telemetry stream.

    The tool scans all matching telemetry records. `start` is inclusive and
    `end` is exclusive. When `sensor` is provided, only records where that
    telemetry field exists and is not null are counted.

    Timestamp bounds are validated as ISO 8601 and compared chronologically
    after parsing. Different explicit UTC offsets are supported. Bounds and
    telemetry timestamps must either all include timezone offsets or all omit
    them; ambiguous mixtures return an error.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.
        sensor: Optional telemetry field name from `measured_sensors()`.
            Reserved metadata fields such as `asset_id` and `timestamp` are not
            valid sensor names.
        start: Optional inclusive ISO 8601 timestamp lower bound.
        end: Optional exclusive ISO 8601 timestamp upper bound.

    Returns:
        StreamExtentResult: Contains first and last matching timestamps,
        `total_records`, whether the result exceeds the server page size, and
        the approximate interval between matching records.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    validation_error = _validate_dates(start, end)
    if validation_error:
        return ErrorResult(error=validation_error)
    if not iot_db:
        return ErrorResult(error="IoT records database not connected")
    if sensor is not None and not sensor.strip():
        return ErrorResult(error="sensor must not be empty")
    if sensor in RESERVED_FIELDS:
        return ErrorResult(
            error=f"sensor must be a telemetry field, not reserved metadata field {sensor}"
        )

    start_dt = _parse_iso_timestamp(start) if start is not None else None
    end_dt = _parse_iso_timestamp(end) if end is not None else None
    selector: Dict[str, Any] = {
        "asset_id": asset_id,
        "timestamp": {"$exists": True, "$ne": None},
    }
    if sensor:
        selector[sensor] = {"$exists": True, "$ne": None}

    first_timestamp: Optional[str] = None
    last_timestamp: Optional[str] = None
    first_datetime: Optional[datetime] = None
    last_datetime: Optional[datetime] = None
    stream_is_aware: Optional[bool] = None
    total_records = 0
    try:
        for doc in _iter_records(selector, fields=["timestamp"]):
            timestamp = doc.get("timestamp")
            if timestamp is None:
                continue
            timestamp_dt = _parse_iso_timestamp(timestamp)
            if timestamp_dt is None:
                return ErrorResult(
                    error="telemetry record has an invalid ISO 8601 timestamp"
                )

            timestamp_is_aware = _is_timezone_aware(timestamp_dt)
            if stream_is_aware is None:
                stream_is_aware = timestamp_is_aware
                for bound in (start_dt, end_dt):
                    if (
                        bound is not None
                        and _is_timezone_aware(bound) != stream_is_aware
                    ):
                        return ErrorResult(
                            error=(
                                "timestamp bounds must use the same timezone awareness "
                                "as telemetry timestamps"
                            )
                        )
            elif timestamp_is_aware != stream_is_aware:
                return ErrorResult(
                    error="telemetry timestamps use mixed timezone awareness"
                )

            if start_dt is not None and timestamp_dt < start_dt:
                continue
            if end_dt is not None and timestamp_dt >= end_dt:
                continue

            if first_datetime is None or timestamp_dt < first_datetime:
                first_datetime = timestamp_dt
                first_timestamp = timestamp
            if last_datetime is None or timestamp_dt > last_datetime:
                last_datetime = timestamp_dt
                last_timestamp = timestamp
            total_records += 1

        if total_records == 0:
            return ErrorResult(
                error=f"no records for asset_id {asset_id}"
                + (f", sensor {sensor}" if sensor else "")
            )

        approx_interval: Optional[float] = None
        if (
            first_datetime is not None
            and last_datetime is not None
            and total_records > 1
        ):
            approx_interval = (last_datetime - first_datetime).total_seconds() / (
                total_records - 1
            )

        return StreamExtentResult(
            site_name=site_name,
            asset_id=asset_id,
            sensor=sensor,
            start_time=first_timestamp,
            end_time=last_timestamp,
            total_records=total_records,
            exceeds_page_limit=total_records > PAGE_SIZE,
            approx_interval_seconds=approx_interval,
            message=f"{total_records} record(s) for asset_id {asset_id}"
            + (f" (sensor {sensor})" if sensor else "")
            + f" from {first_timestamp} to {last_timestamp}.",
        )
    except Exception as e:
        logger.error(f"stream_extent failed: {e}")
        return ErrorResult(error="unable to inspect telemetry stream extent")


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
