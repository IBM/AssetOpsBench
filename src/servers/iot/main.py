import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import couchdb3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from servers.iot.models import (
    AssetDetail,
    AssetSensorMatch,
    AssetSummary,
    AssetsResult,
    AssetsWithMetadataResult,
    ErrorResult,
    FindAssetsResult,
    HistoryResult,
    LatestReadingResult,
    SensorCoverageResult,
    SensorStatsResult,
    SensorsResult,
    SitesResult,
    StreamExtentResult,
)
from servers.iot.telemetry_helper import (
    _SensorAccumulator,
    _SensorCoverageAccumulator,
    _TimestampHandlingError,
    _coerce_finite_number,
    _decode_history_cursor,
    _encode_history_cursor,
    _history_cursor_context,
    _history_observation,
    _iter_records,
    _iter_records_in_window,
    _parse_iso_timestamp,
    _timestamp_age_seconds,
    _validate_dates,
)

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
        "Read-only IoT sensor data and asset registry tools. Browse sites, assets, and "
        "sensors; inspect asset details; compare installed and measured sensors; and query "
        "telemetry."
    ),
)

DEFAULT_SITES = ["MAIN"]
PAGE_SIZE = 1000
RESERVED_FIELDS = {"_id", "_rev", "asset_id", "timestamp", "dataset", "type", "doctype"}


_registry_sites_cache: Optional[List[str]] = None
_sensor_list_cache: Dict[str, List[str]] = {}


def get_sensor_list(asset_id: str) -> List[str]:
    """Return sorted telemetry field names observed across all records for an asset."""
    if asset_id in _sensor_list_cache:
        return _sensor_list_cache[asset_id]
    if not iot_db:
        return []
    try:
        found = set()
        seen = False
        for doc in _iter_records(iot_db, {"asset_id": asset_id}):
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


@mcp.tool(title="List Sites")
def sites() -> SitesResult:
    """List sorted site identifiers available in the asset registry.

    Returns:
        SitesResult: Distinct site identifiers, or `["MAIN"]` when none are
        available.
    """
    return SitesResult(sites=known_sites())


@mcp.tool(title="List Asset IDs")
def asset_ids(site_name: str) -> Union[AssetsResult, ErrorResult]:
    """List asset identifiers registered at one site.

    Use `assets()` when registry metadata is also needed.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.

    Returns:
        AssetsResult: Sorted asset identifiers and their count.
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

    Includes identity, description, type, status, location, installation date,
    vintage, and installed-sensor count. Use `installed_sensors()` for names.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        AssetDetail: The available registry fields; optional values are null
        when absent.
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
    """List measurement fields observed in an asset's telemetry stream.

    Returns the sorted union across the full stream, excluding record metadata.
    Use `installed_sensors()` for assigned sensors, which may differ.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorsResult: Observed measurement field names and their count.
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
    """List sensor names assigned to an asset in the registry.

    Use `measured_sensors()` for fields observed in telemetry; the two lists may
    differ.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorsResult: Sensor names in registry order and their count.
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
    """List assets at one site with compact registry metadata.

    Use `asset_ids()` for identifiers only and `asset_detail()` for more fields
    about one asset.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        assettype: Optional exact, case-sensitive asset type filter, such as
            `PUMP` or `COMPRESSOR`.

    Returns:
        AssetsWithMetadataResult: Assets sorted by identifier, with description,
        type, vintage, and installed-sensor count.
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
    """Find site assets by installed or measured sensor names.

    Searches only assets registered at `site_name`. Duplicate queries are
    removed while preserving first occurrence order.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        sensors: Sensor names, or fragments when `substring` is true.
        match: `all` requires every query; `any` requires at least one.
        substring: Use case-insensitive fragment matching instead of exact,
            case-sensitive matching.
        source: `measured` searches observed fields; `installed` searches
            registry assignments. Defaults to `measured`.

    Returns:
        FindAssetsResult: Query settings and matching assets with the concrete
        sensor names found.
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
    """Inspect the size and timestamp span of an asset's telemetry stream.

    Optionally filter by one measured sensor and a half-open ISO 8601 window:
    `start` is inclusive and `end` is exclusive. Sensor-filtered records count
    only when the field is present and non-null. Bounds and record timestamps
    must consistently include or omit timezone offsets; aware timestamps are
    compared by instant even when their offsets differ.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.
        sensor: Optional exact field name from `measured_sensors()`.
            Reserved metadata fields such as `asset_id` and `timestamp` are not
            valid sensor names.
        start: Optional inclusive ISO 8601 date or datetime lower bound.
        end: Optional exclusive ISO 8601 date or datetime upper bound.

    Returns:
        StreamExtentResult: Earliest and latest matching timestamps, record
        count, whether the count exceeds one 1000-record page, and average
        interval implied by the span and count.
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
    total_records = 0
    try:
        for _, timestamp, timestamp_dt in _iter_records_in_window(
            iot_db, selector, start_dt, end_dt, fields=["timestamp"]
        ):
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
    except _TimestampHandlingError as e:
        return ErrorResult(error=str(e))
    except Exception as e:
        logger.error(f"stream_extent failed: {e}")
        return ErrorResult(error="unable to inspect telemetry stream extent")


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
    """Return one chronological page of telemetry observations for an asset.

    Optional bounds form a half-open ISO 8601 window. Use `sensors` to project
    exact measured fields; omit it to return all measured fields present in each
    row. Every row includes `asset_id` and `timestamp` and excludes other record
    metadata.

    Leave `cursor` unset for the first page. When `has_more` is true, repeat the
    same query with `next_cursor`. Cursors are opaque and query-bound. Timestamp
    offset rules match `stream_extent()`.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.
        start: Optional inclusive ISO 8601 lower bound.
        end: Optional exclusive ISO 8601 upper bound.
        sensors: Optional non-empty list of exact measured field names. Duplicate
            names are removed while preserving order.
        limit: Maximum observations in this page, from 1 to 1000.
        cursor: Opaque `next_cursor` from the previous page of the same query.

    Returns:
        HistoryResult: Projected observations, returned count, echoed bounds,
        and cursor fields for the next page.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    validation_error = _validate_dates(start, end)
    if validation_error:
        return ErrorResult(error=validation_error)
    if not iot_db:
        return ErrorResult(error="IoT records database not connected")
    if isinstance(limit, bool) or not 1 <= limit <= PAGE_SIZE:
        return ErrorResult(error=f"limit must be between 1 and {PAGE_SIZE}")

    selected_sensors: Optional[List[str]] = None
    if sensors is not None:
        if not sensors:
            return ErrorResult(error="sensors must not be empty when provided")
        if any(not sensor.strip() for sensor in sensors):
            return ErrorResult(error="sensor names must not be empty")
        selected_sensors = list(dict.fromkeys(sensors))
        reserved = [sensor for sensor in selected_sensors if sensor in RESERVED_FIELDS]
        if reserved:
            return ErrorResult(
                error=(
                    "sensors must be telemetry fields, not reserved metadata "
                    f"fields {reserved}"
                )
            )
        available_sensors = get_sensor_list(asset_id)
        if not available_sensors:
            return ErrorResult(error=f"unknown asset_id {asset_id} or no sensors found")
        unknown = [
            sensor for sensor in selected_sensors if sensor not in available_sensors
        ]
        if unknown:
            return ErrorResult(
                error=f"unknown sensors {unknown} for asset_id {asset_id}"
            )

    cursor_context = _history_cursor_context(
        site_name, asset_id, start, end, selected_sensors
    )
    offset = 0
    if cursor is not None:
        offset, cursor_error = _decode_history_cursor(cursor, cursor_context)
        if cursor_error:
            return ErrorResult(error=cursor_error)
        assert offset is not None

    start_dt = _parse_iso_timestamp(start) if start is not None else None
    end_dt = _parse_iso_timestamp(end) if end is not None else None
    selector: Dict[str, Any] = {
        "asset_id": asset_id,
        "timestamp": {"$exists": True, "$ne": None},
    }
    fields = ["timestamp", *selected_sensors] if selected_sensors else None
    observations: List[Dict[str, Any]] = []
    matched_records = 0
    has_more = False
    previous_datetime: Optional[datetime] = None
    try:
        for doc, timestamp, timestamp_dt in _iter_records_in_window(
            iot_db, selector, start_dt, end_dt, fields=fields
        ):
            if previous_datetime is not None and timestamp_dt < previous_datetime:
                raise _TimestampHandlingError(
                    "telemetry timestamps cannot be returned in chronological order"
                )
            previous_datetime = timestamp_dt
            if matched_records < offset:
                matched_records += 1
                continue
            if len(observations) >= limit:
                has_more = True
                break
            observations.append(
                _history_observation(
                    doc,
                    asset_id,
                    timestamp,
                    selected_sensors,
                    RESERVED_FIELDS,
                )
            )
            matched_records += 1
    except _TimestampHandlingError as e:
        return ErrorResult(error=str(e))
    except Exception as e:
        logger.error(f"history failed: {e}")
        return ErrorResult(error="unable to retrieve telemetry history")

    next_cursor = None
    if has_more:
        next_cursor = _encode_history_cursor(
            offset + len(observations), cursor_context
        )
    return HistoryResult(
        site_name=site_name,
        asset_id=asset_id,
        observations=observations,
        returned=len(observations),
        next_cursor=next_cursor,
        has_more=has_more,
        start=start,
        end=end,
        message=(
            f"returned {len(observations)} observation(s) for asset_id {asset_id}; "
            f"has_more={has_more}."
        ),
    )


@mcp.tool(title="Latest Reading")
def latest_reading(
    site_name: str,
    asset_id: str,
    sensor: Optional[str] = None,
) -> Union[LatestReadingResult, ErrorResult]:
    """Return the newest telemetry observation for an asset.

    Omit `sensor` for all measured fields in the newest record. When a sensor is
    provided, returns the newest record where that field is present and non-null.
    Records are compared by parsed timestamp using the `stream_extent()` rules.

    `age_seconds` is the difference between the current UTC time and the reading
    timestamp. Offset-free timestamps are interpreted as UTC; a negative value
    indicates a timestamp in the future.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.
        sensor: Optional exact field name from `measured_sensors()`. Reserved
            metadata fields are not valid sensor names.

    Returns:
        LatestReadingResult: Selected timestamp, values, and age in seconds.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not iot_db:
        return ErrorResult(error="IoT records database not connected")
    if sensor is not None and not sensor.strip():
        return ErrorResult(error="sensor must not be empty")
    if sensor in RESERVED_FIELDS:
        return ErrorResult(
            error=(
                "sensor must be a telemetry field, not reserved metadata "
                f"field {sensor}"
            )
        )
    if sensor is not None:
        available_sensors = get_sensor_list(asset_id)
        if not available_sensors:
            return ErrorResult(error=f"unknown asset_id {asset_id} or no sensors found")
        if sensor not in available_sensors:
            return ErrorResult(error=f"unknown sensor {sensor} for asset_id {asset_id}")

    selector: Dict[str, Any] = {
        "asset_id": asset_id,
        "timestamp": {"$exists": True, "$ne": None},
    }
    fields = None
    if sensor is not None:
        selector[sensor] = {"$exists": True, "$ne": None}
        fields = ["timestamp", sensor]

    latest_doc: Optional[Dict[str, Any]] = None
    latest_timestamp: Optional[str] = None
    latest_datetime: Optional[datetime] = None
    try:
        for doc, timestamp, timestamp_dt in _iter_records_in_window(
            iot_db, selector, None, None, fields=fields
        ):
            if latest_datetime is None or timestamp_dt > latest_datetime:
                latest_doc = doc
                latest_timestamp = timestamp
                latest_datetime = timestamp_dt
    except _TimestampHandlingError as e:
        return ErrorResult(error=str(e))
    except Exception as e:
        logger.error(f"latest_reading failed: {e}")
        return ErrorResult(error="unable to retrieve latest telemetry reading")

    if latest_doc is None or latest_timestamp is None or latest_datetime is None:
        return ErrorResult(
            error=f"no records for asset_id {asset_id}"
            + (f", sensor {sensor}" if sensor else "")
        )

    if sensor is None:
        values = {
            field: value
            for field, value in latest_doc.items()
            if field not in RESERVED_FIELDS
        }
    else:
        values = {sensor: latest_doc[sensor]}
    return LatestReadingResult(
        site_name=site_name,
        asset_id=asset_id,
        timestamp=latest_timestamp,
        values=values,
        age_seconds=_timestamp_age_seconds(latest_datetime),
        message=f"latest reading for asset_id {asset_id} at {latest_timestamp}.",
    )


@mcp.tool(title="Sensor Coverage")
def sensor_coverage(
    site_name: str,
    asset_id: str,
) -> Union[SensorCoverageResult, ErrorResult]:
    """Summarize non-null observation coverage for every measured sensor.

    Scans the full timestamped stream. `non_null_count` includes present,
    non-null values of any type. Fields observed only with null values have zero
    count and null bounds. Timestamp offset rules match `stream_extent()`.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorCoverageResult: Timestamped records scanned and per-sensor counts
        and chronological bounds, sorted by field name.
    """
    if not _is_known_site(site_name):
        return ErrorResult(error=f"unknown site {site_name}")
    if not iot_db:
        return ErrorResult(error="IoT records database not connected")

    selector: Dict[str, Any] = {
        "asset_id": asset_id,
        "timestamp": {"$exists": True, "$ne": None},
    }
    coverage: Dict[str, _SensorCoverageAccumulator] = {}
    docs_scanned = 0
    try:
        for doc, timestamp, timestamp_dt in _iter_records_in_window(
            iot_db, selector, None, None
        ):
            docs_scanned += 1
            for field, value in doc.items():
                if field in RESERVED_FIELDS:
                    continue
                accumulator = coverage.setdefault(
                    field, _SensorCoverageAccumulator()
                )
                if value is not None:
                    accumulator.add_non_null(timestamp, timestamp_dt)
    except _TimestampHandlingError as e:
        return ErrorResult(error=str(e))
    except Exception as e:
        logger.error(f"sensor_coverage failed: {e}")
        return ErrorResult(error="unable to calculate sensor coverage")

    if docs_scanned == 0:
        return ErrorResult(error=f"unknown asset_id {asset_id} or no records found")

    sensors = [coverage[field].result(field) for field in sorted(coverage)]
    message = (
        f"coverage for {len(sensors)} sensor(s) on asset_id {asset_id} "
        f"across {docs_scanned} timestamped record(s)."
    )
    return SensorCoverageResult(
        site_name=site_name,
        asset_id=asset_id,
        docs_scanned=docs_scanned,
        sensors=sensors,
        message=message,
    )


@mcp.tool(title="Sensor Statistics")
def sensor_stats(
    site_name: str,
    asset_id: str,
    sensor: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Union[SensorStatsResult, ErrorResult]:
    """Compute numeric statistics for one or all measured sensors.

    Omit `sensor` to summarize every measured field. Optional bounds form a
    half-open ISO 8601 window. Timestamp offset rules match `stream_extent()`.

    Finite numbers and numeric strings contribute to the numeric statistics.
    Present null, boolean, non-numeric, and non-finite values contribute to
    `null_count`; missing fields contribute to neither count. `stddev` is the
    population standard deviation: `0.0` for one numeric value and null for none.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.
        sensor: Optional exact field name from `measured_sensors()`. Omit it to
            summarize all measured fields.
        start: Optional inclusive ISO 8601 lower bound.
        end: Optional exclusive ISO 8601 upper bound.

    Returns:
        SensorStatsResult: Per-field numeric and null counts, range, mean,
        population standard deviation, and earliest/latest numeric timestamps.
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

    available_sensors = get_sensor_list(asset_id)
    if not available_sensors:
        return ErrorResult(error=f"unknown asset_id {asset_id} or no sensors found")
    if sensor is not None and sensor not in available_sensors:
        return ErrorResult(error=f"unknown sensor {sensor} for asset_id {asset_id}")

    targets = [sensor] if sensor is not None else available_sensors
    accumulators = {target: _SensorAccumulator() for target in targets}
    start_dt = _parse_iso_timestamp(start) if start is not None else None
    end_dt = _parse_iso_timestamp(end) if end is not None else None
    selector: Dict[str, Any] = {
        "asset_id": asset_id,
        "timestamp": {"$exists": True, "$ne": None},
    }
    if sensor is not None:
        selector[sensor] = {"$exists": True}

    records_in_window = 0
    try:
        for doc, timestamp, timestamp_dt in _iter_records_in_window(
            iot_db,
            selector,
            start_dt,
            end_dt,
            fields=["timestamp", *targets],
        ):
            records_in_window += 1
            for target in targets:
                if target not in doc:
                    continue
                numeric_value = _coerce_finite_number(doc.get(target))
                if numeric_value is None:
                    accumulators[target].add_invalid()
                    continue
                accumulators[target].add_numeric(
                    numeric_value, timestamp, timestamp_dt
                )
    except _TimestampHandlingError as e:
        return ErrorResult(error=str(e))
    except Exception as e:
        logger.error(f"sensor_stats failed: {e}")
        return ErrorResult(error="unable to calculate sensor statistics")

    if records_in_window == 0:
        return ErrorResult(
            error=f"no records for asset_id {asset_id}"
            + (f", sensor {sensor}" if sensor else "")
        )

    stats = [accumulators[target].result(target) for target in targets]
    return SensorStatsResult(
        site_name=site_name,
        asset_id=asset_id,
        stats=stats,
        message=f"numeric stats for {len(stats)} sensor(s) on asset_id {asset_id}.",
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
