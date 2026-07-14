import base64
import binascii
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
    SensorCoverage,
    SensorCoverageResult,
    SensorStat,
    SensorStatsResult,
    SensorsResult,
    SitesResult,
    StreamExtentResult,
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
        "IoT asset registry and telemetry record tools. Use sites() to discover site names, "
        "asset_ids() for bare assetnum values at a site, asset_detail() for one asset's "
        "registry details, assets() for registry metadata with optional assettype filtering, "
        "find_assets_by_sensors() to find assets by installed or measured sensors, "
        "installed_sensors() for registry sensor inventory, and measured_sensors() for "
        "observed telemetry fields. Use stream_extent() to inspect telemetry time bounds "
        "and record counts before planning larger telemetry reads. Use history() for paged "
        "observations and latest_reading() for the newest values. Use sensor_coverage() for "
        "per-field non-null counts and time coverage, and sensor_stats() for numeric summaries "
        "without returning raw telemetry rows."
    ),
)

DEFAULT_SITES = ["MAIN"]
PAGE_SIZE = 1000
RESERVED_FIELDS = {"_id", "_rev", "asset_id", "timestamp", "dataset", "type", "doctype"}


_registry_sites_cache: Optional[List[str]] = None
_sensor_list_cache: Dict[str, List[str]] = {}


class _TimestampHandlingError(ValueError):
    pass


@dataclass
class _SensorAccumulator:
    count: int = 0
    null_count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: float = 0.0
    squared_deviation_sum: float = 0.0
    first_timestamp: Optional[str] = None
    last_timestamp: Optional[str] = None
    first_datetime: Optional[datetime] = None
    last_datetime: Optional[datetime] = None

    def add_invalid(self) -> None:
        self.null_count += 1

    def add_numeric(
        self, value: float, timestamp: str, timestamp_dt: datetime
    ) -> None:
        self.count += 1
        delta = value - self.mean_value
        self.mean_value += delta / self.count
        self.squared_deviation_sum += delta * (value - self.mean_value)
        self.min_value = (
            value if self.min_value is None else min(self.min_value, value)
        )
        self.max_value = (
            value if self.max_value is None else max(self.max_value, value)
        )
        if self.first_datetime is None or timestamp_dt < self.first_datetime:
            self.first_datetime = timestamp_dt
            self.first_timestamp = timestamp
        if self.last_datetime is None or timestamp_dt > self.last_datetime:
            self.last_datetime = timestamp_dt
            self.last_timestamp = timestamp

    def result(self, sensor: str) -> SensorStat:
        mean = None
        stddev = None
        if self.count:
            if math.isfinite(self.mean_value):
                mean = self.mean_value
            variance = self.squared_deviation_sum / self.count
            if math.isfinite(variance):
                stddev = math.sqrt(max(variance, 0.0))
        return SensorStat(
            sensor=sensor,
            count=self.count,
            null_count=self.null_count,
            min=self.min_value,
            max=self.max_value,
            mean=mean,
            stddev=stddev,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
        )


@dataclass
class _SensorCoverageAccumulator:
    non_null_count: int = 0
    first_timestamp: Optional[str] = None
    last_timestamp: Optional[str] = None
    first_datetime: Optional[datetime] = None
    last_datetime: Optional[datetime] = None

    def add_non_null(self, timestamp: str, timestamp_dt: datetime) -> None:
        self.non_null_count += 1
        if self.first_datetime is None or timestamp_dt < self.first_datetime:
            self.first_datetime = timestamp_dt
            self.first_timestamp = timestamp
        if self.last_datetime is None or timestamp_dt > self.last_datetime:
            self.last_datetime = timestamp_dt
            self.last_timestamp = timestamp

    def result(self, sensor: str) -> SensorCoverage:
        return SensorCoverage(
            sensor=sensor,
            non_null_count=self.non_null_count,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
        )


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


def _iter_records_in_window(
    selector: Dict[str, Any],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    fields: Optional[List[str]] = None,
) -> Iterator[Tuple[Dict[str, Any], str, datetime]]:
    """Yield timestamped records in a parsed half-open time window."""
    stream_is_aware: Optional[bool] = None
    for doc in _iter_records(selector, fields=fields):
        timestamp = doc.get("timestamp")
        if timestamp is None:
            continue
        timestamp_dt = _parse_iso_timestamp(timestamp)
        if timestamp_dt is None:
            raise _TimestampHandlingError(
                "telemetry record has an invalid ISO 8601 timestamp"
            )

        timestamp_is_aware = _is_timezone_aware(timestamp_dt)
        if stream_is_aware is None:
            stream_is_aware = timestamp_is_aware
            for bound in (start_dt, end_dt):
                if (
                    bound is not None
                    and _is_timezone_aware(bound) != stream_is_aware
                ):
                    raise _TimestampHandlingError(
                        "timestamp bounds must use the same timezone awareness "
                        "as telemetry timestamps"
                    )
        elif timestamp_is_aware != stream_is_aware:
            raise _TimestampHandlingError(
                "telemetry timestamps use mixed timezone awareness"
            )

        if start_dt is not None and timestamp_dt < start_dt:
            continue
        if end_dt is not None and timestamp_dt >= end_dt:
            continue
        yield doc, timestamp, timestamp_dt


def _coerce_finite_number(value: Any) -> Optional[float]:
    """Return a finite float for numeric values and numeric strings."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _history_cursor_context(
    site_name: str,
    asset_id: str,
    start: Optional[str],
    end: Optional[str],
    sensors: Optional[List[str]],
) -> Dict[str, Any]:
    return {
        "site_name": site_name,
        "asset_id": asset_id,
        "start": start,
        "end": end,
        "sensors": sensors,
    }


def _encode_history_cursor(offset: int, context: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"version": 1, "offset": offset, "context": context},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_history_cursor(
    cursor: str, expected_context: Dict[str, Any]
) -> Tuple[Optional[int], Optional[str]]:
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
        )
    except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None, "invalid cursor"
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None, "invalid cursor"
    offset = payload.get("offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return None, "invalid cursor"
    if payload.get("context") != expected_context:
        return None, "cursor does not match history query"
    return offset, None


def _history_observation(
    doc: Dict[str, Any],
    asset_id: str,
    timestamp: str,
    sensors: Optional[List[str]],
) -> Dict[str, Any]:
    observation: Dict[str, Any] = {
        "asset_id": asset_id,
        "timestamp": timestamp,
    }
    if sensors is None:
        observation.update(
            {
                field: value
                for field, value in doc.items()
                if field not in RESERVED_FIELDS
            }
        )
    else:
        observation.update(
            {field: doc[field] for field in sensors if field in doc}
        )
    return observation


def _timestamp_age_seconds(timestamp_dt: datetime) -> float:
    if not _is_timezone_aware(timestamp_dt):
        timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)
    return (
        datetime.now(timezone.utc) - timestamp_dt.astimezone(timezone.utc)
    ).total_seconds()


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
    """Inspect the size and timestamp span of an asset's telemetry stream.

    Optionally restrict the stream to one measured sensor and/or an ISO 8601
    window. Sensor-filtered records count only when the field is present and
    non-null. The window is half-open: `start` is inclusive and `end` is
    exclusive. Bounds and telemetry timestamps must consistently include or
    omit timezone offsets; explicit offsets may differ and compare by instant.

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
        StreamExtentResult: `start_time` and `end_time` are the earliest and
        latest matching timestamps. `total_records` is the matching count,
        `exceeds_page_limit` indicates more than one server page, and
        `approx_interval_seconds` is the average spacing implied by the span and
        count.
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
            selector, start_dt, end_dt, fields=["timestamp"]
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

    Optional bounds form a half-open ISO 8601 window: `start` is inclusive and
    `end` is exclusive. Use `sensors` to project exact fields from
    `measured_sensors()`. Every row includes `asset_id` and `timestamp`; other
    metadata and internal record identifiers are excluded. Omit `sensors` to
    return every measured field present in each row.

    `limit` must be between 1 and 1000. Leave `cursor` unset for the first page.
    When `has_more` is true, repeat the same query arguments with `next_cursor`.
    Cursors are opaque and bound to the site, asset, window, and sensor list.
    Timestamp offset rules match `stream_extent()`; history returns an error if
    timestamp representations cannot be emitted in chronological order.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.
        start: Optional inclusive ISO 8601 date or datetime lower bound.
        end: Optional exclusive ISO 8601 date or datetime upper bound.
        sensors: Optional non-empty list of exact measured field names. Duplicate
            names are removed while preserving order.
        limit: Maximum observations in this page, from 1 through 1000.
        cursor: Opaque `next_cursor` from the previous page of the same query.

    Returns:
        HistoryResult: `observations` contains the projected chronological rows,
        `returned` is the page size, and `has_more`/`next_cursor` control paging.
        `start` and `end` echo the requested window.
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
            selector, start_dt, end_dt, fields=fields
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
                    doc, asset_id, timestamp, selected_sensors
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

    Omit `sensor` to return every measured field present in the newest record.
    Provide an exact field name from `measured_sensors()` to return the newest
    record where that field is present and non-null. The newest record is chosen
    by parsed timestamp rather than string order, using the same timestamp rules
    as `stream_extent()`.

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
        LatestReadingResult: Contains the selected observation timestamp,
        requested sensor values, age in seconds, and a compact message.
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
            selector, None, None, fields=fields
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

    Scans the asset's full timestamped stream; use `stream_extent()` first when
    you need to estimate its size. `non_null_count` includes present non-null
    values of any type. Fields observed only with null values are returned with
    zero count and null time bounds. First and last timestamps are selected
    chronologically. Mixed offset-aware and offset-free timestamps return an
    error.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.

    Returns:
        SensorCoverageResult: `docs_scanned` is the number of timestamped records
        examined. `sensors` is sorted by field name and contains each field's
        non-null count and earliest/latest non-null timestamps.
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
            selector, None, None
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

    Omit `sensor` to summarize every field returned by `measured_sensors()`.
    Optional bounds form a half-open ISO 8601 window: `start` is inclusive and
    `end` is exclusive. Timestamp offset rules match `stream_extent()`.

    Finite numbers and numeric strings contribute to the numeric statistics.
    Present null, boolean, non-numeric, and non-finite values contribute to
    `null_count`; missing fields contribute to neither count. `stddev` is the
    population standard deviation: `0.0` for one numeric value and null for none.
    First and last timestamps identify the earliest and latest numeric samples.

    Args:
        site_name: Exact site id to query, such as `MAIN`. Use `sites()` to
            discover valid site ids.
        asset_id: Exact asset id from `asset_ids()`, such as `Chiller 6`.
        sensor: Optional exact field name from `measured_sensors()`. Omit it to
            summarize all measured fields.
        start: Optional inclusive ISO 8601 date or datetime lower bound.
        end: Optional exclusive ISO 8601 date or datetime upper bound.

    Returns:
        SensorStatsResult: `stats` contains one entry per requested field, sorted
        by field name when `sensor` is omitted. Each entry includes numeric and
        null counts, range, mean, population standard deviation, and
        earliest/latest numeric sample timestamps.
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
