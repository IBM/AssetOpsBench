import base64
import binascii
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from servers.iot.models import SensorCoverage, SensorStat


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
    database: Any,
    selector: Dict[str, Any],
    fields: Optional[List[str]] = None,
    sort: Optional[List[Dict[str, str]]] = None,
    page_size: int = 1000,
) -> Iterator[Dict[str, Any]]:
    """Yield records matching a selector across paged query results."""
    if not database:
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
        res = database.find(selector, **kwargs)
        docs = res.get("docs", [])
        if not docs:
            break
        yield from docs
        bookmark = res.get("bookmark")
        if bookmark is None or len(docs) < page_size:
            break


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
    database: Any,
    selector: Dict[str, Any],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    fields: Optional[List[str]] = None,
    start_key: Optional[str] = None,
    end_key: Optional[str] = None,
) -> Iterator[Tuple[Dict[str, Any], str, datetime]]:
    """Yield timestamped records in a parsed half-open time window."""
    query_selector = _selector_with_timestamp_window(
        selector, start_dt, end_dt, start_key=start_key, end_key=end_key
    )
    stream_is_aware: Optional[bool] = None
    for doc in _iter_records(database, query_selector, fields=fields):
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


def _selector_with_timestamp_window(
    selector: Dict[str, Any],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    start_key: Optional[str] = None,
    end_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a selector that lets CouchDB use the asset/timestamp index.

    Python still performs the exact datetime comparison after loading each row.
    Offset-free bounds use the original ISO strings when available so CouchDB can
    narrow the asset/timestamp index range. Offset-aware bounds keep the broad
    non-null timestamp selector because lexical ordering is not equivalent to
    instant ordering when offsets differ.
    """
    query_selector = dict(selector)
    timestamp_condition: Dict[str, Any] = {"$gt": None}

    bounds = [bound for bound in (start_dt, end_dt) if bound is not None]
    if bounds and not any(_is_timezone_aware(bound) for bound in bounds):
        if start_dt is not None:
            timestamp_condition = {"$gte": start_key or start_dt.isoformat()}
        if end_dt is not None:
            timestamp_condition["$lt"] = end_key or end_dt.isoformat()

    existing = query_selector.get("timestamp")
    if isinstance(existing, dict):
        merged = {
            key: value
            for key, value in existing.items()
            if key not in {"$exists", "$ne"}
        }
        merged.update(timestamp_condition)
        query_selector["timestamp"] = merged
    else:
        query_selector["timestamp"] = timestamp_condition

    return query_selector


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
    reserved_fields: Set[str],
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
                if field not in reserved_fields
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
