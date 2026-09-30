"""Grounded asset discovery helpers for scenario generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from servers.fmsr.main import get_failure_modes
from servers.iot.main import get_asset_coverage
from servers.vibration.couchdb_client import list_asset_coverage as get_vibration_asset_coverage

from .models import AssetInstance, GroundedTimeRange, GroundingBundle
from .text import slugify_asset_name

_log = logging.getLogger(__name__)

_FAILURE_MAPPING_DIR = Path(__file__).resolve().parent / "failure_mapping"


def _normalize_string_list(values: list[Any]) -> list[str]:
    return sorted(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _normalize_string_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, values in mapping.items():
        clean_key = str(key).strip()
        clean_values = _normalize_string_list(values)
        if clean_key and clean_values:
            normalized[clean_key] = clean_values
    return normalized


def _failure_mapping_path(asset_name: str) -> Path:
    return _FAILURE_MAPPING_DIR / f"{slugify_asset_name(asset_name)}.json"


def _load_failure_mapping(asset_name: str) -> tuple[dict[str, list[str]], dict[str, list[str]]] | None:
    path = _failure_mapping_path(asset_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("Ignoring failure mapping file %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    fm2sensor = _normalize_string_mapping(dict(payload.get("fm2sensor", {}) or {}))
    sensor2fm = _normalize_string_mapping(dict(payload.get("sensor2fm", {}) or {}))
    if not fm2sensor and not sensor2fm:
        return None
    _log.info("Loaded F2S/S2F mapping for %s from %s", asset_name, path)
    return fm2sensor, sensor2fm


def _as_mapping(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return {}


def _iot_timestamp_to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _discover_iot_inventory_and_instances(asset_name: str) -> tuple[list[AssetInstance], list[str]]:
    instances: list[AssetInstance] = []
    iot_union: set[str] = set()

    for row in get_asset_coverage():
        if slugify_asset_name(row.get("asset_class") or "") != slugify_asset_name(asset_name):
            continue
        asset_id = row["asset_id"]
        sensors = row["sensors"]
        for s in sensors:
            if str(s).strip():
                iot_union.add(str(s).strip())
        tr_raw = row["time_range"]
        iot_time_range = GroundedTimeRange(
            start=_iot_timestamp_to_str(tr_raw.get("start")),
            end=_iot_timestamp_to_str(tr_raw.get("end")),
            total_observations=int(tr_raw.get("total_observations") or 0),
        )
        instances.append(
            AssetInstance(
                site_name=row["site_name"],
                asset_id=asset_id,
                has_iot=bool(sensors),
                iot_time_range=iot_time_range,
            )
        )

    return instances, sorted(iot_union)


def _vibration_by_site_asset() -> dict[tuple[str, str], dict[str, Any]]:
    rows = get_vibration_asset_coverage()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        site = str(row.get("site_name", "")).strip()
        aid = str(row.get("asset_id", "")).strip()
        if site and aid:
            out[(site, aid)] = row
    return out


def _overlay_vibration(
    instances: list[AssetInstance],
    vib_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    vib_union: set[str] = set()
    for inst in instances:
        row = vib_by_key.get((inst.site_name, inst.asset_id))
        if not row:
            continue
        raw_sensors = row.get("sensors", []) or []
        for s in raw_sensors:
            if str(s).strip():
                vib_union.add(str(s).strip())
        inst.has_vibration = bool(raw_sensors)
        tr = dict(row.get("time_range", {}))
        inst.vibration_time_range = GroundedTimeRange(
            start=tr.get("start"),
            end=tr.get("end"),
            total_observations=int(tr.get("total_observations") or 0),
        )

    def _sort_weight(inst: AssetInstance) -> int:
        vib_n = inst.vibration_time_range.total_observations if inst.vibration_time_range else 0
        return vib_n

    instances.sort(
        key=lambda inst: (
            -_sort_weight(inst),
            inst.asset_id.lower(),
        ),
    )
    return sorted(vib_union)


def _build_failure_sensor_grounding(
    asset_name: str,
    iot_sensors: list[str],
    vibration_sensors: list[str],
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    raw_modes = _as_mapping(get_failure_modes(asset_class=asset_name))
    failure_modes = list(dict.fromkeys(raw_modes.get("failure_modes", []) or []))
    if not failure_modes:
        return [], {}, {}
    sensors = sorted(
        dict.fromkeys(
            str(s).strip()
            for s in (*iot_sensors, *vibration_sensors)
            if str(s).strip()
        )
    )

    if not sensors:
        return failure_modes, {}, {}

    cached = _load_failure_mapping(asset_name)
    if cached is not None:
        fm2sensor, sensor2fm = cached
        return failure_modes, fm2sensor, sensor2fm

    _log.info("No cached failure sensor mapping for %s; mapping generation is disabled", asset_name)
    return failure_modes, {}, {}


def discover_grounding(asset_name: str, requested_open_form: bool = False) -> GroundingBundle:
    if not requested_open_form:
        return GroundingBundle(asset_name=asset_name, requested_open_form=False)

    asset_instances, iot_sensor_names = _discover_iot_inventory_and_instances(asset_name)

    vib_by_key = _vibration_by_site_asset()
    vibration_sensor_names = _overlay_vibration(asset_instances, vib_by_key)

    if not asset_instances:
        return GroundingBundle(
            asset_name=asset_name,
            requested_open_form=True,
            open_form_eligible=False,
        )

    failure_modes, failure_sensor_mapping, sensor_failure_mapping = _build_failure_sensor_grounding(
        asset_name=asset_name,
        iot_sensors=iot_sensor_names,
        vibration_sensors=vibration_sensor_names,
    )

    return GroundingBundle(
        asset_name=asset_name,
        requested_open_form=True,
        open_form_eligible=True,
        iot_sensors=iot_sensor_names,
        vibration_sensors=vibration_sensor_names,
        asset_instances=asset_instances,
        failure_modes=failure_modes,
        failure_sensor_mapping=failure_sensor_mapping,
        sensor_failure_mapping=sensor_failure_mapping,
    )


__all__ = ["discover_grounding"]
