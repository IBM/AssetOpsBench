"""Failure mode and sensor reasoning for industrial asset classes."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Union

import couchdb3
from couchdb3.exceptions import NotFoundError
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── Database stores ───────────────────────────────────────────────────────────
# Under AssetOpsBench's loader, database name == collection key. Failure modes
# live in the 'failure_mode' database; generic catalog lookups live in the
# utilities MCP server.

COUCHDB_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME", "admin")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD", "password")
FAILURE_MODE_DBNAME = os.environ.get("FAILURE_MODE_DBNAME", "failure_mode")


def _connect(dbname):
    try:
        h = couchdb3.Database(
            dbname,
            url=COUCHDB_URL,
            user=COUCHDB_USERNAME,
            password=COUCHDB_PASSWORD,
        )
        logger.info("Connected to database: %s", dbname)
        return h
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to connect to database '%s': %s", dbname, e)
        return None


fm_db = _connect(FAILURE_MODE_DBNAME)


def _asset_class_key(asset_class: str) -> str:
    """Normalise an asset class to the database key format ('Hydraulic_Pump' -> 'hydraulic pump')."""
    key = re.sub(r"\d+", "", asset_class or "")
    key = re.sub(r"[_\-]+", " ", key)
    return re.sub(r"\s+", " ", key).strip().lower()


def _known_asset_classes(limit: int = 10) -> List[str]:
    """Return known asset classes from the failure_mode collection for error guidance."""
    if not fm_db:
        return []
    try:
        res = fm_db.find({}, fields=["asset_class"], limit=limit)
    except Exception:  # noqa: BLE001
        return []
    classes = [
        doc.get("asset_class")
        for doc in res.get("docs", [])
        if isinstance(doc.get("asset_class"), str) and doc.get("asset_class")
    ]
    return sorted(dict.fromkeys(classes))


def _missing_asset_class_error(original: str, normalized: str) -> ErrorResult:
    message = (
        f"no failure_mode record for asset_class '{normalized}' in database. "
        f"Input was normalized from {original!r}; check that asset_class matches a stored class."
    )
    known = _known_asset_classes()
    if known:
        message += f" Available asset_class values include: {', '.join(known)}."
    return ErrorResult(error=message)


def _is_not_found_error(exc: Exception) -> bool:
    if isinstance(exc, (KeyError, NotFoundError)):
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(
        exc, "status_code", None
    )
    return status_code == 404


# ── Prompt templates ──────────────────────────────────────────────────────────

_RELEVANCY_MATRIX_PROMPT = (
    "Asset class: {asset_class}\n"
    "Evaluate whether each sensor can help monitor or detect each failure mode "
    "for assets of this class.\n\n"
    "Failure modes:\n{failure_modes}\n\n"
    "Sensors:\n{sensors}\n\n"
    "Return only valid JSON as an array with exactly {pair_count} objects, one "
    "object for every failure-mode/sensor pair. Use the exact input strings. "
    'Each object must have keys "failure_mode", "sensor", "answer", and "reason". '
    'The "answer" value must be one of "Yes", "No", or "Unknown".'
)

_FAILURE_MODE_PROMPT = (
    "List up to {max_modes} common failure modes for asset class {asset_class}.\n"
    "Return only failure mode names, one per line."
)

_FAILURE_MODE_EXTEND_PROMPT = (
    "Asset class {asset_class} already has these stored failure modes:\n{stored_modes}\n\n"
    "List up to {max_modes} additional failure modes for asset class {asset_class} "
    "that are not already in the stored list.\n"
    "Return only failure mode names, one per line."
)


def _parse_failure_mode_list(text: str) -> List[str]:
    items: List[str] = []
    for line in text.splitlines():
        item = line.strip()
        if not item:
            continue
        item = re.sub(r"^\s*(?:[-*•]|\d+[\.\)])\s*", "", item).strip()
        if item:
            items.append(item)
    return items


def _parse_relevancy(text: str) -> dict:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("yes"):
        answer = "Yes"
    elif lines and lines[0].lower().startswith("no"):
        answer = "No"
    else:
        answer = "Unknown"
    reason = lines[1] if len(lines) >= 2 else "Unknown"
    return {"answer": answer, "reason": reason}


def _normalize_relevancy_answer(answer: object) -> str:
    if isinstance(answer, bool):
        return "Yes" if answer else "No"
    normalized = str(answer or "").strip().lower()
    if normalized.startswith("yes"):
        return "Yes"
    if normalized.startswith("no"):
        return "No"
    return "Unknown"


def _normalize_mapping_label(label: object) -> str:
    return re.sub(r"\s+", " ", str(label or "")).strip().lower()


def _load_json_payload(text: str) -> object:
    content = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.I
    )
    if fenced:
        content = fenced.group(1).strip()
    else:
        starts = [
            idx for idx in (content.find("["), content.find("{")) if idx != -1
        ]
        end = max(content.rfind("]"), content.rfind("}"))
        if starts and end > min(starts):
            content = content[min(starts) : end + 1]
    return json.loads(content)


def _record_value(record: dict, *keys: str) -> object:
    lower_keys = {str(key).lower(): key for key in record}
    for key in keys:
        if key in record:
            return record[key]
        actual_key = lower_keys.get(key.lower())
        if actual_key is not None:
            return record[actual_key]
    return None


def _extract_matrix_records(payload: object) -> List[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "mappings", "full_relevancy", "relevancy"):
        if isinstance(payload.get(key), list):
            return [item for item in payload[key] if isinstance(item, dict)]
    if _record_value(payload, "failure_mode", "failureMode", "fm") is not None:
        return [payload]

    records: List[dict] = []
    for failure_mode, sensors in payload.items():
        if not isinstance(sensors, dict):
            continue
        for sensor, value in sensors.items():
            if isinstance(value, dict):
                records.append(
                    {
                        "failure_mode": failure_mode,
                        "sensor": sensor,
                        **value,
                    }
                )
            else:
                records.append(
                    {
                        "failure_mode": failure_mode,
                        "sensor": sensor,
                        "answer": value,
                    }
                )
    return records


def _parse_relevancy_matrix(
    text: str, failure_modes: List[str], sensors: List[str]
) -> Dict[tuple[str, str], dict]:
    try:
        payload = _load_json_payload(text)
    except json.JSONDecodeError:
        if len(failure_modes) == 1 and len(sensors) == 1:
            return {(failure_modes[0], sensors[0]): _parse_relevancy(text)}
        raise

    records = _extract_matrix_records(payload)
    if not records:
        raise ValueError("LLM response did not contain relevancy records")

    expected_fms = {
        _normalize_mapping_label(failure_mode): failure_mode
        for failure_mode in failure_modes
    }
    expected_sensors = {
        _normalize_mapping_label(sensor): sensor for sensor in sensors
    }
    results = {
        (failure_mode, sensor): {
            "answer": "Unknown",
            "reason": "LLM response omitted this pair.",
        }
        for sensor in sensors
        for failure_mode in failure_modes
    }
    for record in records:
        failure_mode = expected_fms.get(
            _normalize_mapping_label(
                _record_value(record, "failure_mode", "failureMode", "fm")
            )
        )
        sensor = expected_sensors.get(
            _normalize_mapping_label(_record_value(record, "sensor", "sensor_name"))
        )
        if failure_mode is None or sensor is None:
            continue
        reason = (
            _record_value(record, "reason", "relevancy_reason", "relevancyReason")
            or "Unknown"
        )
        answer = _record_value(record, "answer", "relevancy_answer", "relevancyAnswer")
        if answer is None:
            answer = _record_value(record, "detects", "relevant")
        results[(failure_mode, sensor)] = {
            "answer": _normalize_relevancy_answer(answer),
            "reason": str(reason).strip() or "Unknown",
        }
    return results


# ── LLM backend (lazy init; graceful degradation if creds are absent) ─────────

_DEFAULT_MODEL_ID = "watsonx/meta-llama/llama-3-3-70b-instruct"
_MAX_RETRIES = 3
_MAX_MAPPING_FAILURE_MODES = 5
_MAX_MAPPING_SENSORS = 20
_MAX_MAPPING_PAIRS = 100
_MODEL_ID = os.environ.get("FMSR_MODEL_ID", _DEFAULT_MODEL_ID)


def _build_llm():
    from llm import make_backend

    if _MODEL_ID.startswith("watsonx/"):
        missing = [
            v for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID") if not os.environ.get(v)
        ]
        if missing:
            raise RuntimeError(f"Missing env vars for WatsonX: {missing}")
    elif _MODEL_ID.startswith("tokenrouter/"):
        missing = [
            v
            for v in ("TOKENROUTER_API_KEY", "TOKENROUTER_BASE_URL")
            if not os.environ.get(v)
        ]
        if missing:
            raise RuntimeError(f"Missing env vars for TokenRouter: {missing}")
    else:
        missing = [
            v for v in ("LITELLM_API_KEY", "LITELLM_BASE_URL") if not os.environ.get(v)
        ]
        if missing:
            raise RuntimeError(f"Missing env vars for LiteLLM: {missing}")
    return make_backend(_MODEL_ID)


try:
    _llm = _build_llm()
    _llm_available = True
except Exception as _e:  # noqa: BLE001
    logger.warning("LLM unavailable (generate_* tools disabled): %s", _e)
    _llm = None
    _llm_available = False


def _call_relevancy_matrix(
    asset_class: str, failure_modes: List[str], sensors: List[str]
) -> Dict[tuple[str, str], dict]:
    prompt = _RELEVANCY_MATRIX_PROMPT.format(
        asset_class=asset_class,
        failure_modes="\n".join(f"- {failure_mode}" for failure_mode in failure_modes),
        sensors="\n".join(f"- {sensor}" for sensor in sensors),
        pair_count=len(failure_modes) * len(sensors),
    )
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_relevancy_matrix(_llm.generate(prompt), failure_modes, sensors)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc


def _call_failure_mode_generation(
    asset_class: str, known: List[str], max_modes: int
) -> List[str]:
    prompt = (
        _FAILURE_MODE_EXTEND_PROMPT.format(
            asset_class=asset_class,
            stored_modes="\n".join(f"- {mode}" for mode in known),
            max_modes=max_modes,
        )
        if known
        else _FAILURE_MODE_PROMPT.format(asset_class=asset_class, max_modes=max_modes)
    )
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_failure_mode_list(_llm.generate(prompt))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc


# ── Result models ─────────────────────────────────────────────────────────────


class ErrorResult(BaseModel):
    error: str


class FailureModesResult(BaseModel):
    asset_class: str
    failure_modes: List[str]
    exhaustive: bool = False  # the stored list is not claimed to be complete
    source: Optional[str] = None  # provenance: ISO / curated / LLM:<model>


class GenerateFailureModesResult(BaseModel):
    asset_class: str
    known: List[str]
    generated: List[str]
    failure_modes: List[str]
    source: str
    message: str


class AddFailureModesResult(BaseModel):
    asset_class: str
    added: List[str]
    failure_modes: List[str]
    total: int
    exhaustive: bool
    source: Optional[str] = None
    message: str


class RelevancyEntry(BaseModel):
    asset_class: str
    failure_mode: str
    sensor: str
    relevancy_answer: str
    relevancy_reason: str


class MappingMetadata(BaseModel):
    asset_class: str
    failure_modes: List[str]
    sensors: List[str]


class FailureModeSensorMappingResult(BaseModel):
    metadata: MappingMetadata
    fm2sensor: Dict[str, List[str]]
    sensor2fm: Dict[str, List[str]]
    full_relevancy: List[RelevancyEntry]


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "fmsr",
    instructions="Failure mode and sensor reasoning for industrial asset classes.",
)


@mcp.tool(title="Get Failure Modes")
def get_failure_modes(asset_class: str) -> Union[FailureModesResult, ErrorResult]:
    """READ the known failure modes for an asset class.

    Args:
        asset_class: Asset class to look up, such as "pump". Case, whitespace,
            digits, underscores, and hyphens are normalized before querying.
    """
    raw_asset_class = asset_class
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    try:
        d = _find_failure_mode_doc(key)
        if d is None:
            return _missing_asset_class_error(raw_asset_class, key)
        return FailureModesResult(
            asset_class=d.get("asset_class", key),
            failure_modes=d.get("failure_modes", []),
            exhaustive=d.get("exhaustive", False),
            source=d.get("source"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("get_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


def _find_failure_mode_doc(asset_class: str) -> Optional[dict]:
    """Return the stored failure-mode doc for an asset class, or None."""
    if not fm_db:
        raise RuntimeError("database not connected")
    key = _asset_class_key(asset_class)
    try:
        d = fm_db.get(f"fm:{key}", check=True)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found_error(exc):
            d = None
        else:
            raise RuntimeError(
                f"database lookup failed for asset_class '{key}': {exc}"
            ) from exc
    try:
        if d is None:
            res = fm_db.find({"asset_class": key}, limit=1)
            docs = res["docs"]
            if docs:
                d = docs[0]
        return d
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"database lookup failed for asset_class '{key}': {exc}"
        ) from exc


def _known_failure_modes(asset_class: str) -> List[str]:
    """Return stored failure modes for an asset class, or [] if none are available."""
    d = _find_failure_mode_doc(asset_class)
    if d is None:
        return []
    return [
        mode.strip()
        for mode in d.get("failure_modes", [])
        if isinstance(mode, str) and mode.strip()
    ]


@mcp.tool(title="Generate Failure Modes")
def generate_failure_modes(
    asset_class: str,
    max_modes: int = 10,
) -> Union[GenerateFailureModesResult, ErrorResult]:
    """GENERATE a new or extended failure-mode list for an asset class.

    This tool does not write to the database. If the normalized `asset_class`
    exists in the database, the current stored failure modes are used as context
    and the LLM generates additional modes. If no stored modes exist, the LLM
    generates a new list from scratch.

    Args:
        asset_class: Asset class to reason about, such as "pump". Case,
            whitespace, digits, underscores, and hyphens are normalized before
            prompting the LLM.
        max_modes: Maximum number of new failure modes to request from the LLM.
    """
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    if max_modes <= 0:
        return ErrorResult(error="max_modes must be greater than 0")
    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    try:
        base = _known_failure_modes(key)
        base = [mode.strip() for mode in base if mode and mode.strip()]
        raw = _call_failure_mode_generation(key, base, max_modes)
        seen = {mode.lower() for mode in base}
        generated: List[str] = []
        for mode in raw:
            candidate = mode.strip()
            normalized = candidate.lower()
            if candidate and normalized not in seen:
                seen.add(normalized)
                generated.append(candidate)
        if len(generated) > max_modes:
            generated = generated[:max_modes]
        return GenerateFailureModesResult(
            asset_class=key,
            known=base,
            generated=generated,
            failure_modes=base + generated,
            source=f"LLM:{_MODEL_ID}",
            message=(
                f"generated {len(generated)} new failure mode(s) for asset_class '{key}' "
                f"using {len(base)} stored mode(s) as context; nothing was persisted."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Add Failure Modes")
def add_failure_modes(
    asset_class: str,
    failure_modes: List[str],
    exhaustive: Optional[bool] = None,
    source: Optional[str] = None,
) -> Union[AddFailureModesResult, ErrorResult]:
    """WRITE failure modes for an asset class to the database.

    Existing modes are preserved, incoming modes are merged case-insensitively,
    and only newly added modes are reported. Use this after curated or generated
    modes should become available to future `get_failure_modes` calls.

    Args:
        asset_class: Asset class to update, such as "pump". Case, whitespace,
            digits, underscores, and hyphens are normalized before writing.
        failure_modes: Failure modes to add for the asset class.
        exhaustive: Set true only when the stored list is believed complete. If
            omitted, the existing value is preserved; new records default false.
        source: Optional provenance for the stored list.
    """
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    incoming = [
        mode.strip()
        for mode in (failure_modes or [])
        if isinstance(mode, str) and mode.strip()
    ]
    if not incoming:
        return ErrorResult(error="failure_modes list is required")
    if not fm_db:
        return ErrorResult(error="database not connected")

    try:
        doc_id = f"fm:{key}"
        doc = _find_failure_mode_doc(key)
        existing = [
            mode.strip()
            for mode in (doc or {}).get("failure_modes", [])
            if isinstance(mode, str) and mode.strip()
        ]
        seen = set()
        merged: List[str] = []
        for mode in existing:
            normalized = mode.lower()
            if normalized not in seen:
                seen.add(normalized)
                merged.append(mode)

        added: List[str] = []
        for mode in incoming:
            normalized = mode.lower()
            if normalized not in seen:
                seen.add(normalized)
                merged.append(mode)
                added.append(mode)

        if doc is None:
            doc = {"_id": doc_id, "asset_class": key}
            stored_exhaustive = False
        else:
            doc.setdefault("_id", doc_id)
            doc["asset_class"] = key
            stored_exhaustive = bool(doc.get("exhaustive", False))
        doc["failure_modes"] = merged
        doc["exhaustive"] = stored_exhaustive if exhaustive is None else exhaustive
        doc["source"] = source or doc.get("source") or "user"
        fm_db.save(doc)

        return AddFailureModesResult(
            asset_class=key,
            added=added,
            failure_modes=merged,
            total=len(merged),
            exhaustive=doc["exhaustive"],
            source=doc.get("source"),
            message=(
                f"added {len(added)} new failure mode(s) to asset_class '{key}' "
                f"({len(merged)} total)."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("add_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Generate Failure Mode Sensor Mapping")
def generate_failure_mode_sensor_mapping(
    asset_class: str,
    failure_modes: List[str],
    sensors: List[str],
) -> Union[FailureModeSensorMappingResult, ErrorResult]:
    """GENERATE whether each sensor can detect each failure mode.

    Uses a single full-matrix LLM call and returns a bidirectional mapping
    (fm→sensors, sensor→fms) plus per-pair relevancy details. Inputs are capped
    to keep this one tool call bounded.

    Args:
        asset_class: Asset class to reason about, such as "pump". Case, whitespace,
            digits, underscores, and hyphens are normalized before prompting the LLM.
        failure_modes: Failure modes for the asset class.
        sensors: Sensor names to evaluate for detection relevance.
    """
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    if not failure_modes:
        return ErrorResult(error="failure_modes list is required")
    if not sensors:
        return ErrorResult(error="sensors list is required")
    if len(failure_modes) > _MAX_MAPPING_FAILURE_MODES:
        return ErrorResult(
            error=(
                "failure_modes list is too large; "
                f"provide at most {_MAX_MAPPING_FAILURE_MODES} failure modes"
            )
        )
    if len(sensors) > _MAX_MAPPING_SENSORS:
        return ErrorResult(
            error=(
                "sensors list is too large; "
                f"provide at most {_MAX_MAPPING_SENSORS} sensors"
            )
        )
    total_pairs = len(failure_modes) * len(sensors)
    if total_pairs > _MAX_MAPPING_PAIRS:
        return ErrorResult(
            error=(
                "failure_mode/sensor matrix is too large; "
                f"provide at most {_MAX_MAPPING_PAIRS} total pairs"
            )
        )
    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    full_relevancy: List[RelevancyEntry] = []
    fm2sensor: Dict[str, List[str]] = {}
    sensor2fm: Dict[str, List[str]] = {}
    try:
        pair_relevancy = _call_relevancy_matrix(key, failure_modes, sensors)
        for sensor in sensors:
            for fm in failure_modes:
                gen = pair_relevancy.get(
                    (fm, sensor),
                    {
                        "answer": "Unknown",
                        "reason": "LLM response omitted this pair.",
                    },
                )
                full_relevancy.append(
                    RelevancyEntry(
                        asset_class=key,
                        failure_mode=fm,
                        sensor=sensor,
                        relevancy_answer=gen["answer"],
                        relevancy_reason=gen["reason"],
                    )
                )
                if "yes" in gen["answer"].lower():
                    fm2sensor.setdefault(fm, []).append(sensor)
                    sensor2fm.setdefault(sensor, []).append(fm)
    except Exception as exc:  # noqa: BLE001
        logger.error("relevancy matrix generation failed: %s", exc)
        return ErrorResult(error=str(exc))

    return FailureModeSensorMappingResult(
        metadata=MappingMetadata(
            asset_class=key, failure_modes=failure_modes, sensors=sensors
        ),
        fm2sensor=fm2sensor,
        sensor2fm=sensor2fm,
        full_relevancy=full_relevancy,
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
