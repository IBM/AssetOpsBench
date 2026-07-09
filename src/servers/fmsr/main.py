"""FMSR (Failure Mode and Sensor Reasoning) MCP Server.

Tools:
  get_failure_modes                    – READ the (partial) failure-mode list for an asset class from CouchDB
  generate_failure_modes               – GENERATE a new/extended failure-mode list via the LLM without writing CouchDB
  generate_failure_mode_sensor_mapping – GENERATE the bidirectional FM↔sensor relevancy via the LLM

Failure modes live in CouchDB (collection "failure_mode", one doc per asset class,
loaded from scenario manifests). Coverage is NOT exhaustive: docs carry
`exhaustive: false`, so get_failure_modes returns what is known. Retrieval (get_*),
failure-mode generation, and sensor-mapping generation are separate by design.

LLM backend is configured via FMSR_MODEL_ID (default: watsonx/meta-llama/llama-3-3-70b-instruct).
CouchDB via COUCHDB_URL / FAILURE_MODE_DBNAME /
COUCHDB_USERNAME / COUCHDB_PASSWORD.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Union

from concurrent.futures import ThreadPoolExecutor, as_completed

import couchdb3
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── CouchDB stores ────────────────────────────────────────────────────────────
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
        logger.info("Connected to CouchDB: %s", dbname)
        return h
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to connect to CouchDB '%s': %s", dbname, e)
        return None


fm_db = _connect(FAILURE_MODE_DBNAME)


def _asset_class_key(asset_class: str) -> str:
    """Normalise an asset class to the CouchDB key format ('Hydraulic_Pump' -> 'hydraulic pump')."""
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
        f"no failure_mode record for asset_class '{normalized}' in DB. "
        f"Input was normalized from {original!r}; check that asset_class matches a stored class."
    )
    known = _known_asset_classes()
    if known:
        message += f" Available asset_class values include: {', '.join(known)}."
    return ErrorResult(error=message)


# ── Prompt templates ──────────────────────────────────────────────────────────

_RELEVANCY_PROMPT = (
    "For asset class {asset_class}, if the failure {failure_mode} occurs, "
    "can sensor {sensor} help monitor or detect the failure for assets of class {asset_class}?\n"
    "Provide the answer in the first line and reason in the second line."
)

_FAILURE_MODE_PROMPT = (
    "List up to {max_modes} common failure modes for asset class {asset_class}.\n"
    "Return only failure mode names, one per line."
)

_FAILURE_MODE_EXTEND_PROMPT = (
    "Asset class {asset_class} already has these known failure modes:\n{known}\n\n"
    "List up to {max_modes} additional failure modes for asset class {asset_class} "
    "that are not already in the known list.\n"
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


# ── LLM backend (lazy init; graceful degradation if creds are absent) ─────────

_DEFAULT_MODEL_ID = "watsonx/meta-llama/llama-3-3-70b-instruct"
_MAX_RETRIES = 3
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


def _call_relevancy(asset_class: str, failure_mode: str, sensor: str) -> dict:
    prompt = _RELEVANCY_PROMPT.format(
        asset_class=asset_class, failure_mode=failure_mode, sensor=sensor
    )
    last_exc: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            return _parse_relevancy(_llm.generate(prompt))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc


def _call_failure_mode_generation(
    asset_class: str, known: List[str], max_modes: int
) -> List[str]:
    prompt = (
        _FAILURE_MODE_EXTEND_PROMPT.format(
            asset_class=asset_class,
            known="\n".join(f"- {mode}" for mode in known),
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
    instructions=(
        "Failure mode and sensor reasoning. get_failure_modes READS an asset class's (possibly partial) "
        "failure modes; generate_failure_modes GENERATES a new or extended list without writing the DB; "
        "generate_failure_mode_sensor_mapping GENERATES which sensors can detect each failure. "
        "Use the utilities MCP server for asset, sensor, and failure-mode catalog lookups."
    ),
)


@mcp.tool(title="Get Failure Modes")
def get_failure_modes(asset_class: str) -> Union[FailureModesResult, ErrorResult]:
    """READ the known failure modes for an asset class.

    Args:
        asset_class: Asset class to look up, such as "pump". Case, whitespace,
            underscores, and hyphens are normalized before querying.
    """
    raw_asset_class = asset_class
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    if not fm_db:
        return ErrorResult(error="CouchDB not connected")
    try:
        try:
            d = fm_db.get(f"fm:{key}")
        except Exception:  # noqa: BLE001
            d = None
        if d is None:
            res = fm_db.find({"asset_class": key}, limit=1)
            docs = res["docs"]
            if docs:
                d = docs[0]
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


def _known_failure_modes(asset_class: str) -> List[str]:
    """Return stored failure modes for an asset class, or [] if none are available."""
    if not fm_db:
        return []
    key = _asset_class_key(asset_class)
    try:
        try:
            d = fm_db.get(f"fm:{key}")
        except Exception:  # noqa: BLE001
            d = None
        if d is None:
            res = fm_db.find({"asset_class": key}, limit=1)
            docs = res["docs"]
            if docs:
                d = docs[0]
        if d is None:
            return []
        return [
            mode.strip()
            for mode in d.get("failure_modes", [])
            if isinstance(mode, str) and mode.strip()
        ]
    except Exception:  # noqa: BLE001
        return []


@mcp.tool(title="Generate Failure Modes")
def generate_failure_modes(
    asset_class: str,
    max_modes: int = 10,
) -> Union[GenerateFailureModesResult, ErrorResult]:
    """GENERATE a new or extended failure-mode list for an asset class.

    This tool does not write to CouchDB. If the normalized `asset_class` exists
    in CouchDB, the current stored failure modes are used as context and the LLM
    generates additional modes. If no stored modes exist, the LLM generates a
    new list from scratch.

    Args:
        asset_class: Asset class to reason about, such as "pump". Case,
            whitespace, underscores, and hyphens are normalized before prompting
            the LLM.
        max_modes: Maximum number of new failure modes to request from the LLM.
    """
    key = _asset_class_key(asset_class)
    if not key or key == "none":
        return ErrorResult(error="asset_class is required")
    if max_modes <= 0:
        return ErrorResult(error="max_modes must be greater than 0")
    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    base = _known_failure_modes(key)
    base = [mode.strip() for mode in base if mode and mode.strip()]

    try:
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
                f"using {len(base)} known mode(s) as context; nothing was persisted."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("generate_failure_modes failed: %s", exc)
        return ErrorResult(error=str(exc))


@mcp.tool(title="Generate Failure Mode Sensor Mapping")
def generate_failure_mode_sensor_mapping(
    asset_class: str,
    failure_modes: List[str],
    sensors: List[str],
) -> Union[FailureModeSensorMappingResult, ErrorResult]:
    """GENERATE, for each (failure_mode, sensor) pair, whether the sensor can detect the failure
    (one LLM call per pair). Returns a bidirectional mapping (fm→sensors, sensor→fms) plus per-pair
    relevancy details. Keep both lists small (e.g. ≤5 failure modes, ≤10 sensors) to bound runtime.

    Args:
        asset_class: Asset class to reason about, such as "pump". Case, whitespace,
            underscores, and hyphens are normalized before prompting the LLM.
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
    if not _llm_available:
        return ErrorResult(error="LLM unavailable")

    full_relevancy: List[RelevancyEntry] = []
    fm2sensor: Dict[str, List[str]] = {}
    sensor2fm: Dict[str, List[str]] = {}
    try:
        pairs = [(s, fm) for s in sensors for fm in failure_modes]
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_call_relevancy, key, fm, s): (s, fm)
                for s, fm in pairs
            }
            for future in as_completed(futures):
                s, fm = futures[future]
                gen = future.result()
                full_relevancy.append(
                    RelevancyEntry(
                        asset_class=key,
                        failure_mode=fm,
                        sensor=s,
                        relevancy_answer=gen["answer"],
                        relevancy_reason=gen["reason"],
                    )
                )
                if "yes" in gen["answer"].lower():
                    fm2sensor.setdefault(fm, []).append(s)
                    sensor2fm.setdefault(s, []).append(fm)
    except Exception as exc:  # noqa: BLE001
        logger.error("_call_relevancy failed: %s", exc)
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
